import sh
import re
import json
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path
from utils.agent import Agent, GPRO, GFLASH, GLITE, GEMINI
from utils.xlogging import get_logger


logger = get_logger()

class PerfFB:
    """ Gives performance feedback based on ground truth and HLS report. """

    # Modes of operation
    PRECISION = 'p'
    AREA = 'a'
    THROUGHPUT = 't'
    SUGGESTION = 's'

    def __init__(self, input_dir:str|Path=None, 
                 mode:tuple[str,...]=(PRECISION, AREA, THROUGHPUT, SUGGESTION)):
        """ Initializes module.
        
        :param input_dir: (Optional) path of the input directory, default None.
        :param mode: (Optional) specifies which information to be
                returned.
        """

        self.input_dir = Path(input_dir) if input_dir else None
        if self.input_dir:
            self.read_hls_report()
        self.mode = mode
    
    def set_hls_incl_dir(self, hls_incl_path:str|Path):
        """ Provides path of HLS tool's include folder """

        self.hls_incl_path = Path(hls_incl_path)
    
    def read_hls_report(self):
        hls_rpt_path = next(self.input_dir.glob('*.rpt'), None)
        self.hls_rpt = hls_rpt_path.read_text(encoding='utf-8', errors='ignore')

    def _run_c(self, c_args_lst:list[list]) -> list[list[float]]:
        """ Runs `test.cpp`, returns tuple of returned comma-separated values.
        
        :param c_args_lst: groups of arguments fed to the C program.
        
        :return: returned floats from C """

        with TemporaryDirectory() as tmp_dir:
            # Compile
            tmp_dir = Path(tmp_dir)
            cpp_pathstr = str(self.input_dir / 'cordic.cpp')
            cpp_test_pathstr = str(self.input_dir / 'test.cpp')
            hls_incl_pathstr = str(self.hls_incl_path)
            exe_pathstr = str(tmp_dir / 'main')
            sh.Command('g++')(
                    '-g',
                    '-I', hls_incl_pathstr,
                    '-D', 'FIXED_TYPE',
                    cpp_pathstr, cpp_test_pathstr,
                    '-o', exe_pathstr,
                    _tty_out=False
            )
            exe = sh.Command(exe_pathstr)

            # Run
            outs = []
            for c_args in c_args_lst:
                out_tup = exe(*c_args).strip().split(',')
                outs.append([float(x) for x in out_tup])
            
            return outs
    
    def _rmse(self, gt:dict, fut):
        """ Returns RMSE relative to specified ground truth
        
        :param gt: ground truth
        :param fut: a function with the same IO as self._run_c """

        args_lst = []
        exps_lst = []
        for args, exps in gt.items():
            args_lst.append([float(x) for x in args.split(',')])
            exps_lst.append([float(x) for x in exps.split(',')])

        gots_lst = fut(args_lst)
        gots_arr = np.asarray(gots_lst)
        exps_arr = np.asarray(exps_lst)

        rmse = np.sqrt(np.sum(((exps_arr-gots_arr)/exps_arr)**2, axis=0) / (exps_arr.shape[0]-1))

        return rmse

    def accuracy(self):
        """ Returns RMSE of values produced the HLS source """

        # TODO: Should have another datapath for e.g. num_iters
        #       for design-space exploration.
        with open(self.input_dir / 'ground_truth.json', 'r') as f:
            gt:dict = json.load(f)
        
        return self._rmse(gt, self._run_c)

    def _find_line(self, table:str, header:str) -> list[int]:
        """ Finds a row with the specified header in a table in the HLS report.
        
        :param table: the data table
        :param header: row header
        
        :return: list of values of the matched line, from left to right.
        """
        pat = rf'^\|[ \t]*{header}\b[^\n]*$'
        line_match = re.search(pat, table, re.MULTILINE)
        if not line_match:
            raise ValueError(f"Row starting with '{header}' not found")
        line = line_match.group(0)

        # Assumes cells separated by '|'
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        vals_str = cells[1:]
        vals = []
        for cell in vals_str:
            vals.append(int(cell) if cell.isdigit() else cell)
        return vals

    def utilization(self) -> dict[str, tuple[int, int]]:
        """ Parses the HLS report for resource utilization.
        
        :return: Dictionary mapping resource name to utilization,
                e.g., {'BRAM_18K':(0,280), 'LUT':(5214,53200), ...} """
        
        table_pat = re.compile(
            r'''
            ={24,}[^\S\r\n]*\n
            ==[^\n]*Utilization[^\n]*\n
            ={24,}[^\S\r\n]*\n
            \*[\ \t]*Summary:[^\n]*\n
            (?P<table>.*?)
            (?=\n[ \t]*\n)
            ''',
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )
        table_match = table_pat.search(self.hls_rpt)
        if not table_match:
            raise ValueError('Utilization table not found')
        table = table_match.group('table')

        resources = self._find_line(table, 'Name')
        totals = self._find_line(table, 'Total')
        avails = self._find_line(table, 'Available')

        result = {res:(totals[i],avails[i]) for i, res in enumerate(resources)}
        return result

    def _latency(self) -> list[float]:
        """ Parse HLS report for latency. 
        
        :return: [min latency, max latency] in nanoseconds.
        """
        table_pat = re.compile(
            r'''
            \+\ Latency:\ \n
            \ {4}\*\ Summary:\ \n
            (?P<table>.*?)
            (?=\n[ \t]*\n)
            ''',
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )
        table_match = table_pat.search(self.hls_rpt)
        if not table_match:
            raise ValueError('Utilization table not found')

        table = table_match.group('table')

        rows = table.strip().split('\n')
        num_cells = [x.strip() for x in rows[-2].strip().strip('|').split('|')]
        latency_strs = num_cells[2:4]

        suffixes = [('ps', 1e-3), ('ns', 1e0), ('us', 1e3),
                    ('ms', 1e6), ('s', 1e9), ('ks', 1e12)]
        for i, latency_str in enumerate(latency_strs):
            for suffix, scale in suffixes:
                if latency_str.endswith(suffix):
                    latency_strs[i] = \
                            float(latency_str.rstrip(suffix).rstrip()) * scale
                    break
        
        return latency_strs

    def throughput(self):
        """ Return angles count per second. """

        return 1e9 / self._latency()

    def suggestion(self):
        self.agent = Agent(GPRO)
        self.agent.user_prompt = \
'''
You are an expert in high-level synthesis and hardware accelerator design. I am a professional software engineer, who is proficient with software skills and terminologies. However, I know little about hardware design. I wrote a software in C/C++ (see attached file), and wish to use high-level synthesis (HLS) synthesize a hardware accelerator for my software. I gave Vitis HLS my code as input, and got a synthesis report. 

Considering 1) accuracy, 2) resource usage, and 3) throughput, tell me a) how I can optimize my code to achieve Pareto optimality, and b) what are my design tradeoffs.

Style of your answer must be:

- Readable, short, concise and to-the-point
- Objective and unemotional, without words such as 'please', 'apologize', etc.
- Explains hardware concepts if necessary. Remember that I am a professional software engineer, and that you are an expert in high-level synthesis and hardware accelerator design.
'''
        self.agent.add_files(['input/cordic.cpp',
                              'input/cordic.h',
                              'input/cordic_csynth.rpt'])
        return self.agent.generate()

