import sh
import re
import json
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path
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
        self.mode = mode
    
    def set_hls_incl(self, hls_incl_path:str|Path):
        """ Provides path of HLS tool's include folder """

        self.hls_incl_path = Path(hls_incl_path)

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

    def utilization(self) -> dict[str,tuple[int,int]]:
        """ Parses the HLS report for resource utilization.
        
        :param hls_rpt_path: """

        # table_pattern = re.compile(
        #     r'(?m)'
        #     r'^={24,}\s*\n'
        #     r'^={2}\sUtilization\ Estimates\s*\n'
        #     r'^={24,}\s*\n'
        #     r'^\*\ Summary:\s*\n'
        #     r'^\+(?:-+\+)+\s*\n'
        #     r'(?P<body>(?:(?!\n\n).|\n)*?)'
        #     r'^\+(?:-+\+)+\s*\n'
        #     r'^\n'
        # )

        # total_line_pattern = re.compile(r'^|Total.')
        # available_line_pattern = re.compile(r'^\|Available.')

        # with open(self.input_dir / 'cordic_csynth.rpt', 'r') as f:
        #     hls_report_str = f.read()
        
        # table_match = table_pattern.search(hls_report_str)
        # total_match = total_line_pattern.search(table_match.group('body'))
        # available_match = available_line_pattern.search(table_match.group('body'))
        # return total_match.groups()


    def utilization(self) -> dict[str, tuple[int, int]]:
        """ Parse HLS report text to extract utilization (Total/Available) per resource
        
        :return: Dictionary mapping resource name to utilization,
                e.g., {'BRAM_18K':(0,280), 'LUT':(5214,53200), ...} """
        
        hls_rpt_path = next(self.input_dir.glob('*.rpt'), None)
        hls_rpt = hls_rpt_path.read_text(encoding='utf-8', errors='ignore')

        # Find utilization table
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
        m = table_pat.search(hls_rpt)
        if not m:
            raise ValueError('Utilization table not found')

        table = m.group('table')

        def find_line(prefix:str) -> str:
            pat = rf'^\|[ \t]*{prefix}\b[^\n]*$'
            line_match = re.search(pat, table, re.MULTILINE)
            if not line_match:
                raise ValueError(f"Row starting with '{prefix}' not found")
            return line_match.group(0)

        def parse_line(line: str) -> list[int]:
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            # Drop the row header
            vals = cells[1:]
            retlst = []
            for cell in vals:
                retlst.append(int(cell) if cell.isdigit() else cell)
            return retlst

        # Assume 'Name' header is always present in table
        header_line = find_line('Name')
        total_line = find_line('Total')
        avail_line = find_line('Available')
        resources = parse_line(header_line)
        totals = parse_line(total_line)
        avails = parse_line(avail_line)

        result = {res:(totals[i],avails[i]) for i, res in enumerate(resources)}
        return result
