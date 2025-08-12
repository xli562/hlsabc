import sh
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

    

        
