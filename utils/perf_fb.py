import sh
import json
import numpy as np
import xml.etree.ElementTree as ET
from tempfile import TemporaryDirectory
from pathlib import Path
from utils.agent import Agent, GPRO, GFLASH, GLITE
from loguru import logger


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
    
    def set_hls_incl_dir(self, hls_incl_path:str|Path):
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

    def utilization(self) -> dict[str, tuple[int, int]]:
        """ Parses the .xml HLS report for resource utilization.
        
        :return: Dictionary mapping resource name to utilization,
                e.g., {'BRAM_18K':(0,280), 'LUT':(5214,53200), ...} """
        
        root = ET.parse(self.input_dir/'csynth.xml')
        area = root.find('AreaEstimates')
        usages = area.find('Resources')
        avails = area.find('AvailableResources')

        retdct = {}
        for resource in usages:
            resource_name = resource.tag
            usage = int(resource.text)
            avail = int(avails.find(resource_name).text)
            retdct[resource_name] = (usage, avail)
        
        return retdct

    def suggestion(self, clk_period:str=None):
        """ Provides detailed performance feedback and
        C/C++ source improvement suggestions using LLM.
        
        :param clk_period: (Optional) specifies clock 
                period, eg '10ns' or '400 ns' """

        self.agent = Agent(GPRO)

        if clk_period is None:
            clk_period = 'as specified by the target clock period in the HLS report'

        self.agent.system_prompt = \
'''
Style of your answer must be:

- Summarize the thought process to be readable, short, concise and to-the-point. Do not include *any* more text than absolutely necessary.
- Objective and unemotional, without words such as 'please', 'apologize', etc.
- Explains hardware concepts if necessary. Remember that I am a professional software engineer, and that you are an expert in high-level synthesis and hardware accelerator design.
'''
        self.agent.user_prompt = \
f'''
## Role introduction

You are an expert in high-level synthesis and hardware accelerator design. I am a professional software engineer, who is proficient with software skills and terminologies. However, I know little about hardware design.

A C/C++ software (see attached files) is fed into Vitis HLS, and csynth report(s) are generated (see attached), as well as code for a hardware accelerator for the software. Give estimations of high-level metrics of this accelerator.

## Methodology

- Throughput = 1 / Latency
- If the csynth report gives multiple latency values, it is because of branches in the control flow. Find the branching points, carefully examine them, considering the design's input and outputs, to give an estimate of the proportion of branch hits. Use this ratio to give a weighted-sum of the multiple latency values from the csynth report. Only fall back to uniform-weight averaging (or "Average-case Latency" provided by Vitis) as a last resort.
- For dataflow-optimized designs, if the intermediate values are vectors, the FIFOs between dataflow stages are implemented as dual-port BRAMs. Csynth reports over-optimistically estimate the dataflow latency as the max latency of all components. In reality, the dataflow pipeline is implemented as a PIPO that causes stalls. For large vectors, the dataflow optimization has trivial effects. For scalars, csynth gives the correct latency estimation for dataflow pipelines.
- **Make sure maths and unit conversion are done correctly.**
- The clock period is {clk_period}.

## Job description

Estimate the throughput for the synthesized hardware accelerator.
'''

#         self.agent.user_prompt = \
# r'''
# For this accelerator design in MLIR, answer the following questions.

# 1. Identify *one* core operation in the core kernel. For example, addition or MAC. MAC can count as a core operation.
# 2. How many times is this core operation performed?
# 3. How many bytes of *external* memory access are there?

# Hint: pay attention to the control flow, indicated by affine.for, affine.if, etc.

# Count explicitly, without any "educated guesses".

# Note size of the variables processed. For example, int32 is 32 bits, or 32/8=4 bytes.

# Exhaustive list of data movement commands:
# - affine.load
# - affine.store
# - memref.load
# - memref.store

# Note that some data movement are between on-chip buffers, these do not count as external memory access. Only count the ones that move data between the FPGA and the external memory.

# For example, in example.mlir:

# 1. The core operation is `arith.addi`
# 2. The core operation is performed $8 \times 8 \times 3 \times 3 \times 1 = 576$ times.
# 3. External memory access happens when the program performs any of the data movement commands on A or B.  That is $8 \times 8 \times (3 \times 3 \times (32 \div 8) + (32 \div 8)) = \pu{2560 bytes}$
# '''
        self.agent.user_prompt = 'What is the benchmark with the most complicated memory access pattern and data dependency?'
        # self.agent.add_files([self.input_dir/'conv_rpt'])
        self.agent.add_files(['/home/xl2296/allo/examples/polybench'])

        return self.agent.generate()

