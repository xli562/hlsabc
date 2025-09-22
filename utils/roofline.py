from pathlib import Path
import shutil
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt

from allo.customize import Schedule
from utils.agent import Agent, GPRO, GFLASH, GLITE
from utils.xlogging import logger


class Roofline:
    """ Gets roofline coordinates of Allo Schedule """

    def __init__(self, s:Schedule, input_dir:str|Path='input'):
        """ Initialization.
        
        :param s: Allo hardware schedule.
        :param input_dir: (optional) directory containing
                file attachments to the LLM
        """
        self.s = s
        self.file_input_dir = Path(input_dir)
        self.prj_name = 'none'
        self.eval_model = Agent(GPRO)       # Does performance estimation
        # self.critic_model = Agent(GPRO)     # Critisizes `eval_model`
        self.extract_model = Agent(GLITE)   # Extracts numbers from raw answers
        self.eval_model.system_prompt = \
'''
Style of your answer must be:

- Summarize the thought process to be readable, short, concise and to-the-point. Do not include *any* more text than absolutely necessary.
- Objective and unemotional, without words such as 'please', 'apologize', etc.
'''
        self.extract_model.system_prompt = \
'''
You are a tool that extracts machine-parsable values from a raw text. You *must* follow the example formats precisely, character-by-character. DO NOT REPEAT the bad examples (if any).
'''

    def _eval_extract(self, eval_prompt:str, extract_prompt:str,
                      output_convert, tries=3):
        """ Runs `self.eval_model`, then `self.extract_model`. 
                `model.files` are automatically added to the prompts.
        
        :param eval_prompt: overwrites the user prompt for `self.eval_model`.
        :param extract_prompt: overwrites the user prompt for 
                `self.extract_model`.
        :param output_convert: callable that converts the output string.
        :param tries: (optional) re-run count for `self.extract_model` before
                failing.

        :return: tuple of 1) the converted output, passed
                through output_convert. None if
                conversion fails; 2) raw output of `self.eval_model`.

        :raise warning: fails to produce answer that is convertible by
                `output_convert`.
        """
        self.eval_model.user_prompt = eval_prompt
        raw_output = self.eval_model.generate()
        extract_prompt += \
            f'\n\nHere is the raw text:\n\n{raw_output}'

        wrong_answers = set()
        for _ in range(tries):
            hint = ''
            if wrong_answers:
                hint = f'\n\nBad examples (*wrong* formats):\n{wrong_answers}'
            self.extract_model.user_prompt = extract_prompt + hint
            extract_output = self.extract_model.generate()
            try:
                return output_convert(extract_output.splitlines()[-1]), raw_output
            except Exception as e:
                wrong_answers.add(extract_output)
                logger.debug(f'Failed to convert extract_model output:\n{e}')

        logger.warning('Failed to produce a convertible answer.')
        
    def get_opcount_per_kernel(self) -> int:
        """ Returns the estimated number of operations per kernel.
        
        :return: estimated number of operations per kernel.
        """
        eval_prompt = \
r'''
For this accelerator design in MLIR, answer the following questions.

1. Identify *one* core operation in the core kernel. For example, addition or MAC. MAC can count as one core operation.
2. How many times is this core operation performed?

Hint: pay attention to the control flow, indicated by affine.for, affine.if, etc.

Count explicitly, without any "educated guesses".

For example, in example.mlir:

1. The core operation is `arith.addi`
2. The core operation is performed $8 \times 8 \times 3 \times 3 \times 1 = 576$ times.
'''
        eval_prompt += str(self.s.module)
        self.eval_model.add_files([self.file_input_dir / 'example.mlir'])

        extract_prompt = \
r'''
Return the core operation count as an integer at the exact *last* line of your answer.

Example input
-------------

### Summary of Analysis

The provided MLIR describes a General Matrix Multiplication (GEMM) operation, `C = A * B`, where A, B, and C are 1024x1024 matrices of `f32` (4-byte floats). The implementation uses a three-level nested loop structure (`i`, `k`, `j`) and an on-chip buffer (`%alloc_0`) to accumulate results for one row of the output matrix `C` at a time.

### Core Operation (`optm_gemm.mlir`)

1.  **Core operation**: `arith.addf`. This is the accumulation part of the multiply-accumulate (MAC) operation central to matrix multiplication.

2.  **Execution count**: The `arith.addf` operation is inside three nested loops, each iterating 1024 times.
    $1024 \times 1024 \times 1024 = 1024^3 = \pu{1073741824}$ times.

Example output
--------------

The core operation count is needed. The execution count of the core operation is needed. From the input, the execution count is 1073741824 times. The last line of this answer must be a machine-parsable integer, without any symbols, such as commas, whitespace, brackets, quotation marks, or backticks.

1073741824
'''
        return self._eval_extract(eval_prompt, extract_prompt,
                                  lambda x: int(x))[0]

    def get_bytes_per_kernel(self) -> int:
        """ Return the estimated bytes of external memory access per kernel.
        
        :return: estimated bytes of external memory access per kernel.
        """
        eval_prompt = \
r'''
For this accelerator design in MLIR, answer the following question.

How many bytes of *external* memory access are there?

Hint: pay attention to the control flow, indicated by affine.for, affine.if, etc.

Count explicitly, without any "educated guesses".

Note size of the variables processed. For example, int32 is 32 bits, or $32 \div 8 = \pu{4 bytes}$.

Exhaustive list of data movement commands:
- `affine.load`
- `affine.store`
- `memref.load`
- `memref.store`

Note that some data movement are between on-chip buffers, these do not count as external memory access. Only count the ones that move data between the FPGA and the external memory.

For example, in example.mlir:

External memory access happens when the program performs any of the data movement commands on `A` or `B`.  That is $8 \times 8 \times (3 \times 3 \times (32 \div 8) + (32 \div 8)) = \pu{2560 bytes}$
'''
        eval_prompt += str(self.s.module)
        self.eval_model.add_files([self.file_input_dir / 'example.mlir'])
        extract_prompt = \
r'''
Return the number of bytes of *external* memory access, as an integer, at the exact *last* line of your answer.

Example input
-------------

### Summary of Analysis

The provided MLIR describes a General Matrix Multiplication (GEMM) operation, `C = A * B`, where A, B, and C are 1024x1024 matrices of `f32` (4-byte floats). The implementation uses a three-level nested loop structure (`i`, `k`, `j`) and an on-chip buffer (`%alloc_0`) to accumulate results for one row of the output matrix `C` at a time. External memory accesses are identified as loads from function arguments `%arg0` (A) and `%arg1` (B), and stores to the final output buffer `%alloc` (C).

### External Memory Access

    *   **Load from A (`%arg0`)**: The load `affine.load %arg0` is within all three loops.
        $1024 \times 1024 \times 1024 \times (32 \div 8) = 4 \cdot 1024^3 \text{ bytes}$.
    *   **Load from B (`%arg1`)**: The load `affine.load %arg1` is also within all three loops.
        $1024 \times 1024 \times 1024 \times (32 \div 8) = 4 \cdot 1024^3 \text{ bytes}$.
    *   **Store to C (`%alloc`)**: The store `affine.store %alloc` happens after the accumulation for each row is complete, within two nested loops.
        $1024 \times 1024 \times (32 \div 8) = 4 \cdot 1024^2 \text{ bytes}$.
    *   **Total**:
        $4 \cdot 1024^3 + 4 \cdot 1024^3 + 4 \cdot 1024^2 = 4 \cdot (2 \cdot 1024^3 + 1024^2) = \pu{8594128896 bytes}$.


Example output
--------------

The number of bytes of external memory access is needed. The total external memory access in bytes is needed. From the input, the total external memory access is 8594128896 bytes. The last line of this answer must be a machine-parsable integer, without any symbols, such as commas, whitespace, brackets, quotation marks, or backticks.

8594128896
'''
        return self._eval_extract(eval_prompt, extract_prompt,
                                  lambda x: int(x))[0]

    def _run_csynth(self):
        """ Generates the Vivado HLS project directory.
        May take ~1-5 minutes.
        
        :return: 0 for success, 1 for failed. """

        try:
            tid = threading.get_ident()
            prj_name = f'vivado/{dt.now().strftime('%m_%d_%H_%M_%S_%f')}_{tid}.prj'
            mod = self.s.build(target='vivado_hls', mode='csyn', project=prj_name)
            logger.debug('Csynth starts.')
            mod()
            self.prj_name = prj_name
            return 0
        except Exception as e:
            logger.error(e)
            shutil.rmtree(prj_name, ignore_errors=True)
            return 1

    def _get_hls_sources(self, prj_path:Path|str) -> list[Path]:
        """ Returns list of paths of the source code files used for Vivado HLS.
        
        :param prj_path: path of the project directory.

        :return: list of paths of the source code files used for Vivado HLS.
        """
        src_dir = Path(prj_path)
        src_files = []
        for pat in ('*.c*', '*.h*'):
            tmp_lst = [f for f in src_dir.glob(pat) if f.name != 'timer.h']
            src_files.extend(tmp_lst)
        return src_files
    
    def _get_hls_rpts(self, prj_path:Path|str):
        """ Returns a list of paths of all `*.rpt` files.

        :param prj_path: path of the project directory.

        :return: list of all `*.rpt` file paths in
                `out.prj/solution1/syn/report`.
        """
        rpt_dir = Path(prj_path) / 'out.prj' / 'solution1' / 'syn' / 'report'
        rpt_files = [f for f in rpt_dir.glob('*.rpt')]
        return rpt_files

    def _get_clk_period(self, prj_path:Path|str):
        """ Returns the target clock period of the design in seconds.
        
        :param prj_path: path of the project directory.
        
        :return: target clock period of the design in seconds.
        """
        conversion_table = (('fs', 1e-15),
                            ('ps', 1e-12),
                            ('ns', 1e-9),
                            ('us', 1e-6),
                            ('ms', 1e-3),
                            ('s', 1.0))
        with (Path(prj_path) / 'report.json').open('r') as f:
            rpt = json.load(f)
        try:
            clk_unit = str(rpt['UserAssignments']['unit'])
            clk_period = float(rpt['UserAssignments']['TargetClockPeriod'])
        except Exception as e:
            err_msg = 'Failed to parse HLS report for clock period.\n'
            err_msg += f'Additional error message: {e}\n'
            err_msg += f'Section in report.json:\n'
            err_msg += f'{json.dumps(rpt['UserAssignments'], indent=4)}\n'
            logger.error(err_msg)
            return None
        coeff = 1.0
        for unit, unit_coeff in conversion_table:
            if clk_unit == unit:
                coeff = unit_coeff
                break
        return clk_period * coeff

    def _get_latency_range(self, prj_path:Path|str):
        """ Returns (min_latency, max_latency) of the design, based on the II.

        :param prj_path: path of the project directory.

        :return: (min_latency, max_latency) of the design in clock cycle count.
                Returns None if fails.
        """
        with (Path(prj_path) / 'report.json').open('r') as f:
            rpt = json.load(f)
        latency_estms = rpt['PerformanceEstimates']['SummaryOfOverallLatency']
        try:
            min_latency = int(latency_estms['Interval-min'])
            max_latency = int(latency_estms['Interval-max'])
        except Exception as e:
            err_msg = 'Failed to parse HLS report for latency range.\n'
            err_msg += f'Additional error message: {e}\n'
            err_msg += f'Section in report.json:\n'
            err_msg += f'{json.dumps(latency_estms, indent=4)}\n'
            logger.error(err_msg)
            return None
        return min_latency, max_latency

    def _get_once_cycles_per_kernel(self,
                                     attach_paths:list[str|Path],
                                     additional_prompt:str) -> tuple[float, str]:
        """ Return the estimated latency in cycles per kernel.
        
        :param attach_paths: list of paths that include all 
                attachments (source code files and csynth reports) to
                self.eval_model.
        :param additional_prompt: prepended to the prompt, gives
                additional information such as conversation history.

        :return: tuple of 1) estimated latency in cycles per kernel, and
                2) raw answer of self.eval_model
        """
        eval_prompt = additional_prompt
        eval_prompt += \
rf'''
You are an expert in high-level synthesis and hardware accelerator design. A C/C++ software (see attached files) is fed into Vivado HLS, and csynth report(s) are generated (see attached), as well as code for a hardware accelerator for the software. Estimate the *overall latency* in *cycles per kernel* for the synthesized hardware accelerator, if it is to run continually for a long time. Clearly state the final answer as *one single* number.

## Methodology

- If the csynth report gives multiple latency values, it is because of branches in the control flow. Find the branching points, carefully examine them, considering the design's input and outputs, to give an estimate of the proportion of branch hits. Use this ratio to give a weighted-sum of the multiple latency values from the csynth report. Only fall back to uniform-weight averaging (or "Average-case Latency" provided by Vivado HLS) as a last resort.
- For dataflow-optimized designs, if the intermediate values are vectors, the FIFOs between dataflow stages are implemented as dual-port BRAMs. Csynth reports over-optimistically estimate the dataflow latency as the max latency of all components. In reality, the dataflow pipeline is implemented as a PIPO that causes stalls. For large vectors, the dataflow optimization has trivial effects. For scalars, csynth gives the correct latency estimation for dataflow pipelines.
- **Make sure maths and unit conversion are done correctly.**
'''
        self.eval_model.add_files(attach_paths)
        extract_prompt = \
r'''
Return the estimated latency in cycles per kernel, as one single integer, at the exact *last* line of your answer.

Example input
-------------

### Thought Process Summary

1.  **Identify Top-Level Module:** The top-level module for the accelerator is `dut`, as specified in the main HLS report (`csynth.xml` and `dut_csynth.xml`). This module encapsulates the entire Binarized Neural Network (`bnn_xcel`).

2.  **Extract Core Metrics:** From the `dut_csynth.xml` report, the key performance metrics are:
    *   Average-case Latency: 48,483 clock cycles.
    *   Best-case Latency: 40,579 clock cycles.
    *   Worst-case Latency: 56,099 clock cycles.

3.  **Analyze Latency Variation:** The latency varies because of data-dependent control flow. Examining the sub-module reports reveals the `conv` functions (`conv_1_16_18_0_s` and `conv_16_32_10_1_s`) have variable latency. Analysis of the source code (`layer.hpp`) shows an `if (if_mac<I>(...))` statement within the convolution loops. This conditional statement skips multiply-accumulate (MAC) operations for padded border pixels. The path taken depends only on loop counters, making the latency deterministic for any given execution. The HLS tool reports a range because its static analysis of this complex control flow is conservative. The tool's "Average-case Latency" is the most reasonable estimate, likely representing an unweighted average of the control flow paths.

4.  **Calculate Latency:** The `dut` module is not pipelined (`PipelineType: none`), meaning a new transaction can only start after the previous one completes. Therefore, the initiation interval (II) equals the total latency.

### High-Level Estimation

*   **Design:** `dut`
*   **Clock Period:** 10.00 ns (100 MHz)
*   **Latency (cycles):** 48,483 cycles (using the HLS report's average-case estimate)

Example output
--------------

The latency in cycles per kernel is needed. In this specific case, the latency (cycles) is needed. From the input, the latency in units of clock cycles is 48483. The last line of this answer must be a machine-parsable integer, without any symbols, such as commas, whitespace, brackets, quotation marks, or backticks.

48483
'''
        return self._eval_extract(eval_prompt, extract_prompt,
                                  lambda x: float(x))
    
    def _get_cycles_per_kernel(self, attach_paths:list[str|Path], 
                               latency_range:tuple[int], tries=3,
                               rtol=1e-2) -> float:
        """ Gets the estimated latency in cycles per kernel, re-estimates if
        not in (min_latency, max_latency) as in the csynth report.
        
        :param attach_paths: list of paths that include all 
                attachments (source code files and csynth reports) to
                self.eval_model.
        :param latency_range: range of (min_latency, max_latency) that
                the estimated latency should fall in
        :param tries: (optional) number of attempts to re-estimate.
        :param rtol: (optional) does not re-estimate if estimated latency
                is between [`intv_range[0]*(1-rtol)`, `intv_range[1]*(1+rtol)`]

        :return: estimated latency in cycles per kernel.
        """
        assert (len(latency_range) == 2 and
                latency_range[0] <= latency_range[1]), \
               f'Wrong latency_range: {latency_range}'

        additional_prompt = 'Conversation history (oldest first):'
        min_intv, max_intv = latency_range
        for _ in range(tries):
            estm_lat, raw_ans = self._get_once_cycles_per_kernel(attach_paths,
                                                                  additional_prompt)
            if min_intv*(1-rtol) <= estm_lat <= max_intv*(1+rtol):
                return estm_lat
            else:
                debug_warning_msg = f'Estimated latency ({estm_lat} [cycles per kernel]) '
                debug_warning_msg += f'not between min_interval ({min_intv}) '
                debug_warning_msg += f'and max_interval ({max_intv}).'
                logger.debug(debug_warning_msg)
                logger.debug('Trying again...')
                additional_prompt += \
f'''
Large language model (latency estimator):

{raw_ans}

User:

Are you sure? The HLS report has minimum interval {min_intv} and maximum interval {max_intv}. Your estimation, {estm_lat}, is not between these values. Try estimating the latency again.
'''
        logger.warning(f'{debug_warning_msg}. Proceeding to next step...')
        return estm_lat

    def get_seconds_per_kernel(self, tries=3):
        """ Gets latency in seconds per kernel. Runs csynth and LLM.
         
        :param tries: (optional) number of attempts to re-estimate.
        """
        # Run csynth if and only if no csynth dirs exist
        if not Path(self.prj_name).is_dir():
            csynth_exit_state = self._run_csynth()
            if csynth_exit_state != 0:
                logger.error('Csynth failed.')
                return None
        prj_path = Path(self.prj_name)
        prj_files_paths = self._get_hls_sources(prj_path)
        prj_files_paths.extend(self._get_hls_rpts(prj_path))
        latency_range = self._get_latency_range(prj_path)
        seconds_per_cycle = self._get_clk_period(prj_path)
        if (latency_range is None) or (seconds_per_cycle is None):
            return None
        
        cycles_per_kernel = self._get_cycles_per_kernel(prj_files_paths,
                                                        latency_range,
                                                        tries=tries)
        seconds_per_kernel = cycles_per_kernel * seconds_per_cycle
        return seconds_per_kernel

    def get_coords(self):
        """ Gets the coordinates of self.s, for roofline modelling.
        
        :param tries: (optional) number of attempts to re-estimate throughput.

        :return: coordinates for roofline modelling. (Throughput, OI) =
                (opcount_per_second, opcount_per_byte_external_mem_access).
                Returns (0, 0) if error.
        """
        try:
            with ThreadPoolExecutor() as ex:
                th_opcount_per_kernel = ex.submit(self.get_opcount_per_kernel)
                th_bytes_per_kernel = ex.submit(self.get_bytes_per_kernel)
                th_seconds_per_kernel = ex.submit(self.get_seconds_per_kernel)
                opcount_per_kernel = th_opcount_per_kernel.result() 
                bytes_per_kernel = th_bytes_per_kernel.result() 
                seconds_per_kernel = th_seconds_per_kernel.result()
            if seconds_per_kernel is None:
                logger.error('Failed to get seconds_per_kernel')
                return (0, 0)
            opcount_per_second = opcount_per_kernel / seconds_per_kernel
            opcount_per_byte = opcount_per_kernel / bytes_per_kernel

            coords = (opcount_per_second, opcount_per_byte)
            logger.debug(f'Coordinates generated: {coords}')
            return coords
        finally:
            shutil.rmtree(self.prj_name, ignore_errors=True)
