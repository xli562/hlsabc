from allo.customize import Schedule
from utils.agent import Agent, GPRO, GFLASH, GLITE
from utils.xlogging import get_logger


logger = get_logger()

class Roofline:
    """ Gets roofline coordinates of Allo Schedule """

    def __init__(self, s:Schedule):
        self.s = s
        self.eval_model = Agent(GPRO)  # Does performance estimation
        self.extract_model = Agent(GLITE)  # Extracts numbers from raw answers
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
        """ Runs self.eval_model, then self.extract_model. 
        
        :param eval_prompt: user prompt for self.eval_model.
        :param extract_prompt: user prompt for self.extract_model.
        :param output_convert: callable that converts the output string.
        :param tries: (optional) re-run count for self.extract_model before
                failing.

        :return: the converted output, passed through output_convert. None if
                conversion fails.

        :raises warning: fails to produce answer that is convertible by
                output_convert.
        """
        self.eval_model.user_prompt = eval_prompt
        logger.debug('Running eval_model...')
        raw_output = self.eval_model.generate()
        logger.debug('eval_model finished.')
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
                return output_convert(extract_output.splitlines()[-1])
            except Exception as e:
                wrong_answers.add(extract_output)
                logger.debug(f'Failed to convert extract_model output:\n{e}')

        logger.warning('Failed to produce a convertible answer.')
        
    def get_opcount_per_kernel(self):
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
        extract_prompt = \
r'''
Return the core operation count an integer at exact *last* line of your answer.

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

The core operation count is needed. The execution count of the core operation is needed. From the input, the execution count is 1073741824 times. The last line of this answer must be a machine-parsable integer, without any symbols, such as brackets, or quotation marks, or backticks.

1073741824
'''
        return self._eval_extract(eval_prompt, extract_prompt,
                                  lambda x: int(x))
