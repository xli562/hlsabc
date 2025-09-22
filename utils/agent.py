from pathlib import Path
from google import genai
from google.genai import types
import requests
import json
import os

from dotenv import load_dotenv
from utils.xlogging import logger


load_dotenv()

USE_GEMINI = False

GPRO = 'gemini-2.5-pro' if USE_GEMINI else 'google/gemini-2.5-pro'
GFLASH = 'gemini-2.5-flash' if USE_GEMINI else 'google/gemini-2.5-flash'
GLITE = 'gemini-2.5-flash-lite' if USE_GEMINI else 'google/gemini-2.5-flash-lite'

class Agent:
    """ LLM agent """

    def __init__(self, model:str, tok_budget=100_000):
        """ Initializes the agent.
        
        :param model: name of model, eg 'google/gemini-2.5-flash'
        :param tok_budget: (Optional) max allowable tokens
                used by a class instance. Logs warnings on high usage, disables
                further prompting if usage exceeds budget.
        """
        self._model = model
        if USE_GEMINI:
            self.client = genai.Client()
        self._tok_budget = tok_budget
        self._tok_used = 0
        self.files: dict[str, str] = {}
        self.system_prompt = ''
        self.user_prompt = ''

    def add_files(self, paths:list[str|Path]) -> None:
        """ Appends text-encoded files to Agent.files. This is the recommended
        way to include files in the prompt.
        
        :param paths: list of paths to files or dirs. Paths are
                processed in list's order. Directories are expanded recursively.
        """

        def build_key(root:Path, path:Path):
            if root.is_file():
                retstr = str(path)
            else:
                retstr = str(root / path.relative_to(root))
            return retstr

        def iter_files(path:Path):
            if path.is_file():
                yield build_key(path, path), path
            elif path.is_dir():
                for subpath in path.rglob('*'):
                    if subpath.is_file():
                        yield build_key(path, subpath), subpath
            else:
                logger.warning(f'Path is neither dir nor file: {path}')
        
        for path in paths:
            path = Path(path)
            for key_name, file_path in iter_files(path):
                try:
                    self.files[key_name] = file_path.read_text(encoding='utf-8')
                except Exception as e:
                    logger.warning(f'Error reading file {file_path}: {e}')

    def accumulate_tok_count(self, delta_tok_count:int):
        """ Increments self._tok_used by the input.
        
        :param delta_tok_count: the amount to increment for.
        """
        log_tiers = [(0.10, 'INFO'),
                     (0.20, 'INFO'),
                     (0.30, 'INFO'),
                     (0.40, 'INFO'),
                     (0.50, 'INFO'),
                     (0.60, 'INFO'),
                     (0.70, 'INFO'),
                     (0.75, 'WARN'),
                     (0.85, 'WARN'),
                     (0.90, 'WARN'),
                     (0.95, 'WARN'),
                     (0.99, 'WARN')]
        log_tiers.reverse()
        self._tok_used += delta_tok_count
        for level, log_level in log_tiers:
            if self._tok_used > self._tok_budget * level:
                tier_percentage = int(level * 100)
                curr_percentage = self._tok_used / self._tok_budget * 100
                log_msg = f'Exceeded {tier_percentage} % token budget, currently {curr_percentage} %'
                if log_level == 'INFO':
                    logger.info(log_msg)
                elif log_level == 'WARN':
                    logger.warning(log_msg)
                break
    
    def get_prompt_with_files(self):
        """ Returns the user prompt with self.files appended.

        :return: the user prompt with self.files appended. Will not append 
                if 'Here are the attached file(s):' is already in 
                self.user_prompt.
        """
        user_prompt = self.user_prompt
        here_str = 'Here are the attached file(s):'
        if self.files and here_str not in self.user_prompt.splitlines():
            user_prompt += '\n\nHere are the attached file(s):\n\n'
            for name, content in self.files.items():
                user_prompt += f'## {name}\n'
                user_prompt += f'{content}\n\n'
            logger.debug(f'Appended {len(self.files)} file(s) to prompt.')
        else:
            logger.debug('No new files are attached to prompt')
        return user_prompt

    def generate(self):
        """ Run the LLM

        :return: (str) response from LLM. Returns '' if token budget exceeded.
        """
        if self._tok_used + 20 >= self._tok_budget:
            logger.warning(f'Exceeded token budget of {self._tok_budget}')
            return ''
        
        # Append files
        user_prompt = self.get_prompt_with_files()

        if USE_GEMINI:
            api_key = os.getenv('GEMINI_API_KEY')
            response = self.client.models.generate_content(
                model=self._model,
                contents=f'{self.system_prompt}\n\n\n{user_prompt}',
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=-1,
                        include_thoughts=False
                    )
                )
            )
            retstr = ''
            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                if part.thought:
                    retstr += 'Thought summary:\n'
                    retstr += '----------------\n'
                    retstr += f'{part.text}\n'
                    retstr += '----------------\n'
                else:
                    retstr += f'{part.text}\n'
        else:
            api_key = os.getenv('OPENROUTER_API_KEY')
        
            # Generate response
            response = requests.post(
                url='https://openrouter.ai/api/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                },
                data=json.dumps({
                    'model':self._model,
                    'reasoning':{
                        # 'max_tokens':100000,  # Explicitly set thinking budget
                        'enabled':True, # Automatically allocate thinking budget
                        'exclude':True  # Exclude thinking tokens from response
                    },
                    'messages':[{
                        'role':'system',
                        'content':self.system_prompt
                    },{
                        'role':'user',
                        'content':user_prompt
                    }]
                })
            )

            # Parse response
            retstr = ''
            
            def log_fail_to_parse():
                logger.error(f'Failed to parse this LLM response:')
                try:
                    corrupt_response = json.dumps(response.json(), indent=4)
                except:
                    corrupt_response = response.text
                logger.error(corrupt_response)
                
            try:
                retstr:str = response.json()['choices'][0]['message']['content']
                tok_count = response.json()['usage']['total_tokens']
                self.accumulate_tok_count(tok_count)
            except Exception as e:
                log_fail_to_parse()
                logger.error('Additional error message:')
                logger.error(e)
            if not retstr:
                log_fail_to_parse()

        return retstr
