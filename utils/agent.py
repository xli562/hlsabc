from pathlib import Path
from google import genai
from google.genai import types

from utils.xlogging import get_logger


logger = get_logger()


GPRO = 'gemini-2.5-pro'
GFLASH = 'gemini-2.5-flash'
GLITE = 'gemini-2.5-flash-lite'
GEMINI = {GPRO, GFLASH, GLITE}

class Agent:
    """ LLM agent """

    def __init__(self, model:str):
        self.model = model
        if model in GEMINI:
            self.client = genai.Client()
        self.files: dict[str, str] = {}
        self.prompt = ''

    def add_files(self, file_paths:list[str|Path]) -> None:
        """ Appends text-encoded files to Agent.files
        
        :param file_path: list of paths to files. Files added in list's order.
        """
        for file_path in file_paths:
            file_path = Path(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_contents = f.read()
                    self.files[file_path.name] = file_contents
            except FileNotFoundError:
                logger.warning(f"Input file not found: {file_path}")
            except UnicodeDecodeError:
                logger.warning(f"Unable to decode input file as text: {file_path}")
            except Exception as e:
                logger.warning(f"Error reading input file {file_path}: {e}")

    def generate(self):
        """ Run the LLM

        :return: (str) response from LLM
        """
        if self.model in GEMINI:
            return self._gemini_generate()
    
    def _gemini_generate(self, thinking_budget=-1, include_thoughts=False):
        """ Generates string with Gemini
        
        :return: (str) response from Gemini
        """
        # Prompt construction
        prompt = self.prompt

        if self.files:
            prompt += '\n\nHere are the attached file(s):\n\n'
            for name, content in self.files.items():
                prompt += f'## {name}\n'
                prompt += f'{content}\n\n'
        
        # Generate response
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                    include_thoughts=include_thoughts
                )
            )
        )

        # Parse response
        retstr = ''
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                retstr += 'Thought summary:'
                retstr += part.text + '\n'
            else:
                retstr += 'Answer:'
                retstr += part.text + '\n'
        
        return retstr