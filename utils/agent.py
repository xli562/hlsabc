from pathlib import Path
import requests
import json

from utils.xlogging import get_logger


logger = get_logger()


GPRO = 'google/gemini-2.5-pro'
GFLASH = 'google/gemini-2.5-flash'
GLITE = 'google/gemini-2.5-flash-lite'

class Agent:
    """ LLM agent """

    def __init__(self, model:str):
        self.model = model
        self.files: dict[str, str] = {}
        self.system_prompt = ''
        self.user_prompt = ''

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
        with open(Path.home() / 'openrouterkey', 'r') as f:
            api_key = f.readline().strip()

        # Prompt construction
        user_prompt = self.user_prompt

        if self.files:
            user_prompt += '\n\nHere are the attached file(s):\n\n'
            for name, content in self.files.items():
                user_prompt += f'## {name}\n'
                user_prompt += f'{content}\n\n'
        
        # Generate response
        response = requests.post(
            url='https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
            },
            data=json.dumps({
                'model':self.model,
                'reasoning':{
                    # 'max_tokens':2000,  # Specify thinking budget
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
        try:
            retstr = response.json()['choices'][0]['message']['content']
        except:
            pass
        if not retstr:
            logger.info(f'Failed to parse this LLM response:')
            logger.info(json.dumps(response.json(), indent=4))
        return retstr
