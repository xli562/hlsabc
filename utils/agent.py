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

    def add_files(self, paths:list[str|Path]) -> None:
        """ Appends text-encoded files to Agent.files
        
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
                    # 'max_tokens':2000,  # Explicitly set thinking budget
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
