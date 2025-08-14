from pathlib import Path

from utils.agent import Agent
from utils.agent import GPRO, GFLASH, GLITE
from utils.xlogging import get_logger


logger = get_logger()

def test_simple():
    agent = Agent(GLITE)
    agent.user_prompt = ''

def test_generate_simple():
    """ Gives simple question, expects non-empty result. """

    agent = Agent(GLITE)
    agent.user_prompt = "How many letter r's are in the word 'strawberry'?"
    got = agent.generate()
    logger.debug(got)
    assert isinstance(got, str)
    assert len(got) > 1

def test_add_file():
    """ Adds test folder to Agent, expects correct treatment of nested dirs """

    agent = Agent(GLITE)
    test_add_file_root = Path('tests/resource/test_add_file/')
    agent.add_files([test_add_file_root/'1'/'1.1',
                    test_add_file_root/'2'/'2.1',
                    test_add_file_root/'2',
                    test_add_file_root/'2'/'2.1.a'])
    got = agent.files
    exp = {'tests/resource/test_add_file/1/1.1/1.1.1.a,':'Contents of 1.1.1.a', 
           'tests/resource/test_add_file/1/1.1/1.1.2.a':'Contents of 1.1.2.a', 
           'tests/resource/test_add_file/2/2.1/2.1.1/2.1.1.1.a':'Contents of 2.1.1.1.a', 
           'tests/resource/test_add_file/2/2.1.a':'Contents of 2.1.a'}
    assert got == exp
