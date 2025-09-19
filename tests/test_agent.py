from pathlib import Path
import sys

from utils.agent import Agent
from utils.agent import GPRO, GFLASH, GLITE
from utils.xlogging import logger


def test_simple():
    """ Makes sure Agent can initialize successfully. """

    agent = Agent(GLITE)
    agent.user_prompt = ''

def test_generate_simple():
    """ Gives simple question, expects non-empty result. """

    agent = Agent(GLITE)
    agent.user_prompt = "How many letter r's are in the word 'strawberry'?"
    got = agent.generate()
    logger.debug(got)
    assert isinstance(got, str), 'Agent.generate() does not return string'
    assert len(got) > 1, 'Agent.generate() returns empty response'

def test_tok_budget():
    """ Sets a low token budget, expects generate to return ''. """

    agent = Agent(GLITE, tok_budget=5)
    agent.user_prompt = "How many letter r's are in the word 'strawberry'?"
    got = agent.generate()
    got = agent.generate()
    assert got == ''

def test_add_file():
    """ Adds test directory to Agent, expects correct treatment of nested dirs """

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

def test_get_prompt_with_files():
    """ Adds test directory to Agent, expect correctly constructed prompt """
    agent = Agent(GLITE)
    agent.user_prompt = 'Hi'
    test_add_file_root = Path('tests/resource/test_add_file/')
    agent.add_files([test_add_file_root/'1'/'1.1',
                    test_add_file_root/'2'/'2.1',
                    test_add_file_root/'2',
                    test_add_file_root/'2'/'2.1.a'])
    got = agent.get_prompt_with_files()
    exp = '''Hi
\nHere are the attached file(s):

## tests/resource/test_add_file/1/1.1/1.1.1.a,
Contents of 1.1.1.a

## tests/resource/test_add_file/1/1.1/1.1.2.a
Contents of 1.1.2.a

## tests/resource/test_add_file/2/2.1/2.1.1/2.1.1.1.a
Contents of 2.1.1.1.a

## tests/resource/test_add_file/2/2.1.a
Contents of 2.1.a\n\n'''
    assert got == exp

def test_generate_repeat_add_files():
    """ Makes sure files cannot be appended more than once. """

    agent = Agent(GLITE)
    agent.user_prompt = 'Hi'
    test_add_file_root = Path('tests/resource/test_add_file/')
    agent.add_files([test_add_file_root/'1'/'1.1',
                    test_add_file_root/'2'/'2.1',
                    test_add_file_root/'2',
                    test_add_file_root/'2'/'2.1.a'])
    agent.user_prompt = agent.get_prompt_with_files()
    agent.user_prompt = agent.get_prompt_with_files()
    got = len(agent.user_prompt)

    assert got <= 308, 'Files are appended more than once to Agent.user_prompt'
