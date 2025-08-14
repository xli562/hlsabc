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
    """ Gives CORDIC report, expects 'cordic' in response. """

    agent = Agent(GLITE)
    agent.add_files(['tests/resource/cordic_csynth.rpt'])
    agent.user_prompt = 'Which algorithm is this HLS report for?'
    assert 'cordic' in agent.generate().lower()
