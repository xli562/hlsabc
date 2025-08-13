from pathlib import Path

from utils.agent import Agent
from utils.agent import GPRO, GFLASH, GLITE, GEMINI
from utils.xlogging import get_logger


logger = get_logger()

def test_simple():
    agent = Agent(GLITE)
    agent.prompt = ''

def test_generate_simple():
    """ Gives simple question, expects non-empty result. """

    agent = Agent(GPRO)
    agent.prompt = "How many letter r's are in the word 'strawberry'?"
    assert isinstance(agent.generate(), str)
    assert len(agent.generate()) > 1

def test_add_file():
    """ Gives CORDIC report, expects 'cordic' in response. """

    agent = Agent(GLITE)
    agent.add_files(['tests/resource/cordic_csynth.rpt'])
    agent.prompt = 'Which algorithm is this HLS report for?'
    assert 'cordic' in agent.generate().lower()
