import unittest

import numpy as np

from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.agents.agent_transformed_mc_v3 import TransformedMCTSAgent
from jass.arena.arena import Arena
from jass.game.const import *
from jass.game.game_sim import GameSim
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber


class TransformedMCAgentTests(unittest.TestCase):

    def test_trump_selection_hand_1(self):
        #TODO
        assert True


    def test_game_skill(self):
        rule = RuleSchieber()
        game = GameSim(rule=rule)

        np.random.seed(1)
        game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

        arena = Arena(nr_games_to_play=10)
        arena.set_players(TransformedMCTSAgent(), AgentRandomSchieber(), TransformedMCTSAgent(), AgentRandomSchieber())

        arena.play_all_games()

        print(arena.points_team_0.sum(), arena.points_team_1.sum())

        assert True


if __name__ == '__main__':
    unittest.main()