import unittest

import numpy as np

from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.agents.agent_transformed_mc_trump_testah import TransformedMCTSAgent
from jass.arena.arena import Arena
from jass.game.const import *
from jass.game.game_sim import GameSim
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber


class TransformedMCAgentTests(unittest.TestCase):

    def test_trump_selection_hand_1(self):
        assert True


    def test_game_skill(self):
        rule = RuleSchieber()
        game = GameSim(rule=rule)
        game_count = 1000

        np.random.seed(5)
        game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

        arena = Arena(nr_games_to_play=game_count)
        arena.set_players(TransformedMCTSAgent(), AgentRandomSchieber(), TransformedMCTSAgent(), AgentRandomSchieber())

        arena.play_all_games()

        print(arena.points_team_0.sum() / game_count, arena.points_team_1.sum() / game_count)

        assert True


if __name__ == '__main__':
    unittest.main()