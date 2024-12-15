import unittest

import numpy as np

from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.agents.agent_transformed_mc_v4 import TransformedMCTSAgent
from jass.arena.arena import Arena
from jass.game.const import *
from jass.game.game_sim import GameSim
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.MCTSAgent_v2 import MCTSAgent


class TransformedMCAgentTests(unittest.TestCase):

    def test_trump_selection_hand_1(self):
        assert True


    def test_game_skill(self):
        rule = RuleSchieber()
        game = GameSim(rule=rule)
        game_count = 30

        np.random.seed(5)
        game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

        arena = Arena(nr_games_to_play=game_count)
        arena.set_players(TransformedMCTSAgent(), AgentRandomSchieber(), TransformedMCTSAgent(), AgentRandomSchieber())

        arena.play_all_games()

        print(arena.points_team_0.sum() / game_count, arena.points_team_1.sum() / game_count)

        assert True

    def test_game_skill2(self):
        rule = RuleSchieber()
        game = GameSim(rule=rule)
        game_count = 10

        np.random.seed(5)
        game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

        arena = Arena(nr_games_to_play=game_count)
        arena.set_players(TransformedMCTSAgent(), MCTSAgent(), TransformedMCTSAgent(), MCTSAgent())

        arena.play_all_games()

        print(arena.points_team_0.sum() / game_count, arena.points_team_1.sum() / game_count)

        assert True

    def test_game_skill3(self):
        rule = RuleSchieber()
        game = GameSim(rule=rule)
        game_count = 100

        np.random.seed(5)
        game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

        arena = Arena(nr_games_to_play=game_count)
        arena.set_players(MCTSAgent(), TransformedMCTSAgent(), MCTSAgent(), TransformedMCTSAgent())

        arena.play_all_games()

        print(arena.points_team_0.sum() / game_count, arena.points_team_1.sum() / game_count)

        assert True

    def test_game_skill4(self):
        rule = RuleSchieber()
        game = GameSim(rule=rule)
        game_count = 30

        np.random.seed(5)
        game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

        arena = Arena(nr_games_to_play=game_count)
        arena.set_players(AgentRandomSchieber(), TransformedMCTSAgent(), AgentRandomSchieber(), TransformedMCTSAgent())

        arena.play_all_games()

        print(arena.points_team_0.sum() / game_count, arena.points_team_1.sum() / game_count)

        assert True

    def test_game_skill5(self):
        rule = RuleSchieber()
        game = GameSim(rule=rule)
        game_count = 30

        np.random.seed(5)
        game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

        arena = Arena(nr_games_to_play=game_count)
        arena.set_players(AgentRandomSchieber(), MCTSAgent(), AgentRandomSchieber(), MCTSAgent())

        arena.play_all_games()

        print(arena.points_team_0.sum() / game_count, arena.points_team_1.sum() / game_count)

        assert True

if __name__ == '__main__':
    unittest.main()