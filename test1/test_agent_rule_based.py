import unittest

import numpy as np

from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.agents.agent_rule_based import AgentRuleBased
from jass.arena.arena import Arena
from jass.game.const import *
from jass.game.game_sim import GameSim
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber


class RuleBasedAgentTests(unittest.TestCase):

    def test_trump_selection_hand_1(self):
        rule = RuleSchieber()
        game = GameSim(rule=rule)
        agent = AgentRuleBased()

        np.random.seed(1)
        game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

        obs = game.get_observation()

        cards = convert_one_hot_encoded_cards_to_str_encoded_list(obs.hand)
        print(cards)
        print(obs.hand)
        trump = agent.action_trump(obs)
        print(f"Selected trump: {trump}", "Hearts, Obenabe:",HEARTS, OBE_ABE)  # Debugging output
        assert trump in [HEARTS, OBE_ABE]

    def test_trump_selection_hand_2(self):
        rule = RuleSchieber()
        game = GameSim(rule=rule)
        agent = AgentRuleBased()

        np.random.seed(22)
        game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

        obs = game.get_observation()

        cards = convert_one_hot_encoded_cards_to_str_encoded_list(obs.hand)
        print(cards)
        print(obs.hand)
        trump = agent.action_trump(obs)
        print(f"Selected trump: {trump}", "PUSH:", PUSH)  # Debugging output
        assert trump == PUSH # value = 10

    def test_trump_selection_hand_3(self):
        rule = RuleSchieber()
        game = GameSim(rule=rule)
        agent = AgentRuleBased()

        np.random.seed(30)
        game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

        obs = game.get_observation()

        cards = convert_one_hot_encoded_cards_to_str_encoded_list(obs.hand)
        print(cards)
        print(obs.hand)
        trump = agent.action_trump(obs)
        print(f"Selected trump: {trump}", "Hearts:", HEARTS)  # Debugging output
        assert trump is HEARTS

    def test_calculate_guaranteed_games_hand_1(self):
        rule = RuleSchieber()
        game = GameSim(rule=rule)
        agent = AgentRuleBased()

        np.random.seed(1)
        game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

        obs = game.get_observation()

        cards = convert_one_hot_encoded_cards_to_str_encoded_list(obs.hand)
        print(cards)
        print(obs.hand)

        expected_guaranteed_games = 5  # Based on the analysis of the hand
        guaranteed_games = 0

        trump = agent.action_trump(obs)

        for suit in range(4):
            guaranteed_games += agent.calculate_score_for_suit(obs.hand, trump, suit)
        print(f"Guaranteed games: {guaranteed_games}")
        assert guaranteed_games == expected_guaranteed_games

    def test_game_skill(self):
        rule = RuleSchieber()
        game = GameSim(rule=rule)

        np.random.seed(1)
        game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

        arena = Arena(nr_games_to_play=3)
        arena.set_players(AgentRuleBased(), AgentRandomSchieber(), AgentRuleBased(), AgentRandomSchieber())

        arena.play_all_games()

        print(arena.points_team_0.sum(), arena.points_team_1.sum())

        assert arena.points_team_0.sum() > arena.points_team_1.sum()


if __name__ == '__main__':
    unittest.main()