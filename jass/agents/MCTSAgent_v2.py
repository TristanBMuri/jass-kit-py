# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import random
import copy

import numpy as np

from jass.agents.Helpers.MCTS_v2 import MCTS
from jass.agents.Helpers.Node import Node
from jass.agents.agent import Agent
from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber



class MCTSAgent(Agent):
    def __init__(self, max_iterations=100):
        super().__init__()
        self.determinizations = max_iterations
        self._rule = RuleSchieber()

    def generate_opponent_hands(self, obs):
        # random.seed(42)
        possible_cards = [i for i in range(36)]

        possible_cards = set(possible_cards) ^ set(convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand))
        if len(obs.tricks[obs.tricks > 0]):
            possible_cards = set(possible_cards) ^ set(obs.tricks[obs.tricks != -1])
        unplayed_cards = np.full((1, 4), 9 - obs.nr_tricks)[0]

        # iterate backwards through players starting from the person who started the trick
        for i in range(obs.trick_first_player[obs.nr_tricks],
                       obs.trick_first_player[obs.nr_tricks] - obs.nr_cards_in_trick, -1):
            unplayed_cards[i % 4] -= 1  # modulo to loop back to the end, e.g. 1, 0, 3, 2
        # print(f"unplayed_cards: {unplayed_cards}")

        hands = np.zeros(shape=[4, 36], dtype=np.int32)
        hands[obs.player_view] = obs.hand

        for player_id in range(4):

            for i in range(unplayed_cards[player_id]):
                if player_id == obs.player_view:
                    continue
                int_possible_cards = list(possible_cards)
                card_choice = random.choice(int_possible_cards)
                possible_cards.remove(card_choice)

                hands[player_id] += get_cards_encoded(card_choice)
        return hands
    def determinize_observation(self, game_obs: GameObservation) -> GameSim:
        """
        Create a GameSim object from an existing GameObservation object. The opponent hands are generated randomly.
        """
        # Create a new GameSim instance with the given rule

        # Set up the game state using the observation
        game_state = GameState()

        # Map properties from GameObservation to GameState
        game_state.dealer = game_obs.dealer
        game_state.player = game_obs.player
        game_state.trump = game_obs.trump
        game_state.forehand = game_obs.forehand
        game_state.declared_trump = game_obs.declared_trump

        # Set tricks, trick winners, points, and related data
        game_state.current_trick = game_obs.current_trick
        game_state.tricks = game_obs.tricks.copy()
        game_state.trick_winner = game_obs.trick_winner.copy()
        game_state.trick_points = game_obs.trick_points.copy()
        game_state.trick_first_player = game_obs.trick_first_player.copy()
        game_state.nr_tricks = game_obs.nr_tricks
        game_state.nr_cards_in_trick = game_obs.nr_cards_in_trick
        game_state.nr_played_cards = game_obs.nr_played_cards
        game_state.points = game_obs.points.copy()

        # Generate the hands for all players based on the observation
        hands = self.generate_opponent_hands(game_obs)
        game_state.hands = hands

        game_sim = GameSim(RuleSchieber())
        game_sim.init_from_state(game_state)

        # Return the initialized GameSim object
        return game_sim

    def action_trump(self, obs: GameObservation) -> int:
        return DIAMONDS

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play using MCTS.
        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        mcts = MCTS(obs.player_view)
        game_sim = self.determinize_observation(obs)
        root = Node(game_sim=copy.deepcopy(game_sim))

        # Get all valid moves from the current observation
        valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
            RuleSchieber().get_valid_cards_from_obs(obs)
        )

        # Iterate over each valid move
        for move in valid_moves:
            move_node = Node(game_sim=copy.deepcopy(game_sim), parent=root, move=move)
            root.add_child(move_node)
            for _ in range(self.determinizations):
                determinized_sim = self.determinize_observation(obs)
                new_game_sim = mcts.simulate_move(determinized_sim, move)
                determinization_node = Node(
                    game_sim=copy.deepcopy(new_game_sim),
                    parent=move_node,
                    move=move
                )
                move_node.add_child(determinization_node)
                outcome = mcts.simulate(determinization_node.game_sim)
                mcts.backpropagate(determinization_node, outcome)

        best_child = root.best_child(exploration_param=2)
        return best_child.move if best_child else None

    # def action_play_card(self, obs: GameObservation) -> int:
    #     """
    #     Determine the card to play using MCTS.
    #     Args:
    #         obs: the game observation
    #
    #     Returns:
    #         the card to play, int encoded as defined in jass.game.const
    #     """
    #     # print(f"playerview: {obs.player_view}")
    #     mcts = MCTS(obs.player_view)
    #     game_sim = self.determinize_observation(obs)
    #     root = Node(game_sim=copy.deepcopy(game_sim))
    #
    #     # for all moves
    #     # determinize n scenarios per move
    #     # sum up outcomes
    #     # profit
    #     valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
    #         RuleSchieber().get_valid_cards_from_obs(obs))
    #     for move in valid_moves:
    #         for i in range(self.determinizations):
    #             # print(f"root hands:\n {np.sum(root.game_sim.state.hands)}")
    #             # print(f"Iteration {i}")
    #             # leaf = mcts.selection(root)
    #             # child = mcts.expansion(leaf)
    #
    #             mcts = MCTS(obs.player_view)
    #             game_sim = self.determinize_observation(obs)
    #             child = Node(game_sim=copy.deepcopy(game_sim))
    #
    #             new_game_sim = mcts.simulate_move(child.game_sim, move)
    #
    #             child_node = Node(game_sim=copy.deepcopy(new_game_sim), parent=child, move=move)
    #             child.add_child(child_node)
    #
    #             if child_node:
    #                 outcome = mcts.simulate(child_node.game_sim)
    #                 mcts.backpropagate(child_node, outcome)
    #             else:
    #                 outcome = mcts.simulate(child.game_sim)
    #                 mcts.backpropagate(child, outcome)
    #
    #     # print("unplayed cards", self.possible_cards(obs))
    #
    #     best_move = root.best_child(exploration_param=2).move
    #     return best_move
        # return np.random.choice(np.flatnonzero(valid_cards))





