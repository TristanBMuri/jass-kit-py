import copy
import logging
import random

import numpy as np
from Tools.demo.sortvisu import Array

from jass.agents.Helpers.Node import Node
from jass.agents.agent import Agent
from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.game_rule import GameRule
from jass.game.game_sim import GameSim
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber


class TrademarkMCTS:
    obe_nabe = [8, 7, 6, 5, 4, 3, 2, 1, 0]
    une_ufe = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    trump_modifier = [9, 9, 9, 16, 9, 16, 9, 9, 9]

    def __init__(self, player_id, max_depth=3):
        self.player_id = player_id
        self.ruleset = RuleSchieber()
        self.max_depth = max_depth

    def create_game_sim_from_observation(self, game_obs: GameObservation) -> GameSim:
        """
        Create a GameSim object from an existing GameObservation object.

        Args:
            game_obs (GameObservation): The game observation object containing the current game state.
            rule (GameRule): The rule set to be applied to the GameSim.

        Returns:
            GameSim: A new GameSim object initialized based on the game observation.
        """
        # Create a new GameSim instance with the given rule
        game_sim = GameSim(RuleSchieber())

        # Set up the game state using the observation
        game_state = game_sim._state

        # Map properties from GameObservation to GameState
        game_state.dealer = game_obs.dealer
        game_state.player = game_obs.player
        game_state.trump = game_obs.trump
        game_state.forehand = game_obs.forehand
        game_state.declared_trump = game_obs.declared_trump

        # Set tricks, trick winners, points, and related data
        game_state.tricks = game_obs.tricks.copy()
        game_state.trick_winner = game_obs.trick_winner.copy()
        game_state.trick_points = game_obs.trick_points.copy()
        game_state.trick_first_player = game_obs.trick_first_player.copy()
        game_state.nr_tricks = game_obs.nr_tricks
        game_state.nr_cards_in_trick = game_obs.nr_cards_in_trick
        game_state.nr_played_cards = game_obs.nr_played_cards
        game_state.points = game_obs.points.copy()

        # Generate the hands for all players based on the observation
        game_obs_copy = GameObservation()  # If needed, clone or copy game_obs
        hands = self.generate_opponent_hands(game_obs_copy)
        game_state.hands = hands

        # Return the initialized GameSim object
        return game_sim

    def selection(self, node):
        # tree policy
        while not node.game_obs.nr_tricks == 9 and node.is_fully_expanded():
            node = node.best_child()
        return node

    def simulate_move(self, game_obs, move):
        game_sim = self.create_game_sim_from_observation(game_obs)
        game_sim.action_play_card(move)
        # print(f"Move {move} simulated")
        return game_sim.get_observation()

    def expansion(self, node):
        if node.is_fully_expanded():
            return None

        valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
            node.ruleset.get_valid_cards_from_obs(node.game_obs))

        tried_moves = [child.move for child in node.children]
        # print(f"tried_moves: \n{tried_moves}")
        # print(f"valid_moves: \n{valid_moves}")
        untried_moves = [move for move in valid_moves if move not in tried_moves]

        if not untried_moves:
            return None

        move = random.choice(untried_moves)
        # print(f"chosen_move:\n{move}")

        new_game_obs = self.simulate_move(node.game_obs, move)

        child_node = Node(game_obs=new_game_obs, parent=node, move=move)
        node.add_child(child_node)

        return child_node

    def evaluate_outcome(self, game_obs):
        # print(f"evaluating outcome: \n{game_obs.points}")
        # print(f"Evaluating: {game_obs.points}; player_view: {game_obs.player_view}")
        if game_obs.points[0] == max(game_obs.points):
            # print(f"Team won")
            return 1  # Win
        else:
            # print(f"Team lost")
            return -1  # Loss

    def backpropagate(self, node, outcome):
        while node is not None:
            node.visits += 1
            # print(f"Backpropegate: node_visits: {node.visits}")
            if node.game_obs.player_view == self.player_id:
                node.total_reward += outcome  #

            else:
                node.total_reward += -outcome  # Opponent's perspective: MCTS agent loses

            node = node.parent

    def generate_opponent_hands(self, obs):
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

    def simulate(self, game_obs: GameObservation):
        # copy the current game state to avoid modifying the original state
        simulated_state = copy.deepcopy(game_obs)

        for i in range(self.max_depth):
            # Check if the game has ended
            # print(f"Depth {i} ")
            if simulated_state.nr_tricks >= 9:
                break

            valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
                self.ruleset.get_valid_cards_from_obs(game_obs))
            if np.sum(valid_moves) == 0:
                break

            move = self.get_lowest_card(valid_moves, game_obs.declared_trump)

            current_trick = game_obs.current_trick

            #print("Lowest?:", card_strings[move])
            #print("Could be lower than:")
            #for i in current_trick:
            #    print(card_strings[i])


            for card in valid_moves:
                if card < self.get_lowest_card(valid_moves, game_obs.declared_trump):
                    move = card
                    break

            # print("updated card:", card_strings[move])


            simulated_state = self.simulate_move(simulated_state, move)

        outcome = self.evaluate_outcome(simulated_state)

        return outcome

    def get_lowest_card(self, moves: Array, declared_trump: int) -> int:
        current_card = None
        current_card_score = 0
        temp_score = 0

        if declared_trump == 5:
            for i in moves:
                if current_card is None:
                    current_card = i
                if self.une_ufe[current_card % 9] < self.une_ufe[i % 9]:
                    current_card = i
        else:
            for i in moves:
                if current_card is None:
                    current_card = i
                    current_card_score = self.obe_nabe[i % 9]
                    if self.is_trump(current_card, declared_trump):
                        current_card_score += self.trump_modifier[i % 9]
                temp_score = self.obe_nabe[i % 9]
                if self.is_trump(i, declared_trump):
                    temp_score += self.trump_modifier[i % 9]
                if temp_score < current_card_score:
                    current_card = i

        return current_card

    def get_highest_card(self, moves: Array, declared_trump: int) -> int:
        current_card = None
        current_card_score = 0
        temp_score = 0

        if declared_trump == 5:
            for i in moves:
                if current_card is None:
                    current_card = i
                if self.une_ufe[current_card % 9] > self.une_ufe[i % 9]:
                    current_card = i
        else:
            for i in moves:
                if current_card is None:
                    current_card = i
                    current_card_score = self.obe_nabe[i % 9]
                    if self.is_trump(current_card, declared_trump):
                        current_card_score += self.trump_modifier[i % 9]
                temp_score = self.obe_nabe[i % 9]
                if self.is_trump(i, declared_trump):
                    temp_score += self.trump_modifier[i % 9]
                if temp_score > current_card_score:
                    current_card = i

        return current_card

    def is_trump(self, card: int, trump: int) -> bool:
        if trump > 3:
            return False
        elif card >= 9 * trump and card > 9 * (trump + 1):
            return True
        else:
            return False