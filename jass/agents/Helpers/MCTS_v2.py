import copy
import random
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber


class MCTS:
    def __init__(self, player_id, team_id, max_depth=9):
        self.player_id = player_id
        self.ruleset = RuleSchieber()
        self.max_depth = max_depth
        self.team_id = team_id



    def selection(self, node):
        # tree policy
        while not node.game_sim.state.nr_tricks == 9 and node.is_fully_expanded():
            node = node.best_child()
        return node

    def simulate_move(self, game_sim, move):
        sim_copy = copy.deepcopy(game_sim)
        sim_copy.action_play_card(move)
        if sim_copy.is_done():
            return sim_copy

        for i in range(4 - np.sum(sim_copy.state.nr_cards_in_trick)):
            valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
                RuleSchieber().get_valid_cards_from_obs(sim_copy.get_observation()))
            player_move = random.choice(valid_moves)
            sim_copy.action_play_card(player_move)
        return sim_copy

    def evaluate_outcome(self, game_state):
        # print(f"evaluating outcome: \n{game_obs.points}")
        # print(f"Evaluating: {game_state.points}; player_view: {game_state.player_view}")

        if game_state.points[self.team_id] == max(game_state.points):
            # print(f"Team won")
            return 1  # Win
        else:
            # print(f"Team lost")
            return -1  # Loss

    def backpropagate(self, node, outcome):
        while node is not None:
            node.visits += 1
            # print(f"Backpropegate: node_visqits: {node.visits}")
            node.total_reward += outcome  #
            node = node.parent

    def simulate(self, game_sim):
        # copy the current game state to avoid modifying the original state

        for i in range(self.max_depth):
            # Check if the game has ended
            # print(f"Depth {i} ")
            if game_sim.state.nr_tricks >= 9:
                break

            valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
                self.ruleset.get_valid_cards_from_obs(game_sim.get_observation()))
            if np.sum(valid_moves) == 0:
                break

            move = random.choice(valid_moves)

            game_sim = self.simulate_move(copy.deepcopy(game_sim), move)

        outcome = self.evaluate_outcome(game_sim.state)

        return outcome
