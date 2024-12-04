import math

from jass.game.rule_schieber import RuleSchieber


class Node:
    def __init__(self, game_sim, parent=None, move=None):
        self.game_sim = game_sim
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.total_reward = 0
        self.ruleset = RuleSchieber()

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_leaf(self):
        return len(self.children) == 0

    def is_fully_expanded(self):
        obs = self.game_sim.get_observation()
        valid_moves = self.ruleset.get_valid_cards_from_obs(obs)
        return len(valid_moves) == len(self.children)

    def best_child(self, exploration_param=1.41):
        # print("selecting best child")
        # print(self.children)
        best = max(self.children, key=lambda child:
                   (child.total_reward / child.visits) +
                   exploration_param * math.sqrt(math.log(self.visits) / child.visits))
        # print(f"best_cild: {best}\ntotal reward: {best.total_reward}")
        return best