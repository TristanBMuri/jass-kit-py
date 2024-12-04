# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import random
from copy import deepcopy

import numpy as np

from jass.agents.Helpers.MCTS import MCTS
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
        self.max_iterations = max_iterations
        self._rule = RuleSchieber()

    def generate_opponent_hands(self, obs):
        random.seed(42)
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
            """
            Determine trump action for the given observation
            Args:
                obs: the game observation, it must be in a state for trump selection

            Returns:
                selected trump as encoded in jass.game.const or jass.game.const.PUSH
            """
            # add your code here using the function above
            # if obs.forehand == -1:
            #     # If forehand is not yet set, we are the forehand player and can select trump or push
            #     trump_scores = [0] * 6
            #     for i in range(6):
            #         trump_scores[i] = self.calculate_gap_consideration(obs.hand, i) * 5
            #         if i <= 3:  # For suits (Diamonds, Hearts, Spades, Clubs)
            #             trump_scores[i] += self.calculate_score_for_suit(obs.hand, i, i)
            #         elif i == 4:  # Obenabe
            #             for suit in range(4):
            #                 trump_scores[i] += self.calculate_score_for_suit(obs.hand, i, suit)
            #         elif i == 5:  # Uneufe
            #             for suit in range(4):
            #                 trump_scores[i] += self.calculate_score_for_suit(obs.hand, i, suit)
            #
            #     # Select the trump with the highest score
            #     result = int(trump_scores.index(max(trump_scores)))
            #     print("Trump selection scores:", trump_scores)
            #     print("Selected trump result:", result)
            #
            #     if max(trump_scores) < 50:  # Adjust the threshold as needed
            #         return PUSH

            return DIAMONDS

    def calculate_score_for_suit(self, cards, trump: int, suit: int) -> int:
        """
        Calculate the score for a given suit as trump.
        Args:
            cards: the hand of cards, represented as a list of 36 integers (one-hot encoded)
            trump: the current suit being evaluated as trump (0-5)
            suit: the suit being evaluated

        Returns:
            score for the selected suit
        """
        # Scoring systems
        trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
        no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
        obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0]
        uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

        score = 0
        for i in range(9):
            card_index = suit * 9 + i
            if cards[card_index] == 1:
                # Determine score based on trump type
                if trump <= 3:  # One of the suits (Diamonds, Hearts, Spades, Clubs)
                    if suit == trump:
                        score += trump_score[i]
                    else:
                        continue  # Skip non-trump suits when evaluating a specific trump
                elif trump == 4:  # Obenabe
                    score += obenabe_score[i]
                elif trump == 5:  # Uneufe
                    score += uneufe_score[i]

        return score

    def calculate_guaranteed_games(self, hand, trump: int) -> int:
        """
        Calculate the number of guaranteed games ("bocks") given the trump selection.
        Args:
            hand: the hand of cards, represented as a list of 36 integers (one-hot encoded)
            trump: the suit selected as trump (0-5)

        Returns:
            Number of guaranteed winning cards (bocks)
        """
        guaranteed_games = 0
        trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]  # Assume standard trump score
        obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0]
        uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

        for i in range(4):  # Iterate over the four suits
            consecutive_high_cards = 0  # Track consecutive high cards
            for j in range(9):  # Iterate over ranks in the suit
                card_index = i * 9 + j
                if hand[card_index] == 1:
                    # If the card belongs to the trump suit
                    if trump <= 3:  # One of the suits (Diamonds, Hearts, Spades, Clubs)
                        # High-value cards are guaranteed winning cards (bocks)
                        if trump_score[j] >= 15:  # Arbitrary threshold for "guaranteed game"
                            guaranteed_games += 1
                            consecutive_high_cards += 1
                        elif consecutive_high_cards >= 3:  # Forced play scenario - next card is likely to win
                            guaranteed_games += 1
                    elif trump == 4:  # Obenabe
                        # High cards are guaranteed in obenabe
                        if obenabe_score[j] >= 7:  # More flexible threshold for "guaranteed game" in obenabe
                            guaranteed_games += 1
                            consecutive_high_cards += 1
                        elif consecutive_high_cards >= 3:  # Forced play scenario
                            guaranteed_games += 1
                    elif trump == 5:  # Uneufe
                        # Low cards are guaranteed in uneufe
                        if uneufe_score[j] >= 7:  # Arbitrary threshold for "guaranteed game" in uneufe
                            guaranteed_games += 1
                            consecutive_high_cards += 1
                        elif consecutive_high_cards >= 3:  # Forced play scenario
                            guaranteed_games += 1
                else:
                    # If there is a break in the sequence of high cards
                    consecutive_high_cards = 0

            # Add guaranteed game count for forced plays if there are sufficient high cards
            if consecutive_high_cards >= 4:
                guaranteed_games += 1  # Add for the next card in the sequence being forced to win

        return guaranteed_games

    def calculate_gap_consideration(self, hand, trump: int) -> int:
        """
        Calculate the gap consideration where certain cards might be forced.
        Args:
            hand: the hand of cards, represented as a list of 36 integers (one-hot encoded)
            trump: the suit selected as trump (0-5)

        Returns:
            Modified guaranteed game score considering forced plays.
        """
        guaranteed_games = self.calculate_guaranteed_games(hand, trump)
        gap_penalty = 0

        for i in range(4):  # Iterate over suits
            consecutive_high_cards = 0
            for j in range(9):  # Iterate over ranks in the suit
                card_index = i * 9 + j
                if hand[card_index] == 1:
                    # If the card belongs to the trump suit, check for gaps
                    if (i == trump and trump <= 3) or (trump == 4 or trump == 5):
                        consecutive_high_cards += 1
                        # Assume that lower value cards without higher cards could be forced and result in a loss
                        if (j >= 1 and j <= 5) and hand[i * 9 + (j - 1)] == 0:  # Example of a gap penalty
                            # Only apply penalty if there aren't enough high cards to force a win
                            if consecutive_high_cards < 3:
                                gap_penalty += 1
                else:
                    consecutive_high_cards = 0

        return guaranteed_games - gap_penalty

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play using MCTS.
        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        print(f"playerview: {obs.player_view}")
        mcts = MCTS(obs.player_view)
        game_sim = self.determinize_observation(obs)
        root = Node(game_sim=deepcopy(game_sim))


        for i in range(self.max_iterations):
            print(f"root hands:\n {np.sum(root.game_sim.state.hands)}")
            print(f"Iteration {i}")
            leaf = mcts.selection(root)
            child = mcts.expansion(leaf)
            if child:
                outcome = mcts.simulate(child.game_sim)
                mcts.backpropagate(child, outcome)
            else:
                outcome = mcts.simulate(leaf.game_sim)
                mcts.backpropagate(leaf, outcome)

        # print("unplayed cards", self.possible_cards(obs))

        best_move = root.best_child(exploration_param=2).move
        return best_move
        # return np.random.choice(np.flatnonzero(valid_cards))

