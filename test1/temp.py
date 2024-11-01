# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import numpy as np
import unittest


class AgentRuleBased:
    """
    Select rule based actions for the game of jass (Schieber)
    """

    def __init__(self):
        # Set up logging
        self._logger = logging.getLogger(__name__)

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

                # Debug output to trace computation
                self._logger.debug(f"Suit {suit}, Rank {i}, Current Score: {score}")

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

    def action_trump(self, obs) -> int:
        """
        Determine trump action for the given observation
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        if obs.forehand == -1:
            # If forehand is not yet set, we are the forehand player and can select trump or push
            trump_scores = [0] * 6
            for i in range(6):
                trump_scores[i] = self.calculate_gap_consideration(obs.hand, i) * 5
                if i <= 3:  # For suits (Diamonds, Hearts, Spades, Clubs)
                    trump_scores[i] += self.calculate_score_for_suit(obs.hand, i, i)
                elif i == 4:  # Obenabe
                    for suit in range(4):
                        trump_scores[i] += self.calculate_score_for_suit(obs.hand, i, suit)
                elif i == 5:  # Uneufe
                    for suit in range(4):
                        trump_scores[i] += self.calculate_score_for_suit(obs.hand, i, suit)

            # Select the trump with the highest score
            result = int(trump_scores.index(max(trump_scores)))
            print("Trump selection scores:", trump_scores)
            print("Selected trump result:", result)

            if max(trump_scores) < 50:  # Adjust the threshold as needed
                self._logger.info('Result: {}'.format(PUSH))
                return PUSH

            self._logger.info('Result: {}'.format(result))
            return result


class TestAgentRuleBased(unittest.TestCase):

    def setUp(self):
        self.agent = AgentRuleBased()

    def test_calculate_guaranteed_games_hand_1(self):
        # Test hand 1: (Diamonds: A, K, 9, 6; Hearts: A, Q, J, 8, 7)
        hand = [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0]
        trump = 1  # Hearts
        expected_guaranteed_games = 5  # Expected number of guaranteed games based on high cards
        result = self.agent.calculate_guaranteed_games(hand, trump)
        self.assertEqual(result, expected_guaranteed_games)

    def test_calculate_guaranteed_games_hand_2(self):
        # Test hand 2: (Diamonds: A, 10, J; Hearts: K, Q; Spades: A, K, 10, 7)
        hand = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0]
        trump = 2  # Spades
        expected_guaranteed_games = 4  # Expected number of guaranteed games based on high cards
        result = self.agent.calculate_guaranteed_games(hand, trump)
        self.assertEqual(result, expected_guaranteed_games)

    def test_calculate_guaranteed_games_hand_3(self):
        # Test hand 3: (Hearts: A, K, Q, J, 9, 8, 7; Spades: 10; Clubs: A)
        hand = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0]
        trump = 4  # Obenabe
        expected_guaranteed_games = 7  # Expected number of guaranteed games based on high cards
        result = self.agent.calculate_guaranteed_games(hand, trump)
        self.assertEqual(result, expected_guaranteed_games)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
