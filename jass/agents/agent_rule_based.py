# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
from lib2to3.pgen2.tokenize import printtoken

import numpy as np
from jass.agents.agent import Agent
from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber


class AgentRuleBased(Agent):
    """
    Select rule based actions for the game of jass (Schieber)
    """
    trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
    no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
    obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0]
    uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rng = np.random.default_rng()
        self._rule = RuleSchieber()
        self._logger = logging.getLogger(__name__)

    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        # add your code here using the function above
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
        #print("Trump selection scores:", trump_scores)
        #print("Selected trump result:", result)
        #print("Push value", PUSH)

        if max(trump_scores) < 50 and obs.forehand == -1:  # Adjust the threshold as needed
            self._logger.info('Result: {}'.format(PUSH))
            return PUSH

        self._logger.info('Result: {}'.format(result))
        print(result)
        if result == 0:
            return DIAMONDS
        if result == 1:
            return HEARTS
        if result == 2:
            return SPADES
        if result == 3:
            return CLUBS
        if result == 4:
            return OBE_ABE
        if result == 5:
            return UNE_UFE

        return result

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

        score = 0
        for i in range(9):
            card_index = suit * 9 + i
            if cards[card_index] == 1:
                # Determine score based on trump type
                if trump <= 3:  # One of the suits (Diamonds, Hearts, Spades, Clubs)
                    if suit == trump:
                        score += self.trump_score[i]
                    else:
                        continue  # Skip non-trump suits when evaluating a specific trump
                elif trump == 4:  # Obenabe
                    score += self.obenabe_score[i]
                elif trump == 5:  # Uneufe
                    score += self.uneufe_score[i]

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

        for i in range(4):  # Iterate over the four suits
            consecutive_high_cards = 0  # Track consecutive high cards
            for j in range(9):  # Iterate over ranks in the suit
                card_index = i * 9 + j
                if hand[card_index] == 1:
                    # If the card belongs to the trump suit
                    if trump <= 3:  # One of the suits (Diamonds, Hearts, Spades, Clubs)
                        # High-value cards are guaranteed winning cards (bocks)
                        if self.trump_score[j] >= 15:  # Arbitrary threshold for "guaranteed game"
                            guaranteed_games += 1
                            consecutive_high_cards += 1
                        elif consecutive_high_cards >= 3:  # Forced play scenario - next card is likely to win
                            guaranteed_games += 1
                    elif trump == 4:  # Obenabe
                        # High cards are guaranteed in obenabe
                        if self.obenabe_score[j] >= 7:  # More flexible threshold for "guaranteed game" in obenabe
                            guaranteed_games += 1
                            consecutive_high_cards += 1
                        elif consecutive_high_cards >= 3:  # Forced play scenario
                            guaranteed_games += 1
                    elif trump == 5:  # Uneufe
                        # Low cards are guaranteed in uneufe
                        if self.uneufe_score[j] >= 7:  # Arbitrary threshold for "guaranteed game" in uneufe
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
        Determine the card to play.

        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)

        card_to_play = 0

        # checks if we have the highest trump and if we do we play it
        if obs.declared_trump < 4:
            player_tricks = self.get_trump_segment(obs.hand, obs.declared_trump)
            cards_still_in_play = self.get_cards_still_in_play(obs)

            possible_tricks = self.get_trump_availability(cards_still_in_play, obs)
            highest_player_trick = 0
            trick_index = 0
            for i in player_tricks:
                if player_tricks[i] > 0:
                    if self.trump_score[i] > highest_player_trick:
                        highest_player_trick = player_tricks[i]
                        trick_index = i
            for i in possible_tricks:
                if possible_tricks[i] > 0 and player_tricks[i] < 1:
                    if self.trump_score[i] > highest_player_trick:
                        highest_player_trick = 0

            if highest_player_trick > 0:
                # print("Uses highest player trick")
                card_to_play = (trick_index + 9 * obs.declared_trump)

                if valid_cards[card_to_play] == 1:
                    return card_to_play
                else:
                    card_to_play = (trick_index + 9 * obs.declared_trump)
                    if valid_cards[card_to_play] == 1:
                        return card_to_play

        cards_still_in_play = self.get_cards_still_in_play(obs)
        # print(cards_still_in_play)
        # card_to_play = self._rng.choice(np.flatnonzero(valid_cards))
        card_to_play = self.check_for_bock(obs)

        #print(valid_cards)
        #print(card_to_play)
        if valid_cards[card_to_play] == 1:
            return card_to_play
        else:
            card_to_play = self.play_highest_card(valid_cards)
            if valid_cards[card_to_play] == 1:
                return card_to_play
            card = self._rng.choice(np.flatnonzero(valid_cards))
            return card

    def get_cards_still_in_play(self, obs):
        possible_cards = [i for i in range(36)]

        possible_cards = set(possible_cards) ^ set(convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand))
        if len(obs.tricks[obs.tricks > 0]):
            possible_cards = set(possible_cards) ^ set(obs.tricks[obs.tricks != -1])

        return possible_cards

    def check_for_bock(self, obs):
        # check if there are cards that are valid that aren't trump
        # if we have the highest value card
        # we chck if there are tricks left that aren't ours
        # if there are we check if we have the next highest card as well and play that instead
        # if we dont have the highest card we play the lowest point value card
        # Step 1: Get valid cards that can be played
        valid_cards = self._rule.get_valid_cards_from_obs(obs)

        # Step 2: Check for trump suit validity and determine the highest card
        is_trump = obs.declared_trump < 4  # Check if current game mode is trump (0-3)
        highest_card = None
        lowest_card = None
        lowest_card_value = float('inf')

        # Step 3: Identify the highest and lowest cards among valid cards
        if obs.declared_trump != 5:
            for card in range(len(valid_cards)):
                if valid_cards[card] == 1:  # Card is playable
                    # Card rank in suit
                    rank_in_suit = card % 9
                    # Determine if it's a trump card and its score
                    if is_trump and (card // 9 == obs.declared_trump):  # Trump card
                        card_value = self.trump_score[rank_in_suit]
                    else:  # Non-trump card
                        card_value = self.no_trump_score[rank_in_suit]

                    # Update highest card if this card has a higher value
                    if highest_card is None or card_value > self.no_trump_score[highest_card % 9]:
                        highest_card = card

                    # Update lowest card if this card has a lower value
                    if card_value < lowest_card_value:
                        lowest_card_value = card_value
                        lowest_card = card
        else:
            for card in range(len(valid_cards)):
                if valid_cards[card] == 1:  # Card is playable
                    # Card rank in suit
                    rank_in_suit = card % 9

                    card_value = self.uneufe_score[rank_in_suit]

                    # Update highest card if this card has a higher value
                    if highest_card is None or card_value > self.no_trump_score[highest_card % 9]:
                        highest_card = card

                    # Update lowest card if this card has a lower value
                    if card_value < lowest_card_value or lowest_card is None:
                        lowest_card_value = card_value
                        lowest_card = card
        # Step 4: Decide which card to play
        if highest_card is not None:
            # If we hold the highest card in the current trick context, play it
            return highest_card
        else:
            # Otherwise, play the lowest value card
            return lowest_card


    def get_trump_segment(self, card_set, trump_segment):
        card_array = np.array(list(card_set))
        start = (trump_segment - 1) * 9  # Calculate the starting index
        end = start + 9  # Calculate the end index (9 cards per segment)
        return card_array[start:end]

    def get_trump_availability(self, available_trumps, obs: GameObservation):
        # Create an array of length 36 with all elements initialized to 0
        trump_availability = [0] * 36

        # Set each index to 1 if it exists in the available_trumps set
        for index in available_trumps:
            trump_availability[index] = 1

        # Get the start and end indices based on trump_segment
        start = (obs.declared_trump - 1) * 9
        end = start + 9

        # Return only the segment corresponding to the given trump segment
        return trump_availability[start:end]

    def play_highest_card(self, valid_cards):
        highest_score = -1
        card_to_play = 0

        # Iterate over all 36 cards
        for i in range(len(valid_cards)):
            # Only consider cards that are valid (value of 1 in valid_cards)
            if valid_cards[i] == 1:
                # Calculate index within no_trump_score (i % 9 gives position within suit)
                score_index = i % 9
                card_score = self.no_trump_score[score_index]

                # Check if this card has the highest score so far
                if card_score > highest_score:
                    highest_score = card_score
                    card_to_play = i

        return card_to_play