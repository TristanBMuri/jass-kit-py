# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import numpy as np
from jass.agents.agent import Agent
from jass.game.const import PUSH, MAX_TRUMP, card_strings
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber


class AgentRuleBased(Agent):
    """
    Randomly select actions for the game of jass (Schieber)
    """

    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()
        self._logger = logging.getLogger(__name__)


    def calculate_trump_selection_score(self, cards, trump: int) -> int:
        # score if the color is trump
        trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
        # score if the color is not trump
        no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
        # score if obenabe is selected (all colors)
        obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0, ]
        # score if uneufe is selected (all colors)
        uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]
        # Initialize trump selection score
        trump_selection_score = 0

        # Print debug information
        # print("cards:", cards, "trump:", trump)

        for i in range(9):  # Loop through 9 cards
            card_suit = cards[i] // 9  # Get the suit of the card (0 to 3)
            card_rank = cards[i] % 9  # Get the rank of the card (0 to 8)

            # If the card suit matches the trump suit, use trump_score
            if trump <= 3:
                if card_suit == trump:
                    trump_selection_score += trump_score[card_rank]
                    self._logger.info(f"Card {i}: suit={card_suit}, rank={card_rank}, current score={trump_selection_score}")
                else:
                    # Use a different score array when the card is not from the trump suit
                    trump_selection_score += no_trump_score[card_rank]  # Example for handling non-trump scores
            elif trump == 4:
                trump_selection_score += obenabe_score[card_rank]
            elif trump == 5:
                trump_selection_score += uneufe_score[card_rank]

            # Debug output to trace computation
            print(f"Card {i}: suit={card_suit}, rank={card_rank}, current score={trump_selection_score}")

        return trump_selection_score

    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        # add your code here using the function above
        if obs.forehand == -1:
            # if forehand is not yet set, we are the forehand player and can select trump or push
            trump_scores = [0] * 6
            for i in trump_scores:
                trump_scores[i] = self.calculate_trump_selection_score(obs.hand, i)

            # if not push or forehand, select a trump
            result = int(trump_scores.index(max(trump_scores)))
            print("result:", result, "5", trump_scores.index(max(trump_scores)))
            if result < 50:
                self._logger.info('Result: {}'.format(PUSH))
                return PUSH

            self._logger.info('Result: {}'.format(result))
            return result

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play.

        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        # we use the global random number generator here
        return np.random.choice(np.flatnonzero(valid_cards))