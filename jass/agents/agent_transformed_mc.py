import random
from collections import namedtuple

from jass.agents.Helpers.Node import Node
from jass.agents.agent import Agent
from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.game_rule import GameRule
from jass.game.game_sim import GameSim
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber

class TransformedMCTSAgent(Agent):
    def __init__(self):
        # log actions
        # Use rule object to determine valid actions
        self._rule = RuleSchieber()
        # init random number generator
        self._rng = np.random.default_rng()

    def action_trump(self, obs: GameObservation) -> int:
        """
        Select trump randomly. Pushing is selected with probability 0.5 if possible.
        Args:
            obs: the current game
        Returns:
            trump action
        """
        # if forehand is not yet set, we are the forehand player and can select trump or push
        if obs.forehand == -1:
            if self._rng.choice([True, False]):
                return PUSH
        # if not push or forehand, select a trump
        result = int(self._rng.integers(low=0, high=MAX_TRUMP, endpoint=True))
        return result

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Select randomly a card from the valid cards
        Args:
            obs: The observation of the jass game for the current player
        Returns:
            card to play
        """
        # cards are one hot encoded
        # they start look like this but in [1, 0, 1, 0 ... ,1] etc: DA',
        #     'DK',
        #     'DQ',
        #     'DJ',
        #     'D10',
        #     'D9',
        #     'D8',
        #     'D7',
        #     'D6',
        #     'HA',
        #     'HK',
        #     'HQ',
        #     'HJ',
        #     'H10',
        #     'H9',
        #     'H8',
        #     'H7',
        #     'H6',
        #     'SA',
        #     'SK',
        #     'SQ',
        #     'SJ',
        #     'S10',
        #     'S9',
        #     'S8',
        #     'S7',
        #     'S6',
        #     'CA',
        #     'CK',
        #     'CQ',
        #     'CJ',
        #     'C10',
        #     'C9',
        #     'C8',
        #     'C7',
        #     'C6',
        valid_cards = self._rule.get_valid_cards_from_obs(obs)

        if obs.tricks.all() == -1:
            print("balls")

        # current trick looks like [10, 20, -1, -1]
        if obs.current_trick[0] == -1:
            print("giga Balls")

        #print("nothing")

        # convert to list and draw a value
        card = self._rng.choice(np.flatnonzero(valid_cards))
        return card

# Define the Card namedtuple to represent cards more effectively
Card = namedtuple('Card', ['rank', 'suit'])

# Define the suits and ranks used in Jass
suits = ["H", "D", "C", "S"]
ranks = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]

obe_nabe = [8, 7, 6, 5, 4, 3, 2, 1, 0]
une_ufe = [0, 1, 2, 3, 4, 5, 6, 7, 8]
trump_modifier = [9, 9, 9, 16, 9, 16, 9, 9, 9]

# Create the full deck of cards
deck = [Card(rank, suit) for suit in suits for rank in ranks]


# Define point values for each rank
def card_points(card):
    # TODO use points that are specific for each trump
    points = {"6": 0, "7": 0, "8": 0, "9": 0, "10": 10, "J": 2, "Q": 3, "K": 4, "A": 11}
    return points[card.rank]


# Sample function to generate a random hand of cards
def get_random_hand(deck, hand_size=9):
    return random.sample(deck, hand_size)


# Function to simulate the opponent's play
def simulate_opponent_hand(my_hand, played_cards, hand_size=9):
    remaining_deck = [c for c in deck if c not in my_hand and c not in played_cards]
    return get_random_hand(remaining_deck, hand_size)


# Function to decide if we play the first card or respond to a round started by another player
def is_starting_round(current_round):
    return len(current_round) == 0


# Function to simulate opponent play based on the lead card and strategy
def simulate_opponent_play(opponent_hand, lead_card, trump_suit):
    if lead_card:
        # Follow suit if possible
        follow_suit_cards = [card for card in opponent_hand if card.suit == lead_card.suit]
        if follow_suit_cards:
            return random.choice(follow_suit_cards)
    # Otherwise, play a random card (could add more sophisticated heuristics here)
    return random.choice(opponent_hand)


# Function to simulate a round and estimate win rate and points with strategies
def monte_carlo_simulate(my_hand, num_simulations=100, played_cards=None,
                         current_round=None, trump_suit="Hearts", team_started=False):
    if played_cards is None:
        played_cards = []
    if current_round is None:
        current_round = []

    win_counts = {card: 0 for card in my_hand}
    point_totals = {card: 0 for card in my_hand}

    for card in my_hand:
        for _ in range(num_simulations):
            # Simulate the opponent's hand
            opponent_hand = simulate_opponent_hand(my_hand, played_cards)

            if team_started:
                # If our team started the round, prioritize cards that maximize points
                opponent_play = random.choice(opponent_hand)
            else:
                # If we are responding to a lead, opponents play with a heuristic to maximize their chance of winning
                lead_card = current_round[0] if current_round else None
                opponent_play = simulate_opponent_play(opponent_hand, lead_card, trump_suit)

            # Determine if we are playing the first card or responding to a round
            if not is_starting_round(current_round):
                lead_suit = current_round[0].suit
                if card.suit != lead_suit and any(c.suit == lead_suit for c in my_hand):
                    continue  # Skip if we can't follow suit

            # Heuristic: Estimate if playing 'card' will win the trick
            if card_strength(card, trump_suit) > card_strength(opponent_play, trump_suit):
                win_counts[card] += 1
                point_totals[card] += card_points(card) + card_points(opponent_play)

    # Estimate win rate and average points for each card
    estimated_win_rate = {card: wins / num_simulations for card, wins in win_counts.items()}
    estimated_points = {card: points / num_simulations for card, points in point_totals.items()}

    # Combine win rate and point estimate to make a decision
    combined_scores = {card: (estimated_win_rate[card] * 0.6) + (estimated_points[card] * 0.4) for card in my_hand}

    return combined_scores


# Function to determine the "strength" of a card, with trumps given more value
def card_strength(card, trump_suit="Hearts"):
    rank_value = {"6": 1, "7": 2, "8": 3, "9": 4, "10": 5, "J": 6, "Q": 7, "K": 8, "A": 9}
    base_value = rank_value[card.rank]

    # Trump cards are inherently stronger
    if card.suit == trump_suit:
        if card.rank == "J" or card.rank == "9":
            return base_value + 16
        return base_value + 10  # Add extra value if it's a trump card
    return base_value


# My initial hand of cards
my_hand = get_random_hand(deck)
played_cards = []  # Cards that have already been played
current_round = []  # Cards already played in the current round

# Decide if our team started the round or not
team_started = is_starting_round(current_round)

# Estimate combined scores for each card using Monte Carlo simulation with strategies
estimated_scores = monte_carlo_simulate(my_hand, played_cards=played_cards, current_round=current_round,
                                        team_started=team_started)

# Choose the best card to play based on combined scores
best_card = max(estimated_scores, key=estimated_scores.get)

print(f"My hand: {[f'{card.rank} of {card.suit}' for card in my_hand]}")
print(f"Estimated scores: {estimated_scores}")
print(f"Best card to play: {best_card.rank} of {best_card.suit}")
