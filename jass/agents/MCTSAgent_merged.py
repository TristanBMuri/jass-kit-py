# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import random
import copy

import numpy as np
import time

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
    def __init__(self, time_per_action=2):
        super().__init__()
        self.time_per_action = time_per_action
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

        """
                Select the trump suit based on the highest Monte Carlo scores for each possible trump.
                Args:
                    obs: the current game observation
                Returns:
                    trump action
                """

        # Initialize the trump scores for each possible trump suit
        trump_scores = {DIAMONDS: 0, HEARTS: 0, SPADES: 0, CLUBS: 0, OBE_ABE: 0, UNE_UFE: 0}

        # Evaluate each trump by running Monte Carlo simulations
        time_per_trump = self.time_per_action / len(trump_scores)
        for trump_suit in trump_scores.keys():
            trump_scores[trump_suit] = sum(self.monte_carlo_simulate(
                my_hand=np.flatnonzero(obs.hand),
                time_limit=time_per_trump,
                played_cards=[card for trick in obs.tricks for card
                              in trick if card != -1],
                current_round=obs.current_trick[
                    obs.current_trick != -1],
                trump_suit=trump_suit).values())

        # Choose the trump with the highest score
        best_trump = max(trump_scores, key=trump_scores.get)
        print(trump_scores)
        if obs.forehand == -1:
            # If no trump has a reasonable score, push
            if trump_scores[best_trump] < 26:  # Threshold
                return PUSH
        print(f"Playing trump {best_trump}")
        return best_trump

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play using MCTS, with a fixed time budget for analysis.
        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        team_id = obs.player_view % 2
        mcts = MCTS(obs.player_view, team_id)
        game_sim = self.determinize_observation(obs)
        root = Node(game_sim=copy.deepcopy(game_sim))

        # Get all valid moves from the current observation
        valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
            RuleSchieber().get_valid_cards_from_obs(obs)
        )

        if len(valid_moves) == 1:
            return valid_moves[0]

        total_time = self.time_per_action  # seconds
        start_time = time.time()
        time_per_move = total_time / len(valid_moves)  # Divide time equally among moves

        for move in valid_moves:
            move_start_time = time.time()
            move_node = Node(game_sim=copy.deepcopy(game_sim), parent=root, move=move)
            root.add_child(move_node)
            simulations_explored = 0

            while time.time() - move_start_time < time_per_move:
                simulations_explored += 1
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

                if time.time() - start_time >= total_time:
                    break
            print(f"move {move}; simulations explored: {simulations_explored}")

        best_child = root.best_child(exploration_param=2)
        return best_child.move if best_child else None

    def monte_carlo_simulate(self, my_hand, trump_suit, time_limit=5.0, played_cards=None, current_round=None,
                             team_started=None):
        """
        Perform Monte Carlo simulations to estimate the best card to play based on the given hand.
        Args:
            my_hand: List of card IDs representing the player's hand
            time_limit: Maximum time allowed for simulations (in seconds)
            played_cards: List of cards already played in the game
            current_round: Cards already played in the current trick
            trump_suit: The trump suit for the current game
            team_started: Whether the player's team started the round
        Returns:
            A dictionary of estimated scores for each card in my_hand
        """
        if played_cards is None:
            played_cards = []
        if current_round is None:
            current_round = []

        win_counts = {card: 0 for card in my_hand}
        point_totals = {card: 0 for card in my_hand}

        total_time = time_limit  # total time budget
        start_time = time.time()
        time_per_card = total_time / my_hand.size if my_hand.size > 0 else 0


        for card in my_hand:
            card_start_time = time.time()
            simulations_explored = 0

            while time.time() - card_start_time < time_per_card:
                simulations_explored += 1

                # Simulate the opponent's hand by excluding played cards and current hand
                opponent_hand = self.simulate_opponent_hand(my_hand, played_cards)

                if team_started:
                    # If our team started the round, prioritize cards that maximize points
                    opponent_play = random.choice(opponent_hand)
                else:
                    # If we are responding to a lead, opponents play with a heuristic to maximize their chance of winning
                    lead_card = current_round[0] if current_round else None
                    opponent_play = self.simulate_opponent_play(opponent_hand, lead_card, trump_suit, current_round)

                # Determine if we are playing the first card or responding to a round
                if current_round:
                    lead_suit = self.get_suit(current_round[0])
                    if self.get_suit(card) != lead_suit and any(self.get_suit(c) == lead_suit for c in my_hand):
                        continue  # Skip if we can't follow suit

                # Estimate if playing 'card' will win the trick
                if opponent_play is not None and self.card_strength(card, trump_suit) > self.card_strength(
                        opponent_play, trump_suit):
                    win_counts[card] += 1
                    point_totals[card] += self.card_points(card, trump_suit) + self.card_points(opponent_play,
                                                                                                trump_suit)

                # Stop if the total allocated time is exceeded
                if time.time() - start_time >= total_time:
                    break

            print(f"Card {card}; simulations explored: {simulations_explored}")

        # Estimate win rate and average points for each card
        estimated_win_rate = {card: wins / max(1, sum(win_counts.values())) for card, wins in win_counts.items()}
        estimated_points = {card: points / max(1, sum(win_counts.values())) for card, points in point_totals.items()}

        # Combine win rate and point estimate to make a decision
        combined_scores = {card: (estimated_win_rate[card] * 0.6) + (estimated_points[card] * 0.4) for card in my_hand}

        return combined_scores

    def simulate_opponent_hand(self, my_hand, played_cards, hand_size=9, total_players=4):
        """
        Simulate the opponent's hand by removing cards that are in the player's hand or already played.
        Args:
            my_hand: List of card IDs representing the player's hand
            played_cards: List of card IDs representing already played cards
            hand_size: Number of cards to generate for the opponent's hand
        Returns:
            A list of card IDs representing the opponent's hand
        """
        remaining_deck = list(filter(lambda card: card not in my_hand and card not in played_cards, range(36)))
        remaining_cards = len(remaining_deck)
        # Calculate hand size per opponent based on remaining cards and number of players
        cards_per_opponent = remaining_cards // (total_players - 1)
        leftover_cards = remaining_cards % (total_players - 1)
        opponent_hands = []

        for i in range(total_players - 1):
            size = cards_per_opponent + (1 if i < leftover_cards else 0)
            opponent_hand = random.sample(remaining_deck, size)
            opponent_hands.append(opponent_hand)
            remaining_deck = [card for card in remaining_deck if card not in opponent_hand]

        # Check that together with the played cards, everyone has the same number of cards
        total_cards = len(my_hand) + len(played_cards)
        for hand in opponent_hands:
            total_cards += len(hand)

        assert total_cards == 36, "Card distribution error: total cards do not sum to the full deck"

        # Flatten the list of opponent hands to simulate a combined opponent hand for Monte Carlo
        return [card for hand in opponent_hands for card in hand]

    def simulate_opponent_play(self, opponent_hand, lead_card, trump_suit, current_round):
        """
        Simulate the play of an opponent based on a lead card.
        Args:
            opponent_hand: List of card IDs representing the opponent's hand
            lead_card: The card that has been led in the current trick
            trump_suit: The trump suit for the current game
        Returns:
            A card ID representing the opponent's play
        """
        if not opponent_hand:
            return None  # If opponent hand is empty, return None to indicate no valid card to play

        # Follow suit if possible
        if lead_card is not None:
            follow_suit_cards = [card for card in opponent_hand if self.get_suit(card) == self.get_suit(lead_card)]
            if follow_suit_cards:
                return random.choice(follow_suit_cards)

        # If no cards of the lead suit are available, decide if playing a trump is beneficial
        trump_cards = [card for card in opponent_hand if self.get_suit(card) == trump_suit]
        if trump_cards:
            # Play a trump card only if we are likely to lose the trick with other cards
            non_trump_cards = [card for card in opponent_hand if self.get_suit(card) != trump_suit]
            if not non_trump_cards or lead_card is not None:
                # Additionally, only play a trump if there are significant points in the trick
                if lead_card is not None and (self.card_points(lead_card, trump_suit) > 0 or
                                              any(self.card_points(card, trump_suit) > 0 for card in current_round)):
                    return random.choice(trump_cards)

        # Otherwise, play a non-trump card (if available)
        non_trump_cards = [card for card in opponent_hand if self.get_suit(card) != trump_suit]
        if non_trump_cards:
            return random.choice(non_trump_cards)

        # If no non-trump cards are available, play any card
        if non_trump_cards:
            return random.choice(opponent_hand)
        return None

    def card_strength(self, card, trump_suit):
        """
        Determine the "strength" of a card, giving higher value to trump cards.
        Args:
            card: Card ID representing the card to evaluate
            trump_suit: The trump suit for the current game
        Returns:
            An integer representing the strength of the card
        """
        rank_value = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Strength values for ranks from 6 to Ace
        base_value = rank_value[card % 9]

        # Trump cards are inherently stronger
        if self.get_suit(card) == trump_suit:
            if card % 9 == 6 or card % 9 == 3:  # Jack or 9
                return base_value + 16
            return base_value + 10  # Add extra value if it's a trump card
        return base_value

    def card_points(self, card, trump_suit):
        """
        Determine the points associated with a card, taking into account the current trump suit.
        Args:
            card: Card ID representing the card to evaluate
            trump_suit: The trump suit for the current game
        Returns:
            An integer representing the points of the card
        """
        if trump_suit in [0, 1, 2, 3]:  # Standard trump suits (Diamonds, Hearts, Spades, Clubs)
            if self.get_suit(card) == trump_suit:
                if card % 9 == 3:  # Jack
                    return 20
                elif card % 9 == 5:  # 9
                    return 14
        elif trump_suit == 4:  # Obenabe
            if card % 9 == 6:  # 8 of any suit
                return 8
        elif trump_suit == 5:  # Uneufe
            if card % 9 == 0:  # Ace
                return 0
            elif card % 9 == 8:  # 6 of any suit
                return 11
            elif card % 9 == 6:  # 8 of any suit
                return 8

        # Default points for non-trump cards
        points = [0, 0, 0, 0, 10, 2, 3, 4, 11]  # Points for each rank, from 6 to Ace
        return points[card % 9]

    def get_suit(self, card):
        """
        Get the suit of a card based on its ID.
        Args:
            card: Card ID representing the card to evaluate
        Returns:
            An integer representing the suit of the card
        """
        return card // 9


