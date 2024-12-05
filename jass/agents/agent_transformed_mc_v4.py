import random
from collections import namedtuple

from jass.agents.agent import Agent
from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.game_rule import GameRule
from jass.game.game_sim import GameSim
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber


class Node_V4:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.action = action

    def is_fully_expanded(self):
        return len(self.children) == len(self.get_legal_actions())

    def is_terminal(self):
        # Assuming terminal state means all cards have been played
        _, played_cards, current_round = self.state
        return len(played_cards) + len(current_round) == 36

    def expand(self):
        legal_actions = self.get_legal_actions()
        for action in legal_actions:
            if action not in [child.action for child in self.children]:
                next_state = self.simulate_action(action)
                child_node = Node_V4(state=next_state, parent=self, action=action)
                self.children.append(child_node)
                return child_node
        return None

    def simulate_action(self, action):
        # Simulate taking the given action and return the resulting state
        my_hand, played_cards, current_round = self.state

        # Convert played_cards and current_round to lists if they are numpy arrays
        if isinstance(played_cards, np.ndarray):
            played_cards = played_cards.tolist()
        if isinstance(current_round, np.ndarray):
            current_round = current_round.tolist()

        # Update new_hand, new_played_cards, and new_current_round
        new_hand = [card for card in my_hand if card != action]
        new_played_cards = played_cards + current_round + [action]
        new_current_round = [] if len(current_round) == 3 else current_round + [action]

        return (new_hand, new_played_cards, new_current_round)

    def get_legal_actions(self):
        # Return the legal actions available from this state
        my_hand, _, current_round = self.state
        if len(current_round) > 0:
            #TODO trump cards are also valid if follow suit card exists
            lead_suit = current_round[0] // 9
            return [card for card in my_hand if card // 9 == lead_suit] or my_hand
        return my_hand


class TransformedMCTSAgent(Agent):
    def get_valid_actions(self, my_hand, current_round):
        """
        Determine the valid actions (cards that can be played) based on the current hand and round.
        Args:
            my_hand: List of card IDs representing the player's hand
            current_round: List of card IDs that have been played in the current trick so far
        Returns:
            A list of card IDs that can be legally played
        """
        if not len(current_round) > 0:
            # If no cards have been played in the current round, all cards in hand are valid
            return my_hand

        lead_suit = self.get_suit(current_round[0])
        # Follow suit if possible
        follow_suit_cards = [card for card in my_hand if self.get_suit(card) == lead_suit]
        #TODO trump cards are also valid if follow suit card exists
        if follow_suit_cards:
            return follow_suit_cards

        # If no cards of the lead suit are available, all cards can be played
        return my_hand
    
    def monte_carlo_tree_search(self, root: Node_V4, max_depth: int):
        """
        Perform a Monte Carlo Tree Search (MCTS) from the given root node up to a specified depth.
        Args:
            root: The root node from which to start the MCTS
            max_depth: The maximum depth to which the search should expand
        Returns:
            The best action based on the MCTS evaluation
        """
        for _ in range(100):  # Loop to perform fixed number of simulations for consistent depth  # Loop to perform multiple simulations from the root node
            # Selection: Traverse from root to leaf using UCT
            current_node = root
            depth = 0
            while current_node.is_fully_expanded() and depth < max_depth:
                current_node = self.select_best_node_uct(current_node)
                depth += 1

            # Expansion: Expand a child if the node is not terminal
            if depth < max_depth and not current_node.is_terminal():
                current_node = current_node.expand()

            # Simulation: Simulate a rollout from this node
            value = self.simulate(current_node.state, root.state[1], max_depth - depth)

            # Backpropagation: Propagate the value up to the root
            self.backpropagate(current_node, value)

        if not root.children:
            # Handle the scenario where no children exist
            return None

        # Return the action leading to the best average score
        best_child = max(root.children, key=lambda child: child.value / (child.visits + 1e-5))
        print("best child:", best_child)
        return best_child.action

    def select_best_node_uct(self, node: Node_V4, exploration_param=1.41):
        """
        Select the best child node based on UCT (Upper Confidence Bound for Trees).
        Args:
            node: The current node from which to select the next move
            exploration_param: The exploration parameter that balances exploration vs. exploitation
        Returns:
            The child node with the highest UCT score
        """
        if not node.children:
            return node  # Return the current node as a fallback

        best_child = max(node.children, key=lambda child:
                         (child.value / (child.visits + 1e-5)) +
                         exploration_param * (np.sqrt(np.log(node.visits + 1) / (child.visits + 1e-5))))
        return best_child

    def backpropagate(self, node: Node_V4, value: float):
        """
        Backpropagate the simulation result up to the root.
        Args:
            node: The node from which to start backpropagation
            value: The value obtained from the simulation
        """
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            current_node.value += value
            current_node = current_node.parent

    def simulate(self, state, played_cards, depth_remaining):
        """
        Simulate a game from the given state until a terminal state or depth limit is reached.
        Args:
            state: The current state to simulate from
            played_cards: The list of cards that have been played in the game
            depth_remaining: The remaining depth to simulate
        Returns:
            A value representing the result of the simulation
        """
        my_hand, played_cards, current_round = state
        points = 0
        for _ in range(depth_remaining):
            if len(played_cards) + len(current_round) >= 36:
                break  # Terminal state, game is over
            # Simulate an action
            valid_actions = self.get_valid_actions(my_hand, current_round)
            if not valid_actions:
                break
            action = random.choice(valid_actions)
            my_hand.remove(action)
            current_round = np.append(current_round, action)
            points += self.card_points(action, played_cards)
        return points

    def get_dynamic_simulation_count(self, played_cards, current_round):
        """
        Determine the number of simulations dynamically based on the game state.
        Args:
            played_cards: List of cards already played in the game
            current_round: Cards already played in the current trick
        Returns:
            The number of simulations to perform
        """
        remaining_cards = 36 - len(played_cards) - len(current_round)
        if remaining_cards > 20:
            return 200  # Early in the game, use more simulations for better accuracy
        elif remaining_cards > 10:
            return 100  # Mid game, balance between accuracy and speed
        else:
            return 50   # Late game, fewer simulations to speed up decisions

    def __init__(self):
        # Use rule object to determine valid actions
        self._rule = RuleSchieber()
        # init random number generator
        self._rng = np.random.default_rng()

        # own obs
        self.global_obs = GameObservation()


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
        for trump_suit in trump_scores.keys():
            trump_scores[trump_suit] = sum(self.monte_carlo_trump(my_hand=np.flatnonzero(obs.hand),
                                                                     num_simulations=100,
                                                                     played_cards=[card for trick in obs.tricks for card
                                                                                   in trick if card != -1],
                                                                     current_round=obs.current_trick[
                                                                         obs.current_trick != -1],
                                                                     trump_suit=trump_suit).values())

        # Choose the trump with the highest score
        best_trump = max(trump_scores, key=trump_scores.get)

        if obs.forehand == -1:
            # If no trump has a reasonable score, push
            if trump_scores[best_trump] < 100:  # Threshold can be adjusted
                return PUSH

        return best_trump

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Select the card to play using Monte Carlo simulations to estimate the best option.
        Args:
            obs: The observation of the jass game for the current player
        Returns:
            card to play
        """
        self.global_obs = obs

        # Get the valid cards the player can play from the current observation
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        my_hand = np.flatnonzero(valid_cards)  # Convert one-hot encoded valid cards to card IDs

        # Extract the played cards from the history of all tricks so far
        played_cards = [card for trick in obs.tricks for card in trick if card != -1]
        current_round = obs.current_trick[
            obs.current_trick != -1]  # Get the currently played cards in the ongoing trick


        # Use Monte Carlo simulation to determine the card to play
        team_started = (obs.current_trick.tolist().count(-1) % 2 == 0)
        num_simulations = self.get_dynamic_simulation_count(played_cards, current_round)
        best_action = self.monte_carlo_simulate(my_hand=my_hand, played_cards=played_cards, current_round=current_round, trump_suit=obs.declared_trump, team_started=team_started, num_simulations=num_simulations)
        best_card = best_action
        return best_card

    def monte_carlo_trump(self, my_hand, trump_suit, num_simulations=100, played_cards=None, current_round=None,
                             team_started=None):
        """
        Perform Monte Carlo simulations to estimate the best card to play based on the given hand.
        Args:
            my_hand: List of card IDs representing the player's hand
            num_simulations: Number of simulations to perform
            played_cards: List of cards already played in the game
            current_round: Cards already played in the current trick
            trump_suit: The trump suit for the current game
            team_started = (sum(1 for card in obs.current_trick if card != -1 and obs.current_trick.index(card) in [0, 2]) > 0): Whether the player's team started the round
        Returns:
            A dictionary of estimated scores for each card in my_hand
        """
        if played_cards is None:
            played_cards = []
        if current_round is None:
            current_round = []

        win_counts = {card: 0 for card in my_hand}
        point_totals = {card: 0 for card in my_hand}

        for card in my_hand:
            for _ in range(num_simulations):
                # Simulate the opponent's hand by excluding played cards and current hand
                opponent_hand = self.simulate_opponent_hand(my_hand, played_cards)

                if team_started:
                    # If our team started the round, prioritize cards that maximize points
                    opponent_play = random.choice(opponent_hand)
                else:
                    # If we are responding to a lead, opponents play with a heuristic to maximize their chance of winning
                    lead_card = current_round[0] if current_round.size > 0 else None
                    opponent_play = self.simulate_opponent_play(opponent_hand, lead_card, trump_suit, current_round)

                # Determine if we are playing the first card or responding to a round
                if current_round.size > 0:
                    lead_suit = self.get_suit(current_round[0])
                    if self.get_suit(card) != lead_suit and any(self.get_suit(c) == lead_suit for c in my_hand):
                        continue  # Skip if we can't follow suit

                # Estimate if playing 'card' will win the trick
                if opponent_play is not None and self.card_strength(card, trump_suit) > self.card_strength(opponent_play, trump_suit):
                    win_counts[card] += 1
                    point_totals[card] += self.card_points(card, trump_suit) + self.card_points(opponent_play, trump_suit)

        # Estimate win rate and average points for each card
        estimated_win_rate = {card: wins / num_simulations for card, wins in win_counts.items()}
        estimated_points = {card: points / num_simulations for card, points in point_totals.items()}

        # Combine win rate and point estimate to make a decision
        combined_scores = {card: (estimated_win_rate[card] * 0.6) + (estimated_points[card] * 0.4) for card in my_hand}

        return combined_scores

    def monte_carlo_simulate(self, my_hand, trump_suit, num_simulations=100, played_cards=None, current_round=None, team_started=None):
        """
        Perform Monte Carlo simulations to estimate the best card to play based on the given hand.
        Args:
            my_hand: List of card IDs representing the player's hand
            num_simulations: Number of simulations to perform
            played_cards: List of cards already played in the game
            current_round: Cards already played in the current trick
            trump_suit: The trump suit for the current game
            team_started = (sum(1 for card in obs.current_trick if card != -1 and obs.current_trick.index(card) in [0, 2]) > 0): Whether the player's team started the round
        Returns:
            A dictionary of estimated scores for each card in my_hand
        """
        if played_cards is None:
            played_cards = []
        if current_round is None:
            current_round = []

        # Initialize the root node for MCTS
        root = Node_V4(state=(my_hand, played_cards, current_round))

        # Perform Monte Carlo Tree Search
        max_depth = 3  # Set the desired depth for the tree search
        best_action = self.monte_carlo_tree_search(root, max_depth)
        return best_action


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
            current_round: The current cards played in the trick so far
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
        return random.choice(opponent_hand)

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
        # cuz pythin is an asshole
        # Convert trump_suit to a scalar if it's a NumPy array with a single value
        try:
            temp = trump_suit[0]
            trump_suit = temp
        except:
            pass

        if np.any(np.isin(trump_suit, [0, 1, 2, 3])):
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
