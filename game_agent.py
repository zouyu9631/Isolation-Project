"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import sys

INF = float("inf")


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return -INF
    if game.is_winner(player):
        return INF

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 2 * opp_moves)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate successors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=7, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def fixed_depth_search(self, game, depth_limit=None):
        if depth_limit is None or not isinstance(depth_limit, int):
            depth_limit = self.search_depth

        if self.method == 'minimax':
            search_fn = self.minimax
        elif self.method == 'alphabeta':
            search_fn = self.alphabeta
        else:
            raise ValueError("search method unknown: " + self.method)

        return search_fn(game, depth_limit)

    def iterative_deepening_search(self, game):
        result = -INF, (-1, -1)
        try:
            for depth in range(1, sys.maxsize):
                result = self.fixed_depth_search(game, depth)
        except Timeout:
            pass
        finally:
            return result

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        # TODO: very first step, but this strategy need to be reconsidered
        if not legal_moves:
            return (-1, -1)
        if len(legal_moves) > 8:
            middle_point = (game.height // 2, game.width // 2)
            if (middle_point in legal_moves):
                return middle_point
            elif (middle_point[0] - 1, middle_point[1] - 1) in legal_moves:
                return (middle_point[0] - 1, middle_point[1] - 1)
            else:
                print(game.to_string())
                print(legal_moves)
                raise ValueError("check the board!")

        res_move = None
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                score, res_move = self.iterative_deepening_search(game)
            else:
                score, res_move = self.fixed_depth_search(game)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            if res_move is None:
                res_move = random.choice(legal_moves)
                print('in getmove, timeout and no move,return a random move: ', res_move)
            return res_move

        # Return the best move from the last completed search iteration
        return res_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # terminal states check
        if depth == 0:
            return self.score(game, self), (-1, -1)
        u = game.utility(self)
        if u != 0.0:
            return u, (-1, -1)

        legal_moves = game.get_legal_moves()

        # initialize function depends on maximizing_player
        if maximizing_player:
            min_or_max = max
            res = -INF, (-1, -1)
        else:
            min_or_max = min
            res = INF, (-1, -1)

        # minimax body
        for move in legal_moves:
            res = min_or_max(res,
                             (self.minimax(game.forecast_move(move), depth - 1, not maximizing_player)[0], move))
        return res

    def alphabeta(self, game, depth, alpha=-INF, beta=INF, maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # terminal states check
        if depth == 0:
            return self.score(game, self), (-1, -1)
        u = game.utility(self)
        if u != 0.0:
            return u, (-1, -1)

        legal_moves = game.get_legal_moves()

        # initialize functions depends on maximizing_player
        if maximizing_player:
            min_or_max = max
            res = -INF, (-1, -1)
            can_prunning = lambda v, alpha, beta: v >= beta
            update_alphabeta = lambda v, alpha, beta: (max(alpha, v), beta)
        else:
            min_or_max = min
            res = INF, (-1, -1)
            can_prunning = lambda v, alpha, beta: v <= alpha
            update_alphabeta = lambda v, alpha, beta: (alpha, min(beta, v))

        # alphabeta body
        for move in legal_moves:
            t_res = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta,
                                   not maximizing_player)
            res = min_or_max(res,((t_res)[0], move))
            if can_prunning(t_res[0], alpha, beta):
                return t_res
            alpha, beta = update_alphabeta(res[0], alpha, beta)
        return res
