from isolation import Board
from sample_players import RandomPlayer
from sample_players import HumanPlayer
from sample_players import null_score
from sample_players import open_move_score
from sample_players import improved_score
from game_agent import CustomPlayer
from game_agent import custom_score
from game_agent import all_boxes_can_move_score
import timeit

TIME_LIMIT = 150  # number of milliseconds before timeout

# |   | - |   | - | - |   |   |
# |   | - |   | - |   | - |   |
# |   |   | - | - | - | - |   |
# | - | - | - | - | 1 | - |   |
# | 2 |   |   | - | - | - |   |
# |   |   | - |   | - |   | - |
# |   |   | - | - | - |   |   |
# [[(6, 3), (6, 2)], [(4, 4), (4, 3)], [(2, 5), (3, 1)], [(0, 4), (5, 2)], [(2, 3), (6, 4)], [(1, 5), (5, 6)], [(0, 3), (3, 5)], [(1, 1), (5, 4)], [(3, 0), (3, 3)], [(2, 2), (4, 5)], [(0, 1), (2, 4)], [(1, 3), (3, 2)], [(3, 4), (4, 0)], [(-1, -1)]]

#  |   |   |   | - | - |   |   |
#  |   | - | - |   | - | - |   |
#  | - |   | - | - | - |   | - |
#  | - | - |   | - | - |   | - |
#  |   | - |   | 2 | - | - | 1 |
#  |   |   | - | - |   | - | - |
#  |   |   |   | - | - | - |   |
#
# [[(3, 0), (2, 2)], [(1, 1), (0, 3)], [(2, 3), (2, 4)], [(0, 4), (3, 6)], [(1, 2), (1, 5)], [(2, 0), (3, 4)], [(4, 1), (5, 5)], [(3, 3), (6, 3)], [(1, 4), (4, 4)], [(2, 6), (5, 6)], [(4, 5), (6, 4)], [(5, 3), (5, 2)], [(6, 5), (3, 1)], [(4, 6), (4, 3)], [(-1, -1)]]

#  |   | - | - |   | - |   |   |
#  | - |   | - | - | 1 |   |   |
#  |   |   | - | - | - |   |   |
#  |   | - | - | - |   | - |   |
#  |   | - | - | - |   | - |   |
#  | - |   | - |   | - |   |   |
#  |   |   | - |   | 2 |   | - |
#
# [[(0, 4), (4, 5)], [(2, 3), (6, 6)], [(0, 2), (5, 4)], [(1, 0), (4, 2)], [(2, 2), (5, 0)], [(0, 1), (6, 2)], [(1, 3), (4, 1)], [(3, 2), (3, 3)], [(2, 4), (1, 2)], [(4, 3), (3, 1)], [(3, 5), (5, 2)], [(1, 4), (6, 4)], [(-1, -1)]]

DIR = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]

CUSTOM_ARGS = {"method": 'alphabeta', 'iterative': True}

curr_time_millis = lambda: 1000 * timeit.default_timer()

p2 = CustomPlayer(score_fn=custom_score, **CUSTOM_ARGS)
p1 = CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS)
#p2 = HumanPlayer()

w = 7
g = Board(p1, p2, w, w)

moves = [[(5, 1), (2, 6)], [(3, 0), (0, 5)], [(1, 1), (1, 3)], [(0, 3), (2, 1)], [(1, 5), (3, 3)], [(2, 3), (5, 2)],
         [(3, 1), (4, 0)], [(1, 0), (3, 2)], [(0, 2), (2, 4)], [(1, 4), (4, 3)], [(2, 2), (3, 5)], [(0, 1), (5, 4)],
         [(2, 0), (4, 6)], [(4, 1), (2, 5)]]
for (move1, move2) in moves:
    g.apply_move(move1)
    if move2: g.apply_move(move2)

print(g.to_string())

res = g.play(TIME_LIMIT)
print(res[0].score, res[1])
print(g.to_string())


def try_moves(moves):
    g = Board(p1, p2, w, w)
    for move in moves:
        g.apply_move(move)
    move_start = curr_time_millis()
    time_left = lambda: TIME_LIMIT - (curr_time_millis() - move_start)
    print(g.to_string())
    active_p = g.active_player
    p_l_m = g.get_legal_moves(active_p)
    p_res = active_p.get_move(g, p_l_m, time_left)


    # p1_pos = [(i, j) for i in range(w//2+1) for j in range(i, w//2+1)]
# p1_pos = [(i, i) for i in range(w//2)]
# for p1_move in p1_pos:
#     for p2_move in [(i, j) for i in range(w) for j in range(i, w) if (i, j) != p1_move]:
#         try_moves([p1_move, p2_move])
#
#
# p1_pos = [(i, w//2) for i in range(w//2)]
# for p1_move in p1_pos:
#     for p2_move in [(i, j) for i in range(w) for j in range(w//2+1) if (i, j) != p1_move]:
#         try_moves([p1_move, p2_move])
#
# p1_pos = [(w//2, w//2)]
# for p1_move in p1_pos:
#     for p2_move in [(i, j) for i in range(w//2) for j in range(i, w//2+1) if (i, j) != p1_move]:
#         try_moves([p1_move, p2_move])
#
# p1_pos = [(i, j) for i in range(w//2) for j in range(i, w//2) if i != j]
# for p1_move in p1_pos:
#     for p2_move in [(i, j) for i in range(w) for j in range(w) if (i, j) != p1_move]:
#         try_moves([p1_move, p2_move])


# for i in range(5):
#     for j in range(5):
#         if (i, j) == (0, 1): continue
#         g2 = g.copy()
#         g2.apply_move((i, j))
#         print(g2.to_string())
#         p1_l_m = g2.get_legal_moves(p1)
#         p1_res = p1.get_move(g2, p1_l_m, time_left)
#         print(p1_res)


# p1_l_m = g.get_legal_moves(p1)
# p1_res = p1.get_move(g, p1_l_m, time_left)
# print(p1_res)

# p2_l_m = g.get_legal_moves(p2)
# p2_res = p2.get_move(g, p2_l_m, time_left)
# print(p2_res)


# p1_move = (0, 2)
#p2_moves = [(a, b) for a in range(5) for b in range(3) if (a, b) != p1_move]
#p2_moves = [(a, b) for a in range(3) for b in range(3) if (a, b) != p1_move]
# p2_moves = [(1, 3)]

# p1_wins = []
# p2_wins = []

# for p2_move in p2_moves:
#     g = Board(p1, p2, 5, 5)
#     g.apply_move(p1_move)
#     g.apply_move(p2_move)
#
#     res = g.play(time_limit=TIME_LIMIT)
#     print(res)

    # move_start = curr_time_millis()
    # time_left = lambda : TIME_LIMIT - (curr_time_millis() - move_start)
    # p1_l_m = g.get_legal_moves(p1)
    # p1_res = p1.get_move(g, p1_l_m, time_left)
    # if p1_res == (-1, -1):
    #     p2_wins.append(p2_move)
    #     print('in game: p1 lose!')
    # else:
    #     g.apply_move(p1_res)
    #     move_start = curr_time_millis()
    #     time_left = lambda: TIME_LIMIT - (curr_time_millis() - move_start)
    #     p2_l_m = g.get_legal_moves(p2)
    #     p2_res = p2.get_move(g, p2_l_m, time_left)
    #     if p2_res == (-1, -1):
    #         p1_wins.append(p1_move)
    #         print('in game: p1 win!')
    #     else:
    #         print('terrible! did not search to the end!!!!!! in game: ')
    # print(g.to_string())
#
# print(p1_wins)
# print(p2_wins)