dir = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]

w = 5
start = (2, 2)
goal = (2, 3)

blocked = []

def successors(pos):
    return {(pos[0] + x, pos[1] + y): str((x, y)) for (x, y) in dir if
            0 <= pos[0] + x < w and 0 <= pos[1] + y < w #}
            and (pos[0] + x, pos[1] + y) not in blocked}


from collections import Counter


def find_all_path():
    explored = set()
    frontier = [[start]]
    all_ways = []
    cnt = Counter()
    while frontier:
        path = frontier.pop(0)
        (x, y) = path[-1]
        for state, action in successors((x, y)).items():
            if state not in path:
                # explored.add(state)
                path2 = path + [action, state]
                if goal == state:
                    p = path2[0::2]
                    all_ways.append(p)
                    cnt[(p[0], p[1])] += 1
                    # print('find a path:', len(p), 'len :', p)
                frontier.append(path2)
    print(len(all_ways))
    print(cnt.most_common())


find_all_path()
# print(successors((2, 2)).items())
