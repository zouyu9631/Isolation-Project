dir = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]

step=[]
step.append([(0, 0)])
w = 7

d = 3

for i in range(w):
    expend = set([(a + x, b + y) for (a, b) in step[i] for (x, y) in dir if -d<a+x<d and -d<b+y<d])
    step.append(expend)

for s in step:
    print(s)


board = []
for i in range(w):
    row = ''
    for j in range(w):
        value = ''
        for t in range(w):
            if (i, j) in step[t]:
                value += str(t)
        value = value.center(w)
        row += '|'+value
    print(row)
    board.append(row)

# print('board: \n', board)

# step1 = set([(a + x, b + y) for (a, b) in step0 for (x, y) in dir])
# step2 = set([(a + x, b + y) for (a, b) in step1 for (x, y) in dir])
# step3 = set([(a + x, b + y) for (a, b) in step2 for (x, y) in dir])
# print(step1)
# print(step2)
# print(step3)