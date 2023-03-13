import sys
import queue

# get the input string
str = input('Enter the dimensions of your matrix separated by a space: ')
tokens = str.split(' ')

# validate the inputs
if len(tokens) < 2:
    print('Not enough numbers')
    sys.exit(0)

if len(tokens) > 2:
    print('Too many numbers')
    sys.exit(0)

try:
    M = int(tokens[0])
    N = int(tokens[1])
except ValueError:
    print('Inputs are not valid integers')
    sys.exit(0)

if M <= 0 or N <= 0:
    print('Inputs are not positive integers')
    sys.exit(0)

matrix = []

# get and validate the matrix
print('Enter the matrix:')
for i in range(M):
    str = input()
    tokens = str.split(' ')

    if len(tokens) != N:
        print('Invalid matrix')
        sys.exit(0)

    matrix.append([])
    
    for j in range(N):
        try:
            num = int(tokens[j])
        except ValueError:
            print('An element is not a number')
            sys.exit(0)
        
        if num != 0 and num != 1:
            print('Only 0 and 1 are allowed as elements of the matrix')
            sys.exit(0)

        matrix[i].append(num)

# create a matrix of boolean flags that indicate 
# if an element has been visited by the search algorithm
visited = []
for i in range(M):
    visited.append([])
    for j in range(N):
        visited[i].append(False)

# this function finds all the pieces of land that 
# belong to the current island and marks them as visited
def search_the_island(matrix, visited, i, j):
    M = len(matrix)
    N = len(matrix[0])
    q = []

    q.append((i, j))

    while len(q) > 0:
        i, j = q.pop()

        if not (0 <= i < M) or not (0 <= j < N) or visited[i][j] or matrix[i][j] == 0:
            continue

        visited[i][j] = True
        
        for di, dj in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            q.append((i + di, j + dj))

# the island counter
count = 0

# solve the problem
for i in range(M):
    for j in range(N):
        if visited[i][j]:
            continue

        if matrix[i][j] == 0:
            visited[i][j] = True
            continue

        count += 1
        search_the_island(matrix, visited, i, j)

print(f'Result: {count}')
