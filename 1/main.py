import sys

try:
    # get input fromthe arguments if provided, 
    # otherwise get it from the termial
    match len(sys.argv):
        case 1:
            N = int(input("Enter N: "))
        case 2:
            N = int(sys.argv[1])
        case _:
            print('Too many arguments')
            sys.exit(0)
except ValueError:
    print('Input is not an integer')
    sys.exit(0)

# check if it's positive
if N <= 0:
    print('Input is not a positive integer')
    sys.exit(0)

# calculate the sum
result = 0
for i in range(1, N + 1):
    result += i

# here's a functional alternative:
# result = sum(range(1, N + 1))

print(f'Result: {result}')
        