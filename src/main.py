from read_input import read_input
from simplex import simplex
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    
    file_path = sys.argv[1]
    n, m, c, A, b = read_input(file_path)
    simplex(c, A, b)
