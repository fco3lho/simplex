import numpy as np

def read_input(file_path):
  with open(file_path, 'r') as file:
    # Lê número de variáveis (n) e número de restrições (m)
    n, m = map(int, file.readline().strip().split())

    # Lê os coeficientes da função objetivo
    c = np.array(list(map(float, file.readline().strip().split())))

    # Lẽ as restrições, onde Ai = bj

    A = []
    b = []

    for _ in range(m):
        line = list(map(float, file.readline().strip().split()))
        A.append(line[:-1]) # Primeiro element ao antipenúltimo
        b.append(line[-1]) # Último elemente

    A = np.array(A)
    b = np.array(b)
    
  return n, m, c, A, b