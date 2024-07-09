import numpy as np
import time

def simplex(c, A, b):
  ######################## 1 Inicialização e preparação
  start_time = time.time() # Armazena o tempo inicial para calcular o tempo total de execução
  num_vars = len(c) # Número de variáveis de decisão
  num_constraints = len(b) # Número de restrições

  A = np.hstack([A, np.eye(num_constraints)]) # Adiciona uma matriz identidade para representar as variáveis de folga
  c = np.hstack([c, np.zeros(num_constraints)]) # Adiciona coeficientes zero para as variáveis de folga na função objetivo
  
  basic_vars = list(range(num_vars, num_vars + num_constraints)) # Lista das variáveis básicas iniciais
  
  ######################## 2 Loop principal
  iter_count = 0
  
  while True:
    iter_count += 1
    
    ######################## 3 Cálculo de custos relativos
    cb = c[basic_vars] # Coeficientes das variáveis básicas na função objetivo
    z = cb @ A # Produto escalar para calcular os custos relativos
    relative_costs = c - z # Custos relativos que indicam quanto a função objetivo melhoraria se aumentássemos cada variável não-básica
    
    ######################## 4 Determinação de variável de entrada
    if all(rc >= 0 for rc in relative_costs): # Verifica se todos os custos relativos são não-negativos
      break # Sai do loop se a solução atual é ótima
 
    entering = np.argmin(relative_costs) # Índice da variável com o custo relativo mais negativo
    
    ######################## 5 Determinação de variável de saída
    ratios = b / A[:, entering] # Calcula os valores de theta
    valid_ratios = [(i, ratio) for i, ratio in enumerate(ratios) if ratio > 0] # Filtra apenas os valores positivos de theta
    
    if not valid_ratios: #  Verifica se não há valores válidos de theta
      print("Solução ilimitada.")
      return
    
    leaving = min(valid_ratios, key=lambda x: x[1])[0] # Índice da variável que sai da base
    
    ######################## 6 Operação de Pivot
    pivot_element = A[leaving, entering] # Obtém o elemento pivot
    A[leaving] /= pivot_element # Normaliza a linha na matriz A
    b[leaving] /= pivot_element # Normaliza o termo independente correspondente
    
    for i in range(num_constraints): # Itera sobre todas as linhas
      if i != leaving: # Ignora a linha da variável de saída
        factor = A[i, entering] # Fator de multiplicação para zerar o elemento
        A[i] -= factor * A[leaving] # Atualiza a linha na matriz A
        b[i] -= factor * b[leaving] # Atualiza o termo independente correspondente
    
    ######################## 7 Atualização das variáveis básicas
    basic_vars[leaving] = entering # Atualiza a lista de variáveis básicas substituindo a variável que saiu pela variável que entrou
    
    ######################## 8 Verificação de otimalidade e impressão de resultados
    print(f"Iteração: {iter_count}")
    print(f"Tempo(s): {time.time() - start_time:.4f}")
    print(f"Objetivo: {(-cb @ b):.4f}")
    print()
  
  ######################## 9 Impressão de solução ótima
  solution = np.zeros(num_vars + num_constraints) # Cria um vetor de solução inicializado com zeros
  solution[basic_vars] = b # Preenche a solução com os valores das variáveis básicas

  print(f"Solução ótima encontrada em {time.time() - start_time:.4f} segundos!")
  print(f"Função objetivo é {-cb @ b:.4f}.")
  print("Solução:")

  for i in range(num_vars):
    print(f"x[{i+1}] = {solution[i]:.4f}")
