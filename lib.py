import random
import json
import math
import copy
import os

def set_data_in_file(col, w1, w2):
    with open('data00.json', 'w', encoding="utf-8") as w:
        temp_dict = {"col": col, "w1": w1, "w2": w2}
        json.dump(temp_dict, w, indent=2)

def get_data_from_file(filename):
    with open('data/' + filename, 'r', encoding="utf-8") as file:
        info = json.load(file)
        return info['col'], info['w1'], info['w2']

def init_w_matrix(row, col):
    W = [[random.randint(-1, 1)/100 for j in range(col)] for i in range(row)]
    return W

def init_training_matrix(sequence, row, col):
    result = [[0 for j in range(col)] for i in range(row)]
    for i in range(row):
        for j in range(col):
            result[i][j] = [sequence[i + j]]
    return result

def transp(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def multipl(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    if cols_A != rows_B:
        print("Cannot multiply the two matrices. Incorrect dimensions")
        return
    else:
        C = [[0 for col in range(cols_B)] for row in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    C[i][j] += A[i][k] * B[k][j]
        return C

def get_sequences():
    with open('sequences.json', 'r', encoding="utf-8") as file:
        info = json.load(file)
        return info['sequences']

def sum_error(E):
    result = 0
    for i in range(len(E)):
        result = result + E[i]
    return result

def activation_function(A):
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = math.sin(math.atan(A[i][j]))
            #A[i][j] = math.log10(A[i][j] + (A[i][j]**2+1)**(1/2))
    return A

def delta(m1, m2):
    return [[m1[i][j] - m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]

def alpha_matrix(matrix, alpha):
    return [[alpha * matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))]

def der_act(X):
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] = -(X[i][j]**2)/((X[i][j]+1)**(3/2)) + 1/(X[i][j]**2+1)**(1/2)
            #X[i][j] = 1/((X[i][j]**2+1)**(1/2))
    return X

def hadamard(Y, Y1):
    if Y == Y1 and Y[0] == Y1[0]:
        print("Cannot multiply the two matrices. Incorrect dimensions")
        return
    else:
        return [[Y[i][j] * Y1[i][j] for j in range(len(Y[0]))] for i in range(len(Y))]

def countSecondW(W2, standard, Y, Z, alpha):
    return delta(W2, alpha_matrix(alpha_matrix(transp(Y), Z - standard), alpha))

def hidden_error(W2, gamma):  # супер частный случай для этой нейронки
    return alpha_matrix(W2, gamma)

def countFirstW(W1, W2, standard, Z, X, Y, alpha):
    error = hidden_error(W2, Z - standard)
    devtos = transp(der_act(multipl(transp(X), W1)))
    delW1 = alpha_matrix(multipl(X, transp(hadamard(error, devtos))), alpha)
    return delta(W1, delW1)

def training(p, e, alpha, N, col_training_matrix, train):
    training_sequences = get_sequences()
    row_training_matrix = len(training_sequences[0]) - col_training_matrix - 2
    W1 = init_w_matrix(col_training_matrix + 1, p)
    W2 = init_w_matrix(p, 1)
    #col, W1, W2 = get_data_from_file()
    E = [99999]
    iteration = 0
    while sum_error(E) > e and iteration < N:
        E = []
        training_matrix = init_training_matrix(training_sequences[train], row_training_matrix, col_training_matrix)
        for k in range(len(training_matrix)):
            context = [0]
            tmp_seq = copy.deepcopy(training_matrix[k])
            for pred in range(2):
                X = tmp_seq[-1 * col_training_matrix:]
                X.append(context)
                print("--------")
                #print("contex   " + str(context))
                Y = activation_function(multipl(transp(X), W1))
                Z = multipl(Y, W2)
                buffer = copy.deepcopy(W2)
                W2 = countSecondW(W2, training_sequences[train][col_training_matrix + k + pred], Y, Z[0][0], alpha)
                W1 = countFirstW(W1, buffer, training_sequences[train][col_training_matrix + k + pred], Z[0][0], X, Y,
                                 alpha)
                print('X        ' + str(X))
                #print('matrix   ' + str(training_matrix[k]))
                print("standart " + str(training_sequences[train][col_training_matrix + k + pred]))
                print('result   ' + str(Z[0][0]))
                context = Z[0]
                tmp_seq.append(Z[0])
                E.append((Z[0][0] - training_sequences[train][col_training_matrix + k + pred]) * (
                            Z[0][0] - training_sequences[train][col_training_matrix + k + pred]) / 2)
        iteration = iteration + 1
        print("Step" + str(iteration))
        print('ERRRRRRRROORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR  ' + str(sum_error(E)))
    set_data_in_file(col_training_matrix, W1, W2)
    print("Done")

def prediction(sequence, n, file):
    col, W1, W2 = get_data_from_file(file)
    if len(sequence) < col:
        print("Not enough data")
        return
    X = sequence[-1*col:]
    context = [0]
    for i in range(n):
        X = sequence[-1 * col:]
        X.append(context)
        Y = activation_function(multipl(transp(X), W1))
        Z = multipl(Y, W2)
        sequence.append(Z[0])
        context = Z[0]
        print(Z[0][0])
    print("Done")
    return

