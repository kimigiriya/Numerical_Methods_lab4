import numpy as np
import general as g


def determinant(matrix):
    n = len(matrix)

    if n == 1:
        return matrix[0][0]
    elif n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    elif n == 3:
        return (matrix[0][0] * matrix[1][1] * matrix[2][2] + matrix[1][0] * matrix[2][1] * matrix[0][2] +
                matrix[0][1] * matrix[1][2] * matrix[2][0] - matrix[0][2] * matrix[1][1] * matrix[2][0] -
                matrix[0][0] * matrix[1][2] * matrix[2][1] - matrix[0][1] * matrix[1][0] * matrix[2][2])
    else:
        det = 0
        for j in range(n):
            minor = []
            for i in range(1, n):
                row = []
                for k in range(n):
                    if k != j:
                        row.append(matrix[i][k])
                minor.append(row)

            det += (-1) ** j * matrix[0][j] * determinant(minor)

        return det

def determinant_gauss(matrix):
    n = len(matrix)
    matrix = np.array(matrix, dtype=float)
    det = 1.0
    swaps = 0

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(matrix[k, i]) > abs(matrix[max_row, i]):
                max_row = k

        if abs(matrix[max_row, i]) < 1e-10:
            return 0.0

        if max_row != i:
            matrix[[i, max_row]] = matrix[[max_row, i]]
            swaps += 1
            det *= -1

        det *= matrix[i, i]

        for j in range(i + 1, n):
            factor = matrix[j, i] / matrix[i, i]
            matrix[j, i:] -= factor * matrix[i, i:]

    return det


def inverse_matrix_gauss(matrix):
    matrix = np.array(matrix, dtype=float)
    n = len(matrix)

    if np.abs(determinant_gauss(A)) < 1e-10:
        raise ValueError("Матрица вырождена")

    E = np.eye(n)
    matrix_E = np.hstack([matrix, E])

    for i in range(n):
        max_row = np.argmax(np.abs(matrix_E[i:, i])) + i
        matrix_E[[i, max_row]] = matrix_E[[max_row, i]]

        pivot = matrix_E[i, i]
        matrix_E[i, :] /= pivot

        for j in range(n):
            if j != i:
                factor = matrix_E[j, i]
                matrix_E[j, :] -= factor * matrix_E[i, :]

    return matrix_E[:, n:]

try:
    n = int(input("Введите n: "))
    A = g.create_matrix(n, 0)
    print()
    print("Матрица A:")
    g.printA(A)
    print()
    det = determinant(A)
    print(f"Определитель матрицы A: {det}")
    det_gauss = determinant_gauss(A)
    print(f"Определитель матрицы A методом Гаусса: {det_gauss}")
    print()
    inv_A = inverse_matrix_gauss(A)
    print("Обратная матрица A^(-1):")
    g.printA(inv_A)
    print()
    #print("Проверка: A * A^(-1) = E:")
    #g.printA(np.dot(A, inv_A))

    g.compare_vectors(A, inv_A, np.eye(n))

except ValueError as e:
    print(f"Ошибка: {e}")