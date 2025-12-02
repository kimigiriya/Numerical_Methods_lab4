import numpy as np
import general as g

def slae_gauss(A, f):
    n = len(f)
    Af = np.hstack([A.astype(float), f.reshape(-1, 1).astype(float)])

    for i in range(n):
        max_row = np.argmax(np.abs(Af[i:, i])) + i
        if max_row != i:
            Af[[i, max_row]] = Af[[max_row, i]]

        if np.abs(Af[i, i]) < 1e-10:
            raise ValueError("Матрица вырождена")

        for j in range(i + 1, n):
            factor = Af[j, i] / Af[i, i]
            Af[j, i:] -= factor * Af[i, i:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Af[i, n] - np.dot(Af[i, i + 1:n], x[i + 1:n])) / Af[i, i]

    return x


try:
    n = int(input("Введите n: "))
    A = g.create_matrix(n, 0)
    print("Матрица A:")
    g.printA(A)
    f = g.explicit_vector(n)
    print()
    print("Вектор f:")
    g.printA(f)
    print()
    x = slae_gauss(A, f)
    print("Решение СЛАУ x:")
    g.printA(x)

    g.compare_vectors(A, x, f)

except ValueError as e:
    print(f"Ошибка: {e}")