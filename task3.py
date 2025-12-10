import numpy as np
import general as g


def slae_jacobi(A, f, eps=10**(-4), flag=False, max_iter=1000):
    n = len(f)
    A = np.array(A, dtype=float)
    f = np.array(f, dtype=float)

    for i in range(n):
        if (np.sum(np.abs(A[i])) - np.abs(A[i][i])) > np.abs(A[i][i]):
            print("Матрица без диагонального преобладания!")
            break

    if flag:
        print("Задайте начальное приближение")
        x = g.explicit_vector(n)
        print("Начальное приближение: ")
        g.printA(x)
    else:
        x = np.zeros(n)

    D = np.diag(A)
    R = A - np.diag(D)

    for k in range(max_iter):
        x_new = (f - np.dot(R, x)) / D

        if g.spherical_norm(x_new - x) < eps:
            return x_new, k + 1

        x = x_new.copy()

    print(f"Было достигнуто максимальное кол-во итераций = {max_iter}")
    return x, max_iter

def slae_seidel(A, f, eps=10**(-4), flag=False, max_iter=1000):
    n = len(f)
    A = np.array(A, dtype=float)
    f = np.array(f, dtype=float)

    for i in range(n):
        if (np.sum(np.abs(A[i])) - np.abs(A[i][i])) > np.abs(A[i][i]):
            print("Матрица без диагонального преобладания!")
            break

    if flag:
        print("Задайте начальное приближение")
        x = g.explicit_vector(n)
        print("Начальное приближение: ")
        g.printA(x)
    else:
        x = np.zeros(n)

    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (f[i] - s1 - s2) / A[i][i]

        if g.spherical_norm(x_new - x) < eps:
            return x_new, k + 1
        x = x_new.copy()

    print(f"Было достигнуто максимальное кол-во итераций = {max_iter}")
    return x, max_iter


try:
    n = int(input("Введите n: "))
    A = g.create_matrix(n, 0)
    print("Матрица A:")
    g.printA(A)
    print()
    f = g.explicit_vector(n)
    print("Вектор f:")
    g.printA(f)
    print()
    eps = float(eval(input("Введите ε: ")))

    print("---Метод Якоби---")
    x_j, count_iter_j = slae_jacobi(A, f, eps)
    print("Решение СЛАУ x:")
    g.printA(x_j)
    print(f"Кол-во итераций: {count_iter_j}")
    g.compare_vectors(A, x_j, f)

    print()
    print("---Метод Зейделя---")
    x_s, count_iter_s = slae_seidel(A, f, eps)
    print("Решение СЛАУ x:")
    g.printA(x_s)
    print(f"Кол-во итераций: {count_iter_s}")
    g.compare_vectors(A, x_s, f)

except ValueError as e:
    print(f"Ошибка: {e}")