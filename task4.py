import numpy as np
import general as g


def mres(A, f, eps=10**(-4), flag=False, max_iter=1000):
    n = len(f)
    A = np.array(A, dtype=float)
    f = np.array(f, dtype=float)

    if not np.allclose(A, A.T):
        print("Матрица не симметрична!")

    if flag:
        print("Задайте начальное приближение")
        x = g.explicit_vector(n)
        print("Начальное приближение: ")
        g.printA(x)
    else:
        x = np.zeros(n)

    for k in range(max_iter):
        r = f - np.dot(A, x)
        residual_norm = g.spherical_norm(r)

        if residual_norm < eps:
            return x, k + 1

        Ar = np.dot(A, r)
        numerator = np.dot(r, r)
        denominator = np.dot(r, Ar)

        if abs(denominator) < 1e-15:
            print(f"Деление на ноль на итерации {k + 1}")
            break

        alpha = numerator / denominator
        x = x + alpha * r

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

    print("---Метод минимальных невязок---")
    x, count_iter = mres(A, f, eps)
    print("Решение СЛАУ x:")
    g.printA(x)
    print(f"Кол-во итераций: {count_iter}")
    g.compare_vectors(A, x, f)

except ValueError as e:
    print(f"Ошибка: {e}")