import numpy as np


def printA(arr):
    for row in np.array(arr):
        if row.ndim == 0:
            print("{:f}".format(row), end=" ")
        else:  # Матрица
            for element in row:
                print("{:f}".format(element), end=" ")
            print()
    if np.array(arr).ndim == 1:
        print()

def explicit_vector(n):
    print(f"Введите элементы вектора f размерностью {n}:")
    elements = list(map(float, input("f: ").split()))
    if len(elements) != n:
        print(f"Ошибка: нужно ввести ровно {n} чисел!")
        return explicit_vector(n)
    return np.array(elements)

def explicit_matrix(n):
    print(f"Введите элементы матрицы {n}x{n} построчно:")
    matrix = []
    for i in range(n):
        row = list(map(float, input(f"Строка {i + 1}: ").split()))
        if len(row) != n:
            print(f"Ошибка: нужно ввести ровно {n} чисел!")
            return explicit_matrix(n)
        matrix.append(row)
    return np.array(matrix)

def rand_matrix(n):
    l = float(input("Введите нижний предел генерации: "))
    r = float(input("Введите верхний предел генерации: "))
    return np.random.uniform(l, r, (n, n))



def degenerate_matrix(n):
    print("Выберите тип вырожденности:")
    print("1 - Две одинаковые строки")
    print("2 - Нулевая строка")

    choice = int(input("Выбор: "))
    matrix = np.random.rand(n, n)

    if choice == 1:
        if n < 2:
            raise ValueError("Матрица слишком мала для двух одинаковых строк!")

        print(f"Доступные строки от 1 до {n}")
        i = int(input("Введите номер первой строки: ")) - 1
        j = int(input("Введите номер второй строки: ")) - 1

        if i == j:
            raise ValueError("Строки должны быть разными!")

        matrix[j] = matrix[i]
        print(f"Строка {j + 1} сделана равной строке {i + 1}")

    elif choice == 2:
        zero_row = int(input(f"Введите номер нулевой строки от 1 до {n}): ")) - 1
        matrix[zero_row] = 0
        print(f"Строка {zero_row + 1} обнулена")

    else:
        raise ValueError("Неизвестный тип вырожденной матрицы")

    return matrix

def hilbert_matrix(n):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = 1 / (i + j + 1)
    return matrix

def create_matrix(n, flag=0):
    if flag:
        print("1 - Ручной ввод матрицы")
        print("2 - Случайная генерация матрицы")
        print("3 - Единичная матрица")
        print("4 - Нулевая матрица")
        print("5 - Вырожденная матрица")
        print("6 - Матрица Гильберта")

    m_type = int(input("Введите тип матрицы: "))

    matrix_creators = {
        1: lambda n: explicit_matrix(n),
        2: lambda n: rand_matrix(n),
        3: lambda n: np.eye(n),
        4: lambda n: np.zeros((n, n)),
        5: lambda n: degenerate_matrix(n),
        6: lambda n: hilbert_matrix(n)
    }
    if m_type in matrix_creators:
        return matrix_creators[m_type](n)
    else:
        raise ValueError("Неизвестный тип матрицы")

def cubic_norm(x):
    return np.max(np.abs(x))

def octahedral_norm(x):
    return np.sum(np.abs(x))

def spherical_norm(x):
    return np.sqrt(np.sum(x**2))

def compare_vectors(A, x, f):
    f1 = np.dot(A, x)
    delta = f - f1

    print(f"Вектор f: ")
    printA(f)
    print(f"Восстоновленный вектор f: ")
    printA(f1)

    print(f"Кубическая норма: {cubic_norm(delta)}")
    print(f"Октаэдральная норма: {octahedral_norm(delta)}")
    print(f"Сферическая норма: {spherical_norm(delta)}")
