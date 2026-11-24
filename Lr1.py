import numpy as np

A = np.array([
    [10.35, -1.24,  2.18,  0.75],
    [ 1.12, 11.48, -1.35,  2.06],
    [ 2.31, -1.17, 12.63, -1.42],
    [ 0.84,  2.25, -1.16,  9.74],
])
b = np.array([14.27, 18.64, 20.15, 12.83])


def gauss(A, b):
    M = A.astype(float)
    c = b.astype(float)
    n = len(b)
    for k in range(n):
        pivot_row = k
        for row in range(k + 1, n):
            if abs(M[row, k]) > abs(M[pivot_row, k]):
                pivot_row = row

        if M[pivot_row, k] == 0:
            raise ValueError("У системы нет единственного решения")

        if pivot_row != k:
            saved_row = M[k].copy()
            M[k] = M[pivot_row]
            M[pivot_row] = saved_row
            c[k], c[pivot_row] = c[pivot_row], c[k]

        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            for j in range(k, n):
                M[i, j] -= factor * M[k, j]
            c[i] -= factor * c[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        known_sum = 0.0
        for j in range(i + 1, n):
            known_sum += M[i, j] * x[j]
        x[i] = (c[i] - known_sum) / M[i, i]
    return x


def jacobi(A, b, x0, eps, kmax=1000, show_steps=False):
    n = len(b)
    x = np.array(x0, dtype=float)
    for k in range(1, kmax + 1):
        old = x.copy()
        for i in range(n):
            total = 0.0
            for j in range(n):
                if j != i:
                    total += A[i, j] * old[j]
            x[i] = (b[i] - total) / A[i, i]

        step = np.linalg.norm(x - old)
        if show_steps:
            print(f"Шаг {k}: x = {x}, изменение = {step:.3e}")
        if step < eps:
            return x, k

    print(f"Якоби: критерий остановки не выполнен за {kmax} шагов")
    return x, kmax


def seidel(A, b, x0, eps, kmax=1000, show_steps=False):
    n = len(b)
    x = np.array(x0, dtype=float)
    for k in range(1, kmax + 1):
        old = x.copy()
        for i in range(n):
            total = 0.0
            for j in range(n):
                if j != i:
                    total += A[i, j] * x[j]
            x[i] = (b[i] - total) / A[i, i]

        step = np.linalg.norm(x - old)
        if show_steps:
            print(f"Шаг {k}: x = {x}, изменение = {step:.3e}")
        if step < eps:
            return x, k

    print(f"Зейдель: критерий остановки не выполнен за {kmax} шагов")
    return x, kmax


def residual(A, x, b):
    return A @ x - b


n = len(b)
print("Определитель:", np.linalg.det(A))
print("Число обусловленности:", np.linalg.cond(A))
for i in range(n):
    off_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
    print(f"Строка {i + 1}: {abs(A[i, i]):.2f} > {off_sum:.2f}: {abs(A[i, i]) > off_sum}")


x0 = np.zeros(n)
eps = 1e-6
x_numpy = np.linalg.solve(A, b)
x_gauss = gauss(A, b)
print("\nПростая итерация (Якоби)")
x_jacobi, steps_jacobi = jacobi(A, b, x0, eps, show_steps=True)
print("\nМетод Зейделя")
x_seidel, steps_seidel = seidel(A, b, x0, eps, show_steps=True)


print("\nСравнение при eps =", eps)
results = [("NumPy", x_numpy), ("Гаусс", x_gauss),
           ("Якоби", x_jacobi), ("Зейдель", x_seidel)]
for name, x in results:
    r = residual(A, x, b)
    print(name, "x =", x, "невязка =", r)
    print(f"Норма невязки = {np.linalg.norm(r):.3e}, отличие от NumPy = {np.linalg.norm(x - x_numpy):.3e}")
print("Итераций Якоби:", steps_jacobi, "Итераций Зейделя:", steps_seidel)


for name, method in [("Якоби", jacobi), ("Зейдель", seidel)]:
    print("\nВлияние начального приближения:", name)
    for start in [x0, np.ones(n), np.array([10., -10., 5., -5.])]:
        x, steps = method(A, b, start, eps)
        print(f"Старт {start}, x = {x}, шагов {steps}, норма невязки {np.linalg.norm(residual(A, x, b)):.3e}")

    print("\nВлияние точности:", name)
    for tolerance in [1e-2, 1e-4, 1e-6]:
        x, steps = method(A, b, x0, tolerance)
        print(f"eps = {tolerance:g}, шагов {steps}, норма невязки {np.linalg.norm(residual(A, x, b)):.3e}")


print("\nВлияние округления")
for digits in [2, 1, 0]:
    rounded_A = np.round(A, digits)
    rounded_b = np.round(b, digits)
    x = gauss(rounded_A, rounded_b)
    change = np.linalg.norm(x - x_numpy)
    print(f"Знаков {digits}, x = {x}, относительное изменение {change / np.linalg.norm(x_numpy):.2%}")
