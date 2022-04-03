import numpy as np
import math
import matplotlib.pyplot as plt


def ensemble_of_Pascal_matrix(number_of_matrix, dimension_of_matrix):
    def C_n_k(n, k):
        return math.factorial(n) / (math.factorial(n - k) * math.factorial(k))

    C = np.zeros(shape=(number_of_matrix, dimension_of_matrix, dimension_of_matrix))
    C[0] = np.eye(dimension_of_matrix)
    for i in range(1, number_of_matrix):
        C_i = np.eye(dimension_of_matrix)
        for m in range(0, dimension_of_matrix):
            for n in range(m + 1, dimension_of_matrix):
                # у Игоря Николаевича ошибка, вот правильно: file:///C:/Users/Admin/Downloads/lowdisc_toolbox_manual.pdf
                C_i[m, n] = C_n_k(n, m) * np.power(i, n - m)
        C[i] = C_i

    return C


def Faure_sequence(d_dimension, N_num_of_points, b_base=None, draw_plot=True):
    def is_prime(num):
        for n in range(2, int(np.sqrt(num)) + 1):
            if num % n == 0:
                return False
        return True

    if b_base is None:
        b_base = d_dimension
        while not is_prime(b_base):
            b_base += 1

    m_number_of_digits = int(np.ceil(np.log(N_num_of_points) / np.log(b_base)))

    def decomposition(k):
        result = []
        while k >= b_base:
            result.append(k % b_base)
            k //= b_base
        result.append(k)
        if len(result) < m_number_of_digits:
            result += [0] * (m_number_of_digits - len(result))
        return np.array(result)

    def Fi(n):
        """https://downloads.hindawi.com/journals/amp/2021/6696895.pdf"""
        b = np.array(np.cumprod(np.ones(m_number_of_digits) * b_base))
        # print(a, b, a / b)
        return np.sum(n / b)

    C = ensemble_of_Pascal_matrix(d_dimension, m_number_of_digits)
    x = np.zeros((N_num_of_points, d_dimension))

    for n in range(N_num_of_points):
        for j in range(d_dimension):
            x[n, j] = Fi(C[j] @ decomposition(n) % b_base)

    return x

dimension = 3
N = 856
base = 3
# Результат для N = 8, dim = base = 3
# совпадает с https://www.researchgate.net/publication/351037438_Low_Discrepancy_Toolbox_Manual
faure = Faure_sequence(d_dimension=dimension, N_num_of_points=N, b_base=base)

print(faure)

plt.scatter(faure[:, 0], faure[:, 1], marker='.')
plt.xlim([0, 1])
plt.ylim([0, 1])
fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.title(f'Faure Sequence (base={base}, N={N})')
plt.savefig(f'Faure Sequence (base={base}, N={N}), first two dimensions.png')
plt.show()

if np.shape(faure)[1] > 2:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(faure[:, 0], faure[:, 1], faure[:, 2])
    ax.set_zlim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Faure Sequence (base={base}, N={N}), first three dimensions)')
    plt.savefig(f'Faure Sequence (base={base}, N={N}), first three dimensions.png')
    plt.show()

digits = 6
base_for_proof = 2
N_for_proof = 2 ** digits
example_for_proof = Faure_sequence(d_dimension=2, N_num_of_points=N_for_proof, b_base=base_for_proof)
possible_a_b = [(1 / (base_for_proof ** i), 1 / base_for_proof ** (digits - i)) for i in range(digits)]

n_plots = int(np.ceil(digits / 2))
f, axes = plt.subplots(nrows=n_plots, ncols=n_plots, sharex=True, sharey=True)
for i in range(n_plots):
    for j in range(n_plots):
        if i + j > digits:
            break
        (a, b) = possible_a_b[i + j]
        tmp_a = a
        while not a >= 1:
            axes[i, j].axvline(a)
            a += tmp_a
        tmp_b=b
        while not b >= 1:
            axes[i, j].axhline(b)
            b += tmp_b
        axes[i, j].scatter(example_for_proof[:, 0], example_for_proof[:, 1], s=64 // digits,  marker='.', color='r', zorder=66)
        axes[i, j].set_xlim([0, 1])
        axes[i, j].set_ylim([0, 1])

plt.suptitle(f'Faure sequence for base={base_for_proof} and {N_for_proof} points.\nElementary intervals of area 1/{str(N_for_proof)}')
plt.show()



