import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from params import *


def MonteCarlo_asian_payoff(s_0, r, sigma, T, K, m_steps, n_simulations):
    """
    :param s_0: current price
    :param r: The expected return (risk-free interest rate)
    :param sigma: The expected volatility
    :param T: Expiration, fraction of year
    :param K: Strike price,  dollars
    :param n_simulations: Number of simulations
    :param m_steps: Number of time intervals (steps)
    :return:
    """
    t = np.linspace(0, T, num=m_steps)
    z = np.random.normal(size=(n_simulations, m_steps))
    s = np.zeros((n_simulations, m_steps))
    c = np.zeros(n_simulations)
    s[:, 0] = s_0
    for i in range(n_simulations):
        for j in range(m_steps - 1):
            s[i, j + 1] = s[i, j] * np.exp(
                (r - np.power(sigma, 2) / 2) * (t[j + 1] - t[j]) + sigma * np.sqrt(t[j + 1] - t[j]) * z[i, j])
            c[i] = np.exp(-r * dt) * np.sum(np.maximum(s[i, j] - K, 0)) / m_steps

    return np.average(c)


MonteCarlo_payoff = MonteCarlo_asian_payoff(s_0, r, sigma, dt, K, steps, paths)

print(f'Вычисление payoff\'а азиатского колл-опциона с параметрами:\n'
      f'Безрисковая процентная ставка: {r}, волатильность: {sigma}, срок исполнения опциона, месяцев: {dt * 12}, страйк-цена: {K},\n'
      f'начальная цена: {s_0}, количество временных отрезков: {steps}, количество симуляций: {paths}.\n'
      f'Вычисленный payoff по методу Монте-Карло: {MonteCarlo_payoff}, с временными интервалами {dt / steps} лет')
