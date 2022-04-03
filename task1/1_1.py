# Задача 1.1. Вычислить методом Монте Карло payoff европейского колл опциона и сравнить с точной формулой Блэка-Шоулза
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from params import *


def MonteCarlo_payoff(s_0, r, sigma, dt, K, steps, paths, draw_plot=True):
    """
    :param r: The expected return
    :param sigma: The expected volatility
    :param dt: The increment of time, fraction of year
    :param K: Strike price,  dollars
    :param draw_plot:
    :return:
    """
    z = np.random.normal(size=(steps, paths))
    s = s_0 * np.exp(np.cumsum((r - np.power(sigma, 2) / 2) * dt + sigma * np.sqrt(dt) * z, axis=0))
    c = np.exp(-r * dt) * np.maximum(s[-1] - K, 0)

    print(c)

    payoff = np.sum(c) / paths

    print(payoff)


    # if draw_plot:
    #     plt.plot(s)
    #     plt.show()

    return payoff


def BlackSholes_payoff(s, r, sigma, dt, K):
    def N(x):
        return norm.cdf(x)

    # dt

    d1 = (np.log(s / K) + (r + np.power(sigma, 2) / 2) * dt) / (sigma * dt)
    d2 = d1 - sigma * dt

    return N(d1) * s - N(d2) * K * np.exp(-r * dt)


MonteCarlo_payoff = MonteCarlo_payoff(s_0, r, sigma, dt, K, steps, paths)

BlackSholes_payoff = BlackSholes_payoff(s_0, r, sigma, dt, K)

print(f'Вычисление payoff\'а европейского колл-опциона с параметрами:\n'
      f'Безрисковая процентная ставка: {r}, волатильность: {sigma}, срок исполнения опциона, месяцев: {dt * 12}, страйк-цена: {K}, начальная цена: {s_0},\n'
      f'количество временных отрезков: {steps}, количество симуляций: {paths}.\n'
      f'Вычисленный по методу Монте-Карло: {MonteCarlo_payoff}, по точной формуле Блэка-Шоулса: {BlackSholes_payoff}\n'
      f'Ошибка метода Монте-Карло составила {abs(MonteCarlo_payoff - BlackSholes_payoff)}')
