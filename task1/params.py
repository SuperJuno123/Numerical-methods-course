import numpy as np


s_0 = 60 # initial price
r = 0.05 # risk-free interest rate
sigma = 0.3 # volatility
# months = 5
# dt = months / 12
dt = 1
K = s_0 * np.exp(r * dt) # strike price
steps = 100
paths = 1000
