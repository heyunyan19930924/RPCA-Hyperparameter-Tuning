'''
Script for testing data generating method.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

dimension = 1000
list1 = np.random.randn(dimension)
list2 = np.random.randn(dimension)
plt.hist(list1, bins=int(dimension/10))
plt.hist(list2, bins=int(dimension/10))
product = list1*list2
plt.hist(product, bins=int(dimension/10))

mu, std = norm.fit(product)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p*np.sqrt(dimension), 'k', linewidth=2)
plt.show()