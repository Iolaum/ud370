"""Softmax."""

import numpy as np
from math import exp
import matplotlib.pyplot as plt

def softmax(x):
    # Computes and returns softmax values for each sets of scores in x.
    # scores that are passed - numpy array
    # 1 row for each score - 3 in this case
    # arbitary number of colums, 1 for each sample !
    # function should be able to iterate over each column
	
	return np.exp(x)/np.sum(np.exp(x), axis = 0)

# Learn numpy lol !!! 

# example:
scores = np.array([3.0, 1.0, 0.2])
print(softmax(scores))


# Plot softmax curves

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

# print scores

plt.plot(x, softmax(scores).T, linewidth=2)
plt.ylabel("Softmax")
plt.legend(["x", "1.0", "0.2"])
plt.show()

"""
Your code failed a test case:
Output shape (80, 3) doesn't match expected shape (3, 80)
"""
