"""Softmax."""

import numpy as np
from math import exp

def softmax(x):
    # Computes and returns softmax values for each sets of scores in x.
    # scores that are passed - numpy array
    # 1 row for each score - 3 in this case
    # arbitary number of colums, 1 for each sample !
    # function should be able to iterate over each column
	
	out = []

	try:
		for col in x.T:
			a = []
			sum1 = 0
			for i in range(len(col)):
			    a.append(exp(col[i]))
			    sum1 += exp(col[i])
			for i in range(len(a)):
			    a[i]=a[i]/sum1
			out.append(a)

	except TypeError:
		sum1 = 0
		a = []
		for i in x:
			a.append(exp(i))
			sum1 += exp(i)
		out = np.array(a)/sum1
	return np.array(out)


# example:
scores = np.array([3.0, 1.0, 0.2])
print(softmax(scores))


# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

# print scores

plt.plot(x, softmax(scores), linewidth=2)
plt.show()

"""
Your code failed a test case:
Output shape (80, 3) doesn't match expected shape (3, 80)
"""
