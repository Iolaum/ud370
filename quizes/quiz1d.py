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
scores = np.array([3.0, 80])
print(softmax(scores))



