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

	for col in x.T:
	    a = []
	    sum1 = 0
	    for i in col:
	        i = exp(i)
	        a.append(i)
	        sum1 += i
	    for i in range(len(a)):
	        a[i]=a[i]/sum1
	    out.append(a)
	return np.array(out)



# example:
scores = np.array([[3.0, 2.0], [1.0, 1.0], [0.2, 0.2]])
print(softmax(scores))


# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

print(scores)

plt.plot(x, softmax(scores), linewidth=2)
plt.show()

""" 
Runs correctly on Test run but gives error on submission !!

Traceback (most recent call last):
  File "vm_main.py", line 33, in <module>
    import main
  File "/tmp/vmuser_nktifkgnvj/main.py", line 2, in <module>
    import aiMain
  File "/tmp/vmuser_nktifkgnvj/aiMain.py", line 100, in <module>
    run_tests()
  File "/tmp/vmuser_nktifkgnvj/aiMain.py", line 84, in run_tests
    outputs = np.float_(softmax(test_case['inputs']))
  File "/tmp/vmuser_nktifkgnvj/main_code.py", line 17, in softmax
    for i in col:
TypeError: 'numpy.float64' object is not iterable
"""
