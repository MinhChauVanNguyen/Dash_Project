import numpy as np

a1 = [['TP', 'FN'], ['FP', 'TN']]
a2 = [['111', '109'], ['80', '166']]


print(np.char.add(a1[1,1], a2[1,1])