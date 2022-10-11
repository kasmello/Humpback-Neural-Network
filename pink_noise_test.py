import numpy as np
import matplotlib.pyplot as plt

test = np.zeros((3000,224))
for i in range(len(test)):
    for j in test[i]:
        test[i][j] = 224-j