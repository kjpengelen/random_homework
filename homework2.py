import matplotlib.pyplot as plt
import numpy as np

#data

x = [26.37,
26.58,
15.25,
21.24,
24.98
]

y = [451.78,
439.46,
467.23,
459.81,
447.15
]

#scaling
x_t = [(x_i - min(x)) / (max(x) - min(x)) for x_i in x]
#print(minmax_x)
y_t = [(y_i - min(y)) / (max(y) - min(y)) for y_i in y]
#print(minmax_y)

#plt.scatter(x_t,y_t)
#plt.show()

#SSE
#transform X to matrix
#m_x = np.matrix(x_t)
#m_y = np.matrix(y_t)
x_mean = sum(x_t)/len(x_t)
y_mean = sum(y_t)/len(y_t)

x_var = 0
y_var = 0

for x in x_t:
    x_var = x_var + ((x-x_mean)**2)
for y in y_t:
    y_var = y_var + ((y-y_mean)**2)

# cov = 0
#
# for i in range(x_t):
#     cov = cov + ((x_t[i] - x_mean) * (y_t[i] - y_mean))
#     print(cov)

