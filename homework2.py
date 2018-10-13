import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

# 1) Data acquisition
data = pd.read_excel('HW2data.xlsx')
row_number = (19-1)*5
data.drop(data.index[:row_number], inplace=True) # remove data of first 18 groups
data = data.head(5) # select our five datapoints
data = data.reset_index(drop=True)
print("Original data")
print(data)
print("")

# 2) Data transformation
ScalerA = preprocessing.StandardScaler()
ScalerB = preprocessing.MinMaxScaler()
ScalerC = preprocessing.Normalizer()

scaler = ScalerB

trans_data = scaler.fit_transform(data) # transform data based on selected scaler
trans_data = pd.DataFrame(trans_data, columns=['X','Y']) # give columns names X and Y
print("Transformed data")
print(trans_data)
print("")

# Put the linear regression in a function def since i needed it multiple times during construction.
def linear_regression(X, y, m_current=0, b_current=0, epochs=10, learning_rate=0.01):
     N = float(len(y))
     m_hist = np.zeros((epochs,2));
     b_hist = np.zeros((epochs,2));
     c_hist = np.zeros(epochs);
     for i in range(epochs):
          y_current = (m_current * X) + b_current
          cost = sum([data**2 for data in (y-y_current)]) / N
          m_gradient = -(2/N) * sum(X * (y - y_current))
          b_gradient = -(2/N) * sum(y - y_current)
          m_current = m_current - (learning_rate * m_gradient)
          b_current = b_current - (learning_rate * b_gradient)

          # Storing values of this iterations
          m_hist[i,:] = m_current
          b_hist[i,:] = b_current
          c_hist[i] = cost
     return m_hist, b_hist, c_hist

# Values
lr = 0.01;
n_iter = 2000;

m_hist,b_hist,c_hist = linear_regression(trans_data.X, trans_data.Y, 0, 0, n_iter, lr);

# A big figure containing all our plots
fig = plt.figure(figsize=(5,4),dpi=200)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# The regression lines plotted
shown = [0,1,2,n_iter-1];
count = 0;
for i in shown:
    count += 1;
    ax = fig.add_subplot(4, 2, count);
    ax.set_xlim(0,1);
    ax.plot(trans_data.X, trans_data.Y,'b.');
    x_vals = np.array(ax.get_xlim());
    y_vals = b_hist[i] + m_hist[i] * x_vals;
    ax.plot(x_vals, y_vals,'r-');

# Adding the cost graph
ax1 = fig.add_subplot(4, 2, count+1);
ax1.plot(range(n_iter),c_hist,'b.')

plt.show();
