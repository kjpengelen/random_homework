import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# function to print parameter values for q5-q8
def print_values(i, m_current, b_current, cost):
    print("Iteration: %.0f" % i)
    print("Intercept: %.4f" % m_current)
    print("Slope: %.3f" % b_current)
    print("Cost: %.4f" % cost)
    print("")

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

'''
# 3) Linear regression using Mean Squared Error
X = pd.DataFrame(data=trans_data['X'])  #put into dataframe suitable for regression lib
Y = pd.DataFrame(data=trans_data['Y'])

reg = linear_model.LinearRegression()
reg.fit(X, Y)   #fit linear model
print('Regression coefficient: %.2f' %reg.coef_)
print('Regression intercept: %.2f' %reg.intercept_)

y_pred = reg.predict(X) #predict y for given data X to derive MSE and variance of model
print('Mean squared error: %.2f' % mean_squared_error(Y, y_pred))
print('Variance score: %.2f' % r2_score(Y, y_pred))

#plot data and regression line
plt.scatter(X, Y, color='r')
plt.plot(X, reg.predict(X), color='b')
plt.show()

# 4) Linear Regression with Gradient Descent - loss function

# Choose a suitable loss function, motivate your choice
# Perform the partial derivatives of the loss function with respect to each of the regression parameters

from sklearn.linear_model import SGDRegressor

features = pd.DataFrame({'feat': ['1', '1', '1', '1', '1']},index=[0,1,2,3,4])
trans_data = pd.concat([trans_data, features],axis=1, sort=False)
X = trans_data.drop(columns=['Y'])
y = trans_data['Y']

LossFunctionA = 'squared_loss'
LossFunctionB = 'huber'
LossFunctionC = 'epsilon_insensitive'
LossFunctionD = 'squared_epsilon_insensitive'

loss = LossFunctionD

clf = linear_model.SGDRegressor(loss=loss, max_iter=100)
clf.fit(X, y)

print("Weights:" + str(clf.coef_))
print("Intercept: " + str(clf.intercept_))

y_pred = clf.predict(X) #predict y for given data X to derive MSE and variance of model
print("R2: " + str(clf.score(X, y_pred)))
'''

# 5-8) Linear Regression with GD - first iteration

# Put the linear regression in a function def since i needed it multiple times during construction.
def linear_regression(X, y, m_current=0, b_current=0, epochs=10, learning_rate=0.01):

    # Performs linear linear regression

    # epochs = number of iterations
    # learning_rate = step size

    # Output:
    # all output are vectors
    # m_hist = slopes that were used during the iterations
    # b_hist = intersections used during the iterations
    # c_hist = costs associated with these values

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

        if i == 1:
            print_values(i, m_current, b_current, cost)
        elif i == 2:
            print_values(i, m_current, b_current, cost)
        elif i == 3:
            print_values(i, m_current, b_current, cost)
        elif i == 1250:
            print_values(i, m_current, b_current, cost)
        elif i == 1500:
            print_values(i, m_current, b_current, cost)
        elif i == 2000:
            print_values(i, m_current, b_current, cost)

    return m_hist, b_hist, c_hist

# Values
lr = 0.01;
n_iter = 2001;

# Could put starting point at 1 and slope at -1, which gut feel says
# is a good starting point if you observe the points. However,
# doing that makes difference between iteration 1 and 2000+ less obvious.
starting_intersect = 0;
starting_slope = 0;

m_hist,b_hist,c_hist = linear_regression(trans_data.X, trans_data.Y, starting_slope, starting_intersect, n_iter, lr);

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