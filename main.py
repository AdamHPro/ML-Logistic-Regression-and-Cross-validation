import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from computeCost import computeCost
from computeGrad import computeGrad
from predict import predict
 
# Load the dataset
# The first two columns contains the exam scores and the third column
# contains the label.
data = np.loadtxt('data1.txt', delimiter=',')
 
X = data[:, 0:2]
y = data[:, 2]

# Plot data 
pos = np.where(y == 1)
neg = np.where(y == 0)
plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not Admitted'])
plt.show()


#Ajout des 1 pour le biais et séparation training/test
X_new = np.ones((X.shape[0], 3))
X_new[:, 1:3] = X
X = X_new[:60, :]
X_test = X_new[60:, :]


# Initialize fitting parameters
initial_theta = np.zeros((3,))

# Run minimize() to obtain the optimal theta
Result = op.minimize(fun = computeCost, x0 = initial_theta, args = (X, y), method = 'TNC', jac = computeGrad);
theta = Result.x;


# Plot the decision boundary
plot_x = np.array([min(X_new[:, 1]), max(X_new[:, 1])])
plot_y = (- 1.0 / theta[2]) * (theta[1] * plot_x + theta[0])
plt.plot(plot_x, plot_y)
plt.scatter(X_new[pos, 1], X_new[pos, 2], marker='o', c='b')
plt.scatter(X_new[neg, 1], X_new[neg, 2], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Decision Boundary', 'Admitted', 'Not Admitted'])
plt.show()


# Compute accuracy on the training set
p = predict(np.array(theta), X_test)
counter = 0
for i in range(p.size):
    if p[i] == y[i+60]:
        counter += 1
print('Train Accuracy: {:.2f}'.format(counter / float(p.size) * 100.0))
