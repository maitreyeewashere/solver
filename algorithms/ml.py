import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Linear Regression
def linear_regression():
    # Sample Data
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
    y = np.array([3, 4, 2, 5, 6, 8, 9, 10, 12, 11])

    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # Plotting
    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Linear Regression')
    plt.show()

    return model.coef_, model.intercept_

#Kmeans