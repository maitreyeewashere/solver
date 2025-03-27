import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def linear_regression():
    y = np.array([3, 4, 2, 5, 6, 8, 9, 10, 12, 11])

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Linear Regression')
    plt.show()

    return model.coef_, model.intercept_

#Kmeans
