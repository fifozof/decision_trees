from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from DecisionTree import *
import numpy as np


iris = load_iris()
X, Y = iris.data, iris.target
Y = np.reshape(Y, (150, -1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=32)

regressor = DecisionTreeClassifier(min_samples_left=3, max_tree_depth=3)
regressor.fit(X_train, Y_train)
regressor.print_tree()


Y_pred = regressor.predict(X_test)
print(np.sqrt(mean_squared_error(Y_test, Y_pred)))

if __name__ == "__main__":
    print(np.sqrt(mean_squared_error(Y_test, Y_pred)))
