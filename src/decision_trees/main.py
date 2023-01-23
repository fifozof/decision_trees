from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from DecisionTree import *
import numpy as np
from matplotlib import pyplot as plt

iris = load_iris()
wine = load_wine()
Xi, Yi = iris.data, iris.target
Xw, Yw = wine.data, wine.target
Yi = np.reshape(Yi, (150, -1))
Yw = np.reshape(Yw, (178, -1))


Xi_train, Xi_test, Yi_train, Yi_test = train_test_split(Xi, Yi, test_size=.2, random_state=32)
Xw_train, Xw_test, Yw_train, Yw_test = train_test_split(Xw, Yw, test_size=.2, random_state=32)

clf = tree.DecisionTreeClassifier(min_samples_split=3,max_depth=3)
clf = clf.fit(Xi_train, Yi_train)
Yi_pred_sk = clf.predict(Xi_test)

clf2 = tree.DecisionTreeClassifier(min_samples_split=3,max_depth=3)
clf2 = clf2.fit(Xw_train, Yw_train)
Yw_pred_sk = clf2.predict(Xw_test)

regressor = DecisionTreeClassifier(min_samples_left=3, max_tree_depth=3)
regressor.fit(Xi_train, Yi_train)
regressor.print_tree()

print("\n\n")

regressor2 = DecisionTreeClassifier(min_samples_left=3, max_tree_depth=3)
regressor2.fit(Xw_train, Yw_train)
regressor2.print_tree()


Yi_pred = regressor.predict(Xi_test)
tree.plot_tree(clf)
plt.show()

Yw_pred = regressor.predict(Xw_test)
tree.plot_tree(clf2)
plt.show()

if __name__ == "__main__":
    print(np.sqrt(mean_squared_error(Yi_test, Yi_pred)))
    print(np.sqrt(mean_squared_error(Yi_test, Yi_pred_sk)))
    print(np.sqrt(mean_squared_error(Yw_test, Yw_pred)))
    print(np.sqrt(mean_squared_error(Yw_test, Yw_pred_sk)))