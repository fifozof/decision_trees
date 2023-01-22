from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from DecisionTree import *
import numpy as np
import graphviz
from matplotlib import pyplot as plt

iris = load_iris()
X, Y = iris.data, iris.target
Y = np.reshape(Y, (150, -1))



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=32)

clf = tree.DecisionTreeClassifier(min_samples_split=3,max_depth=3)
clf = clf.fit(X_train, Y_train)
Y_pred_sk= clf.predict(X_test)



regressor = DecisionTreeClassifier(min_samples_left=3, max_tree_depth=3)
regressor.fit(X_train, Y_train)
regressor.print_tree()


Y_pred = regressor.predict(X_test)


##dot_data = tree.export_graphviz(regressor, out_file=None,
             #     feature_names=iris.feature_names,
            #        class_names=iris.target_names,
            #       filled=True, rounded=True,
            #        special_characters=True)
#graph = graphviz.Source(dot_data)
#graph.render("iris")

tree.plot_tree(clf)
plt.show()

if __name__ == "__main__":
    print(np.sqrt(mean_squared_error(Y_test, Y_pred)))
    print(np.sqrt(mean_squared_error(Y_test, Y_pred_sk)))