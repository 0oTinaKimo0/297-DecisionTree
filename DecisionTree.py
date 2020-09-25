import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as pp
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Loading the Iris dataset from scikit-learn. Here, the third column represents the petal length, and the fourth column the petal width of the flower examples. The classes are already converted to integer labels where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.
diabetes = pd.read_csv('diabetes.csv', header=0)
diabetes.columns = ['PREG', 'GLU', 'BP', 'SKIN', 'INSU', 'BMI', 'DPF', 'AGE', 'OUT']
features = ['PREG', 'GLU', 'BP', 'SKIN', 'INSU', 'BMI', 'DPF', 'AGE']
X = diabetes[features]
y = diabetes['OUT'].T

# print('Class labels:', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# Standardizing the features:
# X_train = pp.scale(X_train)
# X_test = pp.scale(X_test)


# def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
#
#     # setup marker generator and color map
#     markers = ('s', 'x', 'o', '^', 'v')
#     colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#     cmap = ListedColormap(colors[:len(np.unique(y))])
#
#     # plot the decision surface
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#                            np.arange(x2_min, x2_max, resolution))
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)
#     plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())
#
#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[y == cl, 0],
#                     y=X[y == cl, 1],
#                     alpha=0.8,
#                     c=colors[idx],
#                     marker=markers[idx],
#                     label=cl,
#                     edgecolor='black')
#
#     # highlight test examples
#     if test_idx:
#         # plot all examples
#         X_test, y_test = X[test_idx, :], y[test_idx]
#
#         plt.scatter(X_test[:, 0],
#                     X_test[:, 1],
#                     c='none',
#                     edgecolor='black',
#                     alpha=1.0,
#                     linewidth=1,
#                     marker='o',
#                     s=100,
#                     label='test set')


# # Decision tree learning

# ## Maximizing information gain - getting the most bang for the buck


def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))


def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                          ['Entropy', 'Entropy (scaled)', 
                           'Gini impurity', 'Misclassification error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=5, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('impurity index')
#plt.show()

# Building a decision tree
depth_check = [4, 8, 12, 16, 20]
criterion_check = ['gini', 'entropy']
for i in depth_check:
    tree_model = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=1)
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    print(accuracy_score(y_test, y_pred))

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(X_combined, y_combined,
#                       classifier=tree_model,
#                       test_idx=range(105, 150))

# plt.xlabel('petal length [cm]')
# plt.ylabel('petal width [cm]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# #plt.savefig('images/03_20.png', dpi=300)
# plt.show()

tree.plot_tree(tree_model)
# plt.savefig('tree.png')
# plt.show()

