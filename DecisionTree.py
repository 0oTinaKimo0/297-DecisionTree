import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as pp
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


# Loading the diabetes dataset from the csv file using pandas
diabetes = pd.read_csv('diabetes.csv', header=0)
diabetes.columns = ['PREG', 'GLU', 'BP', 'SKIN', 'INSU', 'BMI', 'DPF', 'AGE', 'OUT']
features = ['PREG', 'GLU', 'BP', 'SKIN', 'INSU', 'BMI', 'DPF', 'AGE']
X = diabetes[features]
y = diabetes['OUT'].T
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# Feature standardization (which we choose to omit)
# X_train = pp.scale(X_train)
# X_test = pp.scale(X_test)

# Impurity metrics for the decision tree - maximizing information gain
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
# plt.show()

# Building a decision tree
# depth_check = [4, 8, 12, 16, 20] # 8 was optimal!
# criterion_check = ['gini', 'entropy']
# for i in depth_check:
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=8, random_state=1)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
print("Decision Tree Accuracy: %3f" % accuracy_score(y_test, tree_pred))
print("\n", classification_report(y_test, tree_pred))

tree.plot_tree(tree_model)
# plt.savefig('treegraph.png')
# plt.show()

# Building a random forest
# estimators_check = [10, 25, 40, 55, 70, 85, 100, 200, 300] # 200 was optimal!
# for i in estimators_check:
forest = RandomForestClassifier(criterion='gini', bootstrap=False, n_estimators=200, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
forest_pred = forest.predict(X_test)
print("Random Forest Accuracy: %3f" % accuracy_score(y_test, forest_pred))
print("\n", classification_report(y_test, forest_pred))
