import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/new_train.csv', header=0)

x = df.values
y = df['Survived'].values

# remove survive column from data
x = np.delete(x, 1, axis=1)

# split data for cross validation

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# train model with Decision Tree
#
# clf = DecisionTreeClassifier(max_depth=5, random_state=0)
# clf.fit(x_train, y_train)
# test_pred = clf.predict(x_test)
# test_score = accuracy_score(y_test, test_pred)
# result: 0.815642458101

# train model with Random Forrest
#
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(x_train, y_train)
# score = clf.score(x_test, y_test)
# result: 0.837988826816

# train model with Gradient Boost

# clf = GradientBoostingClassifier(n_estimators=50)
# clf.fit(x_train, y_train)
# score = clf.score(x_test, y_test)
# result: 0.826815642458

# train AdaBoost Classifier
# clf = AdaBoostClassifier(n_estimators=150)
# clf.fit(x_train, y_train)
# score = clf.score(x_test, y_test)

# train logistic regression

clf = LogisticRegression()
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)

print(score)
