import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/new_train.csv', header=0)

x = df.values
y = df['Survived'].values

# remove survive column from data
x = np.delete(x, 1, axis=1)

# split data for cross validation

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# train model with Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=0)
clf.fit(x_train, y_train)
test_pred = clf.predict(x_test)
test_score = accuracy_score(y_test, test_pred)
print(test_score)
