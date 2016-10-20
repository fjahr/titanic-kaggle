import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

df_train = pd.read_csv('data/new_train.csv', header=0)

x = df_train.values
y = df_train['Survived'].values

x = np.delete(x, 1, axis=1)

clf = AdaBoostClassifier(n_estimators=150)
clf.fit(x, y)

# generate kaggle submission
df_test = pd.read_csv('data/new_test.csv', header=0)
df_test['Fare'] = df_test['Fare'].interpolate()

test_results = clf.predict(df_test)
print(df_test.ix[:, 0].as_matrix())
print(test_results)
output = np.column_stack((df_test.ix[:, 0].as_matrix(), test_results))
df_results = pd.DataFrame(output.astype('int'), columns=['PassengerID', 'Survived'])
df_results.to_csv('submission/results.csv', index=False)
