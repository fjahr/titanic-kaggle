import pandas as pd
import numpy as np

vote1 = pd.read_csv('submission/adaboost_results.csv', header=0)
vote1 = vote1.ix[:, 1].as_matrix()
vote2 = pd.read_csv('submission/gradient_boosting_results.csv', header=0)
vote2 = vote2.ix[:, 1].as_matrix()
vote3 = pd.read_csv('submission/random_forest_results.csv', header=0)
vote3 = vote3.ix[:, 1].as_matrix()
vote4 = pd.read_csv('submission/regression_results.csv', header=0)
vote4 = vote4.ix[:, 1].as_matrix()
vote5 = pd.read_csv('submission/decision_tree_results.csv', header=0)
vote5 = vote5.ix[:, 1].as_matrix()

votes = vote1 + vote2 + vote3 + vote4 + vote5

votes = [0 if vote <= 2 else 1 for vote in votes]

# generate kaggle submission
df_test = pd.read_csv('data/new_test.csv', header=0)

output = np.column_stack((df_test.ix[:, 0].as_matrix(), votes))
df_results = pd.DataFrame(output.astype('int'), columns=['PassengerID', 'Survived'])
df_results.to_csv('submission/results.csv', index=False)
