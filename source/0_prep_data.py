import pandas as pd

df = pd.read_csv('data/kaggle/train.csv', header=0)

# drop columns that are not needed
cols = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols, axis=1)

# df.info()

# add one hot encoding columns for non numeric category data
one_hot_cols = []
cols = ['Pclass', 'Sex', 'Embarked']

for col in cols:
    one_hot_cols.append(pd.get_dummies(df[col]))

titanic_dummies = pd.concat(one_hot_cols, axis=1)

df = pd.concat((df, titanic_dummies), axis=1)

# remove old category columns

df = df.drop(['Pclass', 'Sex', 'Embarked'], axis=1)

# fill missing age rows

df['Age'] = df['Age'].interpolate()

# save intermediary data

df.to_csv('data/new_train.csv')
