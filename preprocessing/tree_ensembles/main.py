import pandas as pd
from sklearn.model_selection import train_test_split


def run(versions=None):
    if versions is None:
        versions = ['1.0', '1.2', '1.4', '1.6']

    dfs = []
    for version in versions:
        df = pd.read_csv(f'datasets/camel{version}/labels.csv')
        df.drop(['name'], axis=1, inplace=True)
        df['bug'] = df['bug'].apply(lambda x: 1 if x > 0 else 0)
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv('dist/tree_ensembles/df.csv', index=False)
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv('dist/tree_ensembles/df_train.csv', index=False)
    test.to_csv('dist/tree_ensembles/df_test.csv', index=False)
