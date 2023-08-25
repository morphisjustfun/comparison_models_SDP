import pickle

import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier


def run():
    train_data = pd.read_csv('dist/tree_ensembles/df_train.csv')
    labels = train_data['bug']
    features = train_data.drop(['bug'], axis=1)

    # use SMOTE to oversample
    sm = SMOTE()
    features, labels = sm.fit_resample(features, labels)

    clf = RandomForestClassifier(criterion='gini', max_features='sqrt')
    clf.fit(features, labels)

    # save model in dist/tree_ensembles/model.pkl
    pickle.dump(clf, open('dist/tree_ensembles/model.pkl', 'wb'))
