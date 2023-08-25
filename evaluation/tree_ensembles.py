import pandas as pd
from sklearn.metrics import recall_score, f1_score, roc_auc_score


def run():
    # recall, fscore, roc auc. In this case the model is a Random Forest
    test_data = pd.read_csv('dist/tree_ensembles/df_test.csv')
    labels = test_data['bug']
    features = test_data.drop(['bug'], axis=1)
    # model is at dist/tree_ensembles/model.pkl already trained
    clf = pd.read_pickle('dist/tree_ensembles/model.pkl')

    y_pred = clf.predict(features)

    recall = recall_score(labels, y_pred)
    fscore = f1_score(labels, y_pred)
    roc_auc = roc_auc_score(labels, y_pred)

    print('Tree Ensembles')
    print(f'recall: {recall}')
    print(f'fscore: {fscore}')
    print(f'roc_auc: {roc_auc}')
