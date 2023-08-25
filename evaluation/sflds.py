import pandas as pd
import torch
from sklearn.metrics import recall_score, f1_score, roc_auc_score

from models.sflds.bilstm import BILSTM


def run(device_type="cpu"):
    # recall, fscore, roc auc. In this case the model is a Neural Network
    device = torch.device(device_type)
    test_data = pd.read_csv('dist/sflds/df_test.csv')
    model = BILSTM(8146)
    model.load_state_dict(torch.load('dist/sflds/model2.pt', map_location=device))

    labels = test_data['bug']
    features = test_data.drop(['bug'], axis=1)

    # first 2600 columns are bfs, the rest are dfs
    bfs = features.iloc[:, :2600]
    dfs = features.iloc[:, 2600:]

    bfs = torch.tensor(bfs.values, dtype=torch.int32, device=device)
    dfs = torch.tensor(dfs.values, dtype=torch.int32, device=device)

    y_pred = model(bfs, dfs)

    y_pred_rounded = torch.round(y_pred)
    y_pred_rounded = y_pred_rounded.detach().numpy()
    y_pred = y_pred.detach().numpy()

    recall = recall_score(labels, y_pred_rounded)
    fscore = f1_score(labels, y_pred_rounded)
    roc_auc = roc_auc_score(labels, y_pred)

    print('SF-LDS')
    print(f'recall: {recall}')
    print(f'fscore: {fscore}')
    print(f'roc_auc: {roc_auc}')
