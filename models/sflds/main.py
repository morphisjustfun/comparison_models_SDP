import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from .bilstm import BILSTM


def run(epoch=100, lr=0.01, step_size=3, gamma=0.5, batch_size=15, device_type="cpu"):
    trainData = pd.read_csv('dist/sflds/df_train.csv')
    df_majority = trainData[trainData.bug == 0]
    df_minority = trainData[trainData.bug == 1]

    # Count how many samples for the majority class
    majority_count = df_majority.shape[0]

    # Upsample minority class
    # use SMOTE
    df_minority_unsampled = df_minority.sample(majority_count, replace=True)

    # Combine majority class with upsampled minority class
    trainData_balanced = pd.concat([df_majority, df_minority_unsampled], axis=0)
    trainData_balanced = trainData_balanced.sample(frac=1)
    trainData_balanced = trainData_balanced.reset_index(drop=True)

    bfs_data = trainData_balanced.iloc[:, 0:2600]
    dfs_data = trainData_balanced.iloc[:, 2600:5200]
    labels_data = trainData_balanced['bug']
    vocabSize = bfs_data.max().max() + 1

    # oversampling SMOGN, SMOTER, None
    # normalize use Z, min-max, None
    bfs_data, bfs_eval, dfs_data, dfs_eval, labels_data, labels_eval = train_test_split(bfs_data, dfs_data, labels_data,
                                                                                        test_size=0.1)

    train_data = TensorDataset(torch.from_numpy(bfs_data.values), torch.from_numpy(dfs_data.values),
                               torch.from_numpy(labels_data.values))
    eval_data = TensorDataset(torch.from_numpy(bfs_eval.values), torch.from_numpy(dfs_eval.values),
                              torch.from_numpy(labels_eval.values))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = BILSTM(vocabSize)
    device = torch.device(device_type)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.BCELoss()
    for epoch in range(epoch):
        model.train()
        for i, data in enumerate(train_loader, 0):
            bfs, dfs, labels = data
            bfs, dfs, labels = bfs.to(device), dfs.to(device), labels.float().to(device)
            labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(bfs, dfs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            bfs, dfs, labels = eval_data.tensors
            bfs, dfs, labels = bfs.to(device), dfs.to(device), labels.float().to(device)
            labels = labels.view(-1, 1)
            outputs = model(bfs, dfs)
            val_loss = criterion(outputs, labels)
        print(f"Epoch {epoch + 1}: Eval Loss = {val_loss:.4f}")
        scheduler.step()

    # save model
    torch.save(model.state_dict(), 'dist/sflds/model.pt')
