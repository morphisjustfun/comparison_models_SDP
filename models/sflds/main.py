import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from .bilstm import BILSTM


def run(epoch=20, lr=0.01, step_size=3, gamma=0.5, batch_size=40, device_type="cpu"):
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
    train_data = TensorDataset(torch.from_numpy(bfs_data.values), torch.from_numpy(dfs_data.values),
                               torch.from_numpy(labels_data.values))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = BILSTM(vocabSize)
    device = torch.device(device_type)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.BCELoss()
    for epoch in range(epoch):
        model.train()
        train_loss = 0
        for i, data in enumerate(train_loader, 0):
            bfs, dfs, labels = data
            bfs, dfs, labels = bfs.to(device), dfs.to(device), labels.float().to(device)
            labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(bfs, dfs)
            loss = criterion(outputs, labels)
            train_loss += loss
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")
        scheduler.step()

    # save model
    torch.save(model.state_dict(), 'dist/sflds/model.pt')
