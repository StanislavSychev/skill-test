import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class TrainDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == '__main__':
    train = pd.read_csv("train.csv")
    # import data to numpy and separate features from labels
    train = train.to_numpy()
    labels = train[:, -1].astype('int')
    features = train[:, 1:-1].astype('float')
    # fill np.inf with max in each column
    features_max = np.where(np.logical_or(features != features, features == np.inf), 0, features).max(axis=0)
    features = np.array([np.where(col == np.inf, col_max, col) for col, col_max in zip(features.T, features_max)])
    features = features.T

    # delete columns with 1 unique value, those are not useful
    useful = []
    for col in range(features.shape[1]):
        if len(np.unique(features[:, col])) == 1:
            continue
        useful.append(col)
    features = features[:, useful]

    # replace nan with mean
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    features = imputer.fit_transform(features)
    # scale data
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    # split data in train and test
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # make dataset from train
    dataset = TrainDataset(x_train, y_train)
    # count weights to make dataloader with equal class probability
    counts = np.bincount(y_train)
    weight = np.array([1 / counts[i] for i in y_train])
    sampler = WeightedRandomSampler(torch.tensor(weight), len(weight))
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, sampler=sampler)
    # import test to torch
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).long()

    max_epoch = 30
    torch.manual_seed(42)
    np.random.seed(42)

    n_hidden = 128
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, 2)
    )
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(max_epoch):
        for batch in dataloader:
            x, y = batch
            x = x.float()
            y = y.long()
            y_ = model(x)
            l = loss(y_, y)
            l.backward()
            optimizer.step()
        y_test_ = model(x_test)
        print(f'{epoch}/{max_epoch}\r', end="")
    sm = nn.Softmax(dim=1)
    print(roc_auc_score(y_test.numpy(), sm(y_test_).detach().numpy()[:, 1]))

    val_features = pd.read_csv("test.csv")
    val_features = val_features.to_numpy()
    val_labels = val_features[:, 0]
    val_features = val_features[:, 1:].astype('float')
    val_features = np.array([np.where(col == np.inf, col_max, col) for col, col_max in zip(val_features.T, features_max)])
    val_features = val_features.T

    val_features = val_features[:, useful]

    val_features = imputer.transform(val_features)
    val_features = scaler.transform(val_features)

    val_features = torch.tensor(val_features).float()
    pred = sm(model(val_features))[:, 1].detach().numpy()

    res = pd.DataFrame(data=np.concatenate((val_labels, pred)).reshape(2, -1).T, columns=['sample_id', 'y'])
    res.to_csv('submission.csv', index=False)
