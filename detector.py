# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


class MasksDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        current_sample = self.features[idx]
        current_target = self.targets[idx]
        
        return {
            "features": torch.tensor(current_sample, dtype=torch.float),
            "target": torch.tensor(current_target, dtype=torch.int)             
            }


class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16 * 30 * 30, 2)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        self.dropout = nn.Dropout(0.25)

        x = self.fc1(x)

        return x


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for row in loader:
            data = row['features']
            targets = row['target']
            
            x = data.to(device=device)
            y = targets.to(device=device, dtype=torch.int64)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples
    

if __name__ == '__main__':
    X = pickle.load(open("pickle/features.pickle", "rb"))
    y = pickle.load(open("pickle/targets.pickle", "rb"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify=y)
    
    train_dataset = MasksDataset(X_train, y_train)
    test_dataset = MasksDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, num_workers=2, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 64

    model = CNN(in_channels=3, num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch in train_loader:
            
            data = batch['features']
            targets = batch['target']

            data = data.to(device=device)
            targets = targets.to(device=device, dtype=torch.int64)
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

        print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
        print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")

    torch.save(model.state_dict(), 'model_weights.pth')
