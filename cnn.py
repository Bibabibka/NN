import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Datasets(Dataset):
    def __init__(self, dir="files", split="train", window=30, step=10, seed=27):
        self.window = window
        self.step = step
        self.ear_data = []
        self.mar_data = []
        self.labels = []

        all_files = [os.path.join(dir, f"raw_features_{i}_{j}.json") for i in range(1, 6) for j in range(1, 3)]

        random.seed(seed)
        random.shuffle(all_files)
        n = len(all_files)
        train, val = int(n * 0.7), int(n * 0.85)

        if split == "train":
            files = all_files[:train]
        elif split == "val":
            files = all_files[train:val]
        else:
            files = all_files[val:]

        videos = []
        for path in files:
            with open(path) as fp:
                data = json.load(fp)
                videos.extend(data.get("alert", []) + data.get("tired", []) if isinstance(data, dict) else data)

        for video in videos:
            if not isinstance(video, dict):
                continue
            ear, mar, label = video.get("ear_values", []), video.get("mar_values", []), video.get("label", 0)
            if len(ear) < window or len(mar) < window:
                continue

            ear = np.array(ear, dtype=np.float32)
            mar = np.array(mar, dtype=np.float32)
            ear = (ear - np.mean(ear)) / (np.std(ear) + 1e-6)
            mar = (mar - np.mean(mar)) / (np.std(mar) + 1e-6)

            for i in range(0, len(ear) - window + 1, step):
                self.ear_data.append(ear[i:i + window])
                self.mar_data.append(mar[i:i + window])
                self.labels.append(label)

    def __getitem__(self, idx):
        return {
            "ear": torch.tensor(self.ear_data[idx], dtype=torch.float32),
            "mar": torch.tensor(self.mar_data[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.labels)


class CNN(nn.Module):
    def __init__(self, window_size=30):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(2, 5), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 5), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(kernel_size=(1, 2))

        inp = torch.zeros(1, 1, 2, window_size)
        out = self._forward_conv(inp)
        dim = out.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def _forward_conv(self, x):
        x = self.pool(functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(functional.relu(self.bn2(self.conv2(x))))
        return x

    def forward(self, ear, mar):
        x = torch.stack([ear, mar], dim=1).unsqueeze(1)
        x = self._forward_conv(x)
        return self.classifier(x.flatten(1))


window, batch, epochs = 30, 32, 20
train_ds = Datasets(split="train", window=window)
val_ds = Datasets(split="val", window=window)
test_ds = Datasets(split="test", window=window)
train_da = DataLoader(train_ds, batch, shuffle=True)
val_da = DataLoader(val_ds, batch)
test_da = DataLoader(test_ds, batch)

device = torch.device("cpu")
model = CNN(window).to(device)
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    cor = 0
    total = 0
    for batch_data in train_da:
        ear = batch_data["ear"].to(device)
        mar = batch_data["mar"].to(device)
        labels = batch_data["label"].to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(ear, mar)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        cor += ((torch.sigmoid(outputs) > 0.5).float() == labels).sum().item()
        total += len(labels)

    train_loss = total_loss / total
    train_acc = cor / total * 100

    model.eval()
    total_loss = 0
    cor = 0
    total = 0

    val_pbar = tqdm(
        val_da,
        desc=f"{epoch:02d}/{epochs:02d}",
        leave=False,
        ncols=100
    )

    with torch.no_grad():
        for batch_data in val_pbar:
            ear = batch_data["ear"].to(device)
            mar = batch_data["mar"].to(device)
            labels = batch_data["label"].to(device).unsqueeze(1)

            outputs = model(ear, mar)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            cor += ((torch.sigmoid(outputs) > 0.5).float() == labels).sum().item()
            total += len(labels)
            val_pbar.set_postfix({"loss": f"{total_loss / total:.4f}", "acc": f"{cor / total * 100:.1f}%"})

    val_loss = total_loss / total
    val_acc =  cor / total * 100
    tqdm.write(f"{epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f}, Acc: {train_acc:5.1f}% - Val Loss: {val_loss:.4f}, Acc: {val_acc:5.1f}")