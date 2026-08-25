import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math


def set_seed(seed=27):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Datasets(Dataset):
    def __init__(self, dir="files", split="train", seed=27, max_len=None, norm_stats=None):
        self.ear_data = []
        self.mar_data = []
        self.labels = []
        self.masks = []

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
        r_ear, r_mar, r_label = [], [], []
        for video in videos:
            if not isinstance(video, dict):
                continue
            ear, mar, label = video.get("ear_values", []), video.get("mar_values", []), video.get("label", 0)
            r_ear.append(ear)
            r_mar.append(mar)
            r_label.append(label)


        computed_max = int(np.percentile([len(i) for i in r_ear], 95))
        self.max_len = computed_max

        for ear, mar, label in zip(r_ear, r_mar, r_label):
            seq = min(len(ear), len(mar))
            seq = min(seq, self.max_len)
            ear_seq = ear[:seq]
            mar_seq = mar[:seq]
            pad_len = self.max_len - seq

            ear_padded = ear_seq + [0.0] * pad_len
            mar_padded = mar_seq + [0.0] * pad_len
            mask = [1.0] * seq + [0.0] * pad_len

            self.ear_data.append(ear_padded)
            self.mar_data.append(mar_padded)
            self.masks.append(mask)
            self.labels.append(label)

        ear_arr = np.array(self.ear_data, dtype=np.float32)
        mar_arr = np.array(self.mar_data, dtype=np.float32)
        mask_arr = np.array(self.masks, dtype=np.float32)

        if norm_stats is None:
            valid = mask_arr == 1
            ear_mean, ear_std = ear_arr[valid].mean(), ear_arr[valid].std() + 0.0001
            mar_mean, mar_std = mar_arr[valid].mean(), mar_arr[valid].std() + 0.0001
            self.norm_stats = (float(ear_mean), float(ear_std), float(mar_mean), float(mar_std))
        else:
            self.norm_stats = norm_stats

        ear_mean, ear_std, mar_mean, mar_std = self.norm_stats
        ear_arr = (ear_arr - ear_mean) / ear_std
        mar_arr = (mar_arr - mar_mean) / mar_std
        self.ear_data = ear_arr.tolist()
        self.mar_data = mar_arr.tolist()


    def __getitem__(self, idx):
        x = torch.tensor([self.ear_data[idx], self.mar_data[idx]], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {"x": x, "mask": mask, "label": label}

    def __len__(self):
        return len(self.labels)


class chomp(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dil, dropout=0.3):
        super().__init__()
        padding = (kernel_size - 1) * dil
        self.layer = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, dilation=dil)),
            chomp(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.utils.parametrizations.weight_norm(nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, padding=padding, dilation=dil)),
            chomp(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.change = nn.Conv1d(in_channel, out_channel, 1) if in_channel != out_channel else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer(x)
        res = x if self.change is None else self.change(x)
        return self.relu(out + res)


class Masked(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        maske = mask.unsqueeze(1)
        x_masked = x.masked_fill(maske == 0, float('-inf'))
        pool = torch.max(x_masked, dim=-1)[0]
        return pool


class MultiBlock(nn.Module):
    def __init__(self, num_blocks, in_dim, feature_dim, kernel_size, dropout=0.3):
        super().__init__()
        self.blocks = nn.ModuleList([
            block(in_channel=in_dim if i == 0 else feature_dim,
                  out_channel=feature_dim,
                  kernel_size=kernel_size,
                  dil=2 ** i,
                  dropout=dropout
                  ) for i in range(num_blocks)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class oneD_TCN(nn.Module):
    def __init__(self, in_dim, feature_dim, long, base=2, kernel_size=3, dropout=0.3):
        super().__init__()
        self.long = long
        self.base = base
        self.kernel_size = kernel_size
        self.n = max(1, math.ceil(
            math.log((long - 1) * (base - 1) / (2 * (kernel_size - 1)) + 1, base)
        ))

        self.tcn = MultiBlock(
            num_blocks=self.n,
            in_dim=in_dim,
            feature_dim=feature_dim,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.pool = Masked()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, x, mask):
        ft = self.tcn(x)
        pooled = self.pool(ft, mask)
        out = self.fc(pooled)
        return out


set_seed(27)

train_ds = Datasets(split="train")
val_ds = Datasets(split="val", max_len=train_ds.max_len, norm_stats=train_ds.norm_stats)
test_ds = Datasets(split="test", max_len=train_ds.max_len, norm_stats=train_ds.norm_stats)


batch_size = 32
train_da = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_da = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_da = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = oneD_TCN(in_dim=2, feature_dim=64, long=train_ds.max_len).to(device)


labels_arr = np.array(train_ds.labels, dtype=np.float32)
n_pos = labels_arr.sum()
n_neg = len(labels_arr) - n_pos
pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
epochs = 20

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    cor = 0
    total = 0

    for batch_data in train_da:
        x = batch_data["x"].to(device)
        mask = batch_data["mask"].to(device)
        labels = batch_data["label"].to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(x, mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        cor += (preds == labels).sum().item()
        total += len(labels)

    train_loss = total_loss / total
    train_acc = cor / total * 100

    model.eval()
    total_loss = 0.0
    cor = 0
    total = 0

    val_pbar = tqdm(
        val_da,
        desc=f"Epoch {epoch:02d}/{epochs:02d}",
        leave=False,
        ncols=100
    )

    with torch.no_grad():
        for batch_data in val_pbar:
            x = batch_data["x"].to(device)
            mask = batch_data["mask"].to(device)
            labels = batch_data["label"].to(device).unsqueeze(1)

            outputs = model(x, mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            cor += (preds == labels).sum().item()
            total += len(labels)

    val_loss = total_loss / total
    val_acc = cor / total * 100

    tqdm.write(f"{epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f}, Acc: {train_acc:5.1f}% - Val Loss: {val_loss:.4f}, Acc: {val_acc:5.1f}")