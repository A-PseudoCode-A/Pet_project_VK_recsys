import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

# Устанавливаем настройки для вывода значений в np-array
np.set_printoptions(suppress=True)


# Считываем наши данные
train_data = pd.read_pickle("../../data/ready_data/02_ready_train_data.pkl")

# Длина входядящго слоя
X = train_data.iloc[:, 1:].drop(columns=["target"]).to_numpy()
X[0].shape


# Создаем собственный класс Dataset
class InteractionDataset(Dataset):
    def __init__(self, dataframe):
        # Отделяем признаки от меток
        self.features = dataframe.iloc[:, 1:].drop(columns=["target"]).values
        self.labels = dataframe["target"].values

        # Преобразуем в тензоры для PyTorch
        self.features = torch.tensor(self.features, dtype=torch.float32).to(
            device="cuda"
        )
        self.labels = torch.tensor(self.labels, dtype=torch.long).to(device="cuda")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Создаем наш dataset
train_dataset = InteractionDataset(train_data)

# Делаем train и val выборки
train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])

# Создаем класс DataLoader
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Step 3: Определение MLP модели
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)


# Инициализация модели, функции потерь и оптимизатора
input_dim = X[0].shape[0]
hidden_dim = 64
output_dim = 3  # 3 класса: like, dislike, ignore
model = MLP(input_dim, hidden_dim, output_dim).to(device="cuda")
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


EPOCHS = 50
train_loss = []
val_loss = []

val_auc = []

# Цикл обучения
for epoch in range(EPOCHS):
    running_train_loss = []

    model.train()

    train_loop = tqdm(train_loader, leave=False)

    for x, targets in train_loop:
        # Прямой проход
        pred = model(x)
        loss = loss_function(pred, targets)

        # Обратный проход
        optimizer.zero_grad()
        loss.backward()

        # Шаг оптимизации
        optimizer.step()

        # Расчет среднего значения функции потерь
        running_train_loss.append(loss.item())
        mean_train_loss = sum(running_train_loss) / len(running_train_loss)

        # Дополниетельная информация из tqdm
        train_loop.set_description(
            f"Epoch [{epoch+1} / {EPOCHS}], train_loss = {mean_train_loss:.4f}"
        )

    # Сохранение значения функции порерь
    train_loss.append(mean_train_loss)

    # Проверка модели - валидации
    model.eval()
    with torch.no_grad():
        running_val_loss = []
        running_val_auc = []

        for x, targets in val_loader:
            pred = model(x).detach().numpy()
            targets = targets.detach().numpy()

            loss = loss_function(pred, targets)

            running_val_loss.append(loss.item())
            mean_val_loss = sum(running_val_loss) / len(running_val_loss)

            roc_auc = roc_auc_score(targets, pred, multi_class="ovo")
            running_val_auc.append(roc_auc)

            mean_roc_auc = sum(running_val_auc) / len(running_val_auc)

        val_loss.append(mean_val_loss)
        val_auc.append(mean_roc_auc)

        print(
            f"Epoch [{epoch+1} / {EPOCHS}], train_loss = {mean_train_loss:.4f}, val_loss = {mean_val_loss:.4f}, val_roc_auc = {mean_roc_auc:.4f}"
        )
