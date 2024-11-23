import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Устанавливаем настройки для вывода значений в np-array
np.set_printoptions(suppress=True)


# Считываем наши данные
train_data = pd.read_pickle("../../01_data/ready_data/02_ready_train_data.pkl")
train_data = train_data.sample(n=1_000_000)

# Длина входядящго слоя
X = train_data.iloc[:, 1:].drop(columns=["target"]).to_numpy()
X.shape[1]


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
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
        self.do = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.do(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.do(x)
        
        x = self.fc3(x)
        return x


# Инициализация модели, функции потерь и оптимизатора
input_dim = X.shape[1]
hidden_dim = 64
output_dim = 3  # 3 класса: like, dislike, ignore
model = MLP(input_dim, hidden_dim, output_dim).to(device="cuda")
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode="min", factor=0.1, patience=5
)


EPOCHS = 10
train_loss = []
val_loss = []
val_auc = []
lr_list = []

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

        # Дополнительная информация из tqdm
        train_loop.set_description(
            f"Epoch [{epoch+1} / {EPOCHS}], train_loss = {mean_train_loss:.4f}"
        )

    # Сохранение значения функции потерь для train
    train_loss.append(mean_train_loss)

    # Проверка модели - валидация
    model.eval()
    with torch.no_grad():
        running_val_loss = []
        all_preds = []  # для хранения всех предсказанных вероятностей
        all_targets = []  # для хранения всех истинных меток

        for x, targets in val_loader:
            pred = model(x)

            # Вычисление функции потерь
            loss = loss_function(pred, targets)
            running_val_loss.append(loss.item())

            # Применяем Softmax к выходам модели, чтобы получить вероятности
            pred_probs = F.softmax(pred, dim=1)

            # Сохраняем предсказания и метки
            all_preds.append(pred_probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        mean_val_loss = sum(running_val_loss) / len(running_val_loss)
        val_loss.append(mean_val_loss)

        # Объединяем все батчи для расчета метрики roc_auc_score
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Расчет ROC AUC для многоклассовой задачи
        mean_roc_auc = roc_auc_score(all_targets, all_preds, multi_class="ovo")
        val_auc.append(mean_roc_auc)

        # Работа с шедулером для изменения скорости обучения
        lr_sheduler.step(mean_val_loss)
        lr = lr_sheduler._last_lr[0]
        lr_list.append(lr)

        print(
            f"Epoch [{epoch+1} / {EPOCHS}], train_loss = {mean_train_loss:.4f}, val_loss = {mean_val_loss:.4f}, val_roc_auc = {mean_roc_auc:.4f}, lr = {lr}"
        )


plt.plot(train_loss[:1])
plt.plot(val_loss[:4])
plt.legend(["loss_train", "loss_val"])
plt.show()
