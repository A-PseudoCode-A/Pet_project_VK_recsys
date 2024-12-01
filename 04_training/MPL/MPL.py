import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split


# Custom Dataset class for PyTorch
class InteractionDataset(Dataset):
    def __init__(self, dataframe):
        # Separating features from labels
        self.features = dataframe.iloc[:, 1:].drop(columns=["target"]).values
        self.labels = dataframe["target"].values

        # Converting to Tensors for PyTorch
        self.features = torch.tensor(self.features, dtype=torch.float32).to(
            device="cuda"
        )
        self.labels = torch.tensor(self.labels, dtype=torch.long).to(device="cuda")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# MPL model class where the principle of matrix calculation is explained
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


def train_MPL(input_file="02_train_data_for_training_MPL", epochs=50):
    """_Function for training the model. It describes the entire training and validation cycle

    Args:
        input_file (str, optional): The file where the dataset is stored after applying the train_data_for_training_DeepFM function. Defaults to "01_train_data_for_training_MPL".
        epochs (int, optional): number of full passes over the entire dataset. Defaults to 50.
    """
    # Reading data
    train_data = pd.read_pickle(
        f"../../03_feature_eng_and_ready_data/ready_data/{input_file}.pkl"
    )

    # Creating a training data matrix
    X = train_data.iloc[:, 1:].drop(columns=["target"]).to_numpy()

    # Creating a dataset for training in PyTorch
    train_dataset = InteractionDataset(train_data)

    # Creating train, val samples and loading them into DataLoader
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function and optimizer
    input_dim = X[0].shape[0]
    hidden_dim = 64
    output_dim = 3  # 3 classes: like, dislike, ignore
    model = MLP(input_dim, hidden_dim, output_dim).to(device="cuda")
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Epochs for training, as well as lists for storing metrics
    EPOCHS = epochs
    train_loss = []
    val_loss = []
    val_auc = []

    # Learning Cycle
    for epoch in range(EPOCHS):
        running_train_loss = []

        model.train()

        train_loop = tqdm(train_loader, leave=False)

        for x, targets in train_loop:
            # Larning Cycle
            pred = model(x)
            loss = loss_function(pred, targets)

            # Reverse Pass
            optimizer.zero_grad()
            loss.backward()

            # Optimization step
            optimizer.step()

            # Calculating the average value of the loss function
            running_train_loss.append(loss.item())
            mean_train_loss = sum(running_train_loss) / len(running_train_loss)

            # Additional information from tqdm
            train_loop.set_description(
                f"Epoch [{epoch+1} / {EPOCHS}], train_loss = {mean_train_loss:.4f}"
            )

        # Saving the value of the loss function
        train_loss.append(mean_train_loss)

        # Model Validation
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


# Calling functions
train_MPL()
