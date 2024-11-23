import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.context_aware_recommender import DeepFM
from recbole.trainer import Trainer

# Считываем наши данные
train_interaction = pd.read_pickle("../../01_data/ready_data/02_ready_train_data.pkl")

# Батч для теста модели
train_interaction = train_interaction.sample(n=1_000_000)

# Создание данных interecation для работы с LightGCN
interaction = train_interaction.iloc[:, [0, 1, 8, 5, 7, 3, 4]]

# Преобразование данных
interaction["user_id"] = interaction["user_id"].astype("category").cat.codes
interaction["item_id"] = interaction["item_id"].astype("category").cat.codes
interaction["Gender"] = interaction["Gender"].astype("category").cat.codes
interaction["video_source_id"] = (
    interaction["video_source_id"].astype("category").cat.codes
)
interaction.info()

interaction.to_csv("final_dataset.csv", index=False)


# Загрузка конфигурации
config = Config(model="DeepFM", dataset="data.csv")

# Проверьте путь
print("Dataset path:", config["data_path"])
print("Dataset name:", config["dataset"])

# Создание датасета
dataset = create_dataset(config)

import os

print(os.getcwd())

data_path = "./"
dataset_file = os.path.join(data_path, "final_dataset.csv")

if os.path.exists(dataset_file):
    print("Файл найден:", dataset_file)
else:
    print("Файл не найден по пути:", dataset_file)
