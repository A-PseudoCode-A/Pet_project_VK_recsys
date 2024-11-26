import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.context_aware_recommender import DeepFM
from recbole.trainer import Trainer
from recbole.quick_start import run_recbole

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
interaction = interaction.iloc[:, [0, 1, 2]].rename(
    columns={
        "user_id": "user_id:token",
        "item_id": "item_id:token",
        "target": "rating:float",
    }
)
interaction["rating:float"] = interaction["rating:float"].astype(float)

interaction.to_csv("data.inter", index=False)


config_dict = {
    "model": "DeepFM",
    "dataset": "data",
    "data_path": "../DeepFM",  # Путь к папке с файлом data.inter
    "field_separator": "	",
    "epochs": 10,
    "learning_rate": 0.001,
    "embedding_size": 64,
    "train_batch_size": 256,
    "neg_sampling": None,
}

run_recbole(config_dict=config_dict)
