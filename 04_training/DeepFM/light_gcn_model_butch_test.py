import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.context_aware_recommender import DeepFM
from recbole.trainer import Trainer

from logging import getLogger
from recbole.utils import init_seed, init_logger

# Считываем наши данные
train_interaction = pd.read_pickle("../../01_data/ready_data/02_ready_train_data.pkl")

# Батч для теста модели
train_interaction = train_interaction.sample(n=10_000_000)

# Создание данных interecation для работы с LightGCN
interaction = train_interaction.iloc[:, [0, 1, -1, 5, 7, 3, 4]]

interaction = interaction.rename(
    columns={
        "user_id": "user_id:token",
        "item_id": "item_id:token",
        "target": "label:float",
        "user_age": "user_age:token",
        "Gender": "gender:token",
        "video_source_id": "video_source_id:token",
        "video_duration": "video_duration:token",
    }
)


interaction["user_id:token"] = interaction["user_id:token"].astype(int)
interaction["item_id:token"] = interaction["item_id:token"].astype(int)
interaction["label:float"] = interaction["label:float"].astype(float)
interaction["user_age:token"] = interaction["user_age:token"].astype(int)
interaction["gender:token"] = interaction["gender:token"].astype(int)
interaction["video_source_id:token"] = interaction["video_source_id:token"].astype(int)
interaction["video_duration:token"] = interaction["video_duration:token"].astype(int)

interaction.info()

interaction.iloc[:, [0, 1, 2]].to_csv("data.inter", index=False)
interaction.iloc[:, [0, 3, 4]].to_csv("data.user", index=False)
interaction.iloc[:, [1, 5, 6]].to_csv("data.item", index=False)


config_dict = {
    "model": "DeepFM",
    "dataset": "data",
    "data_path": "../DeepFM",  # Путь к папке с файлом data.inter
    "field_separator": ",",
    "USER_ID_FIELD": "user_id",
    "ITEM_ID_FIELD": "item_id",
    "LABEL_FIELD": "label",
    "RATING_FIELD": None,
    "TIME_FIELD": None,
    "load_col": {
        "inter": ["user_id", "item_id", "label"],
        "user": ["user_id", "gender", "age"],
        "item": ["item_id", "video_source_id", "video_duration"],
    },
    "epochs": 10,
    "learning_rate": 0.001,
    "embedding_size": 64,
    "train_batch_size": 256,
}

# configurations initialization
config = Config(config_dict=config_dict)

# init random seed
init_seed(config["seed"], config["reproducibility"])

# logger initialization
init_logger(config)
logger = getLogger()

# write config info into log
logger.info(config)

# dataset creating and filtering
dataset = create_dataset(config)
logger.info(dataset)

# dataset splitting
train_data, valid_data, test_data = data_preparation(config, dataset)

# model loading and initialization
model = DeepFM(config, train_data.dataset).to("cuda")
logger.info(model)

# trainer loading and initialization
trainer = Trainer(config, model)

# model training
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

# model evaluation
test_result = trainer.evaluate(test_data)
print(test_result)
