import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.context_aware_recommender import DeepFM
from recbole.trainer import Trainer
from recbole.quick_start import run_recbole

from logging import getLogger
from recbole.utils import init_seed, init_logger

config_dict = {
    "model": "DeepFM",
    "dataset": "data_for_DeepFM",
    "data_path": "../../02_feature_eng_and_ready_data/ready_data",  # Путь к папке с файлом data.inter
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
