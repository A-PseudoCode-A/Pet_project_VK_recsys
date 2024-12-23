from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.context_aware_recommender import DeepFM
from recbole.trainer import Trainer

from logging import getLogger
from recbole.utils import init_seed, init_logger


def train_model_DeepFM():
    """Function for training DeepFM. The model takes data from atomic files prepared in advance by the function atomic_files_for_training_DeepFM.
    The file with the model settings can be changed inside the function itself."""

    # Configuration dictionary for setting up the DeepFM model
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
        "epochs": 3,
        "learning_rate": 0.001,
        "embedding_size": 64,
        "train_batch_size": 256,
    }

    # Configurations initialization
    config = Config(config_dict=config_dict)

    # Init random seed
    init_seed(config["seed"], config["reproducibility"])

    # Logger initialization
    init_logger(config)
    logger = getLogger()

    # Write config info into log
    logger.info(config)

    # Dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # Dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Model loading and initialization
    model = DeepFM(config, train_data.dataset).to("cuda")
    logger.info(model)

    # Trainer loading and initialization
    trainer = Trainer(config, model)

    # Model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # Model evaluation
    test_result = trainer.evaluate(test_data)
    print(test_result)
