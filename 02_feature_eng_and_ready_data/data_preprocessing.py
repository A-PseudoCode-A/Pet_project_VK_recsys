import pandas as pd


def train_data_for_training_MPL(
    input_file_name="01_train_data_for_preprocessing",
    output_file_name="02_train_data_for_training_MPL",
    dataset_length=None,
):
    """A function that modifies a dataset to prepare it for training.

    Args:
        input_file_name (str, optional): The file where the dataset is stored. Defaults to "01_train_data_for_preprocessing".
        output_file_name (str, optional): the file name you want to get. Defaults to '02_train_data_for_training'.

    Returns:
        output_file_name.pkl: Prepared dataset in pickle format
    """

    # Reading prepared data
    train_data = pd.read_pickle(f"ready_data/{input_file_name}.pkl")

    if dataset_length is not None:
        train_data = train_data.sample(n=dataset_length)

    train_data["user_gender"] = train_data.apply(
        lambda row: 1 if row["user_gender"] == 2 else 0, axis=1
    )

    # Создание колонки 'target' на основе значений в 'like' и 'dislike'
    train_data["target"] = train_data.apply(
        lambda row: 2 if row["like"] == 1 else (1 if row["dislike"] == 1 else 0), axis=1
    )

    # Удаление ненужных колонок (пробуем пока что обучить без share и bookmarks)
    del train_data["share"]
    del train_data["bookmarks"]
    del train_data["like"]
    del train_data["dislike"]

    # Возвращаем новые готовые данные для обучения
    return train_data.to_pickle(f"ready_data/{output_file_name}.pkl")


def train_data_for_training_DeepFM(
    input_file_name="01_train_data_for_preprocessing",
    output_file_name="02_train_data_for_training_DeepFM",
    dataset_length=None,
):
    # Reading prepared data
    train_data = pd.read_pickle(f"ready_data/{input_file_name}.pkl")

    if dataset_length is not None:
        train_data = train_data.sample(n=dataset_length)

    train_data["user_gender"] = train_data.apply(
        lambda row: 1 if row["user_gender"] == 2 else 0, axis=1
    )

    # Создание колонки 'target' на основе значений в 'like' и 'dislike'
    train_data["target"] = train_data.apply(
        lambda row: 2 if row["like"] == 1 else (1 if row["dislike"] == 1 else 0), axis=1
    )

    # Удаление ненужных колонок (пробуем пока что обучить без share и bookmarks)
    del train_data["share"]
    del train_data["bookmarks"]
    del train_data["like"]
    del train_data["dislike"]
    del train_data["not_full_watched"]

    return train_data.to_pickle(f"ready_data/{output_file_name}.pkl")
    # Возвращаем новые готовые данные для обучения


def atomic_files_for_training_DeepFM(data_lengh=None):
    data_lengh = 1_000_000
    train_interaction = pd.read_pickle(
        "ready_data/02_train_data_for_training_DeepFM.pkl"
    )

    if data_lengh is not None:
        train_interaction = train_interaction.sample(n=data_lengh)

    # Создание данных interecation для работы с LightGCN
    interaction = train_interaction.iloc[:, [0, 1, -1, 6, 7, 3, 4]]

    interaction = interaction.rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "target": "label:float",
            "user_age": "user_age:token",
            "user_gender": "user_gender:token",
            "video_source_id": "video_source_id:token",
            "video_duration": "video_duration:token",
        }
    )

    interaction["user_id:token"] = interaction["user_id:token"].astype(int)
    interaction["item_id:token"] = interaction["item_id:token"].astype(int)
    interaction["label:float"] = interaction["label:float"].astype(float)
    interaction["user_age:token"] = interaction["user_age:token"].astype(int)
    interaction["user_gender:token"] = interaction["user_gender:token"].astype(int)
    interaction["video_source_id:token"] = interaction["video_source_id:token"].astype(
        int
    )
    interaction["video_duration:token"] = interaction["video_duration:token"].astype(
        int
    )

    interaction.info()

    interaction.iloc[:, [0, 1, 2]].to_csv("ready_data/data_for_DeepFM/data_for_DeepFM.inter", index=False)
    interaction.iloc[:, [0, 3, 4]].to_csv("ready_data/data_for_DeepFM/data_for_DeepFM.user", index=False)
    interaction.iloc[:, [1, 5, 6]].to_csv("ready_data/data_for_DeepFM/data_for_DeepFM.item", index=False)
