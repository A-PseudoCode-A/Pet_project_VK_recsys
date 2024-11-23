# Данные находятся по ссылке: https://ods.ai/competitions/aivkchallenge/dataset

# Столбцы:

#     user_id - уникальный идентификатор пользователя;
#     item_id - уникальный идентификатор клипа;
#     timespent - время, которое пользователь провел на клипе;
#     like - лайкнул ли пользователь клип;
#     dislike - дизлайкнул ли пользователь клип;
#     share - поделился ли пользователь клипом;
#     bookmarks - поместил ли пользователь клип в закладки.


# Файл users_meta содержит данные о пользователе:

#     user_id - уникальный идентификатор пользователя;
#     gender - пол пользователя;
#     age - возраст пользователя.


# Файл items_meta содержит информацию о клипе:

#     item_id - уникальный идентификатор клипа;
#     source_id - уникальный идентификатор автора клипа;
#     duration - длительность клипа в секундах;
#     embeddings - нейросетевые эмбеддинги содержимого клипа (видеоряд, звук и тд.).


# Файл test_pairs содержит пары юзеров/клипов, для которых нужно сделать предсказание.
# Пары собраны за седьмую неделю (сразу после тренировочных).

import pandas as pd


def train_data_for_preprocessing():
    # Предварительный просмотр данных
    train_data = pd.read_parquet("train_interactions.parquet")
    users_data = pd.read_parquet("users_meta.parquet")
    items_data = pd.read_parquet("items_meta.parquet")

    # Выбираем только нужные столбцы
    # Объединяем по столбцу item_id
    # Используем left join, чтобы сохранить все строки из train_interactions
    train_data = train_data.merge(
        items_data[["item_id", "source_id", "duration"]],
        on="item_id",
        how="left",
    )

    train_data = train_data.merge(
        users_data[["user_id", "gender", "age"]],
        on="user_id",
        how="left",
    )

    # Добавление переменной, которая будет содержать то, сколько секунд user не досмотрел
    train_data["not_full_watched"] = train_data["duration"] > train_data["timespent"]

    # Приодим все булевые значения к числовым
    train_data = train_data.astype(
        {col: int for col in train_data.select_dtypes("bool").columns}
    )

    # Делаем удобные названия для столбцов
    train_data.rename(
        columns={
            "source_id": "video_source_id",
            "duration": "video_duration",
            "gender": "user_gender",
            "age": "user_age",
        },
        inplace=True,
    )

    # Возвращаем готовые данные
    return train_data.to_pickle("../ready_data/01_train_data_for_preprocessing.pkl")


# Вызов функции
train_data_for_preprocessing()

