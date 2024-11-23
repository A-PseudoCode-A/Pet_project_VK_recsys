import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


def ready_train_data():
    # Считывание подготовленных данных
    train_data = pd.read_pickle(
        "../01_data/ready_data/01_train_data_for_preprocessing.pkl"
    )

    # Создание одного столбца с полом, а не двух, как было раньше.
    # С помощью OHE делается два столбца one-hot векторов, далее один столбец удаляется
    onehotencoder = OneHotEncoder(sparse_output=False)
    encoded_df = pd.DataFrame(onehotencoder.fit_transform(train_data[["user_gender"]]))
    encoded_df.rename(columns={0: "Gender"}, inplace=True)
    del encoded_df[1]
    encoded_df["Gender"].astype(int)

    # Присоединяем столбец с полом к основным данным, столбцы с полом до этого - удаляем
    train_data = pd.concat([train_data, encoded_df], axis=1)
    del train_data["user_gender"]
    train_data["Gender"].astype(int)

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
    train_data.to_pickle("../01_data/ready_data/02_ready_train_data.pkl")


# Вызов функции
ready_train_data()

#! В планах:
# TODO: StandartScaler, !pip install category_encoders from category_encoders.binary import BinaryEncoder, Балансировку классов
# Балансировка классов
# class_0 = train_data[train_data["target"] == 0]
# class_1 = train_data[train_data["target"] == 1]
# class_neg1 = train_data[train_data["target"] == -1]

# class_1_oversampled = resample(
#     class_1, replace=True, n_samples=len(class_0), random_state=42
# )
# class_neg1_oversampled = resample(
#     class_neg1, replace=True, n_samples=len(class_0), random_state=42
# )
# train_data = pd.concat([class_0, class_1_oversampled, class_neg1_oversampled])
