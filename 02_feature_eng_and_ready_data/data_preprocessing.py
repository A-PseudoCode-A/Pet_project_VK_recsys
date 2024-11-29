import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def ready_train_data(file_name):
    # Считывание подготовленных данных
    file_name = '01_inter_data_for_preprocessing'
    train_data = pd.read_pickle(f"ready_data/{file_name}.pkl")

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
ready_train_data(file_name='01_inter_data_for_preprocessing')
