import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def graph_settings(size_x=12, size_y=8, dpi=300, style="whitegrid", pallete="viridis"):
    plt.figure(figsize=(size_x, size_y), dpi=dpi)

    # Устанавка стиля
    sns.set_style(style=style)

    # Устанавливка цветовой палитры
    sns.set_palette(palette=pallete)


def vizualization(sample_len=100_000):
    # Считывание данных
    train_data = pd.read_pickle(
        "../01_data/ready_data/01_train_data_for_preprocessing.pkl"
    )

    # Проверка данных на пустые значения
    missing_values = train_data.isnull().sum()
    print(missing_values)

    # Создание небольшого датасета
    chunk_train_data = train_data.sample(n=sample_len, random_state=42)

    #! Распределение возраста
    # Больше всего людей в диапазоне от 25-35 лет
    graph_settings()
    sns.countplot(data=chunk_train_data, x="user_age")
    plt.xticks(rotation=45, ha="right")
    plt.savefig("age.jpg")

    #! Распределение полов
    graph_settings()
    sns.countplot(data=chunk_train_data, x="user_gender")
    plt.savefig("gender.jpg")
    # Пола 2 в 1.5 раз больше, чем 1

    #! Распределение like и dislike
    graph_settings()
    fig, axes = plt.subplots(1, 2)
    sns.countplot(data=chunk_train_data, x="like", ax=axes[0])
    axes[0].set_title("like")
    sns.countplot(data=chunk_train_data, x="dislike", ax=axes[1])
    axes[1].set_title("dislike")
    plt.savefig("like_dislike.jpg")

    #! Тепловой график
    corr_matrix = chunk_train_data.corr()

    graph_settings()
    sns.heatmap(data=corr_matrix, annot=True)
    plt.savefig("heat_chart.jpg")

    #! Box-plots
    sns.boxplot(data=train_data, x="timespent")
    plt.savefig("heat_chart.jpg")
    # Время просмотра очень сильно разбросано

    #! Анализируем самых популярных авторов видео
    print(
        train_data.groupby("video_source_id")
        .count()
        .item_id.sort_values(ascending=False)
        .head(10)
    )


vizualization()
