import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def graph_settings(size_x=12, size_y=8, dpi=300, style="whitegrid", pallete="viridis"):
    """Function for setting up graphs.

    Args:
        size_x (int, optional): The size of the graphs by x. Defaults to 12.
        size_y (int, optional): The magnitude of the graphs by y. Defaults to 8.
        dpi (int, optional): Graph resolution. Defaults to 300.
        style (str, optional): Graph style. Defaults to "whitegrid".
        pallete (str, optional): Colors in graphs. Defaults to "viridis".
    """
    plt.figure(figsize=(size_x, size_y), dpi=dpi)
    sns.set_style(style=style)
    sns.set_palette(palette=pallete)


def data_vizualization(input_file='01_train_data_for_preprocessing"', sample_len=None):
    """A function that visualizes certain aspects of a dataset using the seaborn library

    Args:
        input_file (str, optional): The file that stores the dataset after the train_data_for_preprocessing function. Defaults to '01_train_data_for_preprocessing"'.
        sample_len (int, optional): Number of elements in the resulting dataset. Defaults to None, meaning all data will be used.
    """
    # Reading data
    train_data = pd.read_pickle(
        f"../03_feature_eng_and_ready_data/ready_data/{input_file}.pkl"
    )

    # Checking data for empty values
    missing_values = train_data.isnull().sum()
    print(missing_values)

    # Checking for a number in the sample_len
    if sample_len not in None:
        chunk_train_data = train_data.sample(n=sample_len, random_state=42)

    #! Age distribution
    graph_settings()
    sns.countplot(data=chunk_train_data, x="user_age")
    plt.xticks(rotation=45, ha="right")
    plt.savefig("age.jpg")

    #! Distribution of male and female sexes
    graph_settings()
    sns.countplot(data=chunk_train_data, x="user_gender")
    plt.savefig("gender.jpg")

    #! Distribution of likes and dislikes
    graph_settings()
    fig, axes = plt.subplots(1, 2)
    sns.countplot(data=chunk_train_data, x="like", ax=axes[0])
    axes[0].set_title("like")
    sns.countplot(data=chunk_train_data, x="dislike", ax=axes[1])
    axes[1].set_title("dislike")
    plt.savefig("like_dislike.jpg")

    #! Heat graph
    corr_matrix = chunk_train_data.corr()
    graph_settings()
    sns.heatmap(data=corr_matrix, annot=True)
    plt.savefig("heat_chart.jpg")

    #! Box-plots
    sns.boxplot(data=train_data, x="timespent")
    plt.savefig("heat_chart.jpg")

    #! Analysis of the most popular video authors
    print(
        train_data.groupby("video_source_id")
        .count()
        .item_id.sort_values(ascending=False)
        .head(10)
    )


# Calling functions
data_vizualization(sample_len=100_000)
