import pandas as pd

#! The data is located at the link: https://ods.ai/competitions/aivkchallenge/dataset

# Columns:

# user_id - unique user identifier;
# item_id - unique clip identifier;
# timespent - time the user spent on the clip;
# like - whether the user liked the clip;
# dislike - whether the user disliked the clip;
# share - whether the user shared the clip;
# bookmarks - whether the user added the clip to bookmarks.

# The users_meta file contains user data:

# user_id - unique user identifier;
# gender - user gender;
# age - user age.

# The items_meta file contains clip information:

# item_id - unique clip identifier;
# source_id - unique clip author identifier;
# duration - clip duration in seconds;
# embeddings - neural network embeddings of clip content (video sequence, sound, etc.).

# The test_pairs file contains user/clip pairs for which a prediction must be made.
# Pairs were collected in the seventh week (immediately after training).


def train_data_for_preprocessing(file_name = "01_train_data_for_preprocessing"):
    """Function that prepares a dataset for training

    Args:
        file_name (str): the file name you want to get

    Returns:
        file_name.pkl: dataset in pickle format
    """

    # Data Preview
    inter_data = pd.read_parquet("raw_data/train_interactions.parquet")
    users_data = pd.read_parquet("raw_data/users_meta.parquet")
    items_data = pd.read_parquet("raw_data/items_meta.parquet")

    # Selecting the required columns
    # Joining by the item_id column
    inter_data = inter_data.merge(
        items_data,
        on="item_id",
        how="left",
    )

    inter_data = inter_data.merge(
        users_data,
        on="user_id",
        how="left",
    )

    # Add a variable that will contain how many seconds the user did not watch
    inter_data["not_full_watched"] = inter_data["duration"] > inter_data["timespent"]

    # Convert all boolean values ​​to numeric
    inter_data = inter_data.astype(
        {col: int for col in inter_data.select_dtypes("bool").columns}
    )

    # Creating standard names for columns
    inter_data.rename(
        columns={
            "source_id": "video_source_id",
            "duration": "video_duration",
            "gender": "user_gender",
            "age": "user_age",
        },
        inplace=True,
    )

    # Create a file for the final dataset
    return inter_data.to_pickle(f"../02_feature_eng_and_ready_data/ready_data/{file_name}.pkl")


# Calling a function
train_data_for_preprocessing()
