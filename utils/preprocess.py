from .amenity_matching import AMENITY_PATTERNS, AMENITY_CATEGORIES
from .amenity_matching import append_amenity_dummies, aggregate_amenity_categories
import numpy as np
import pandas as pd


def keep_only_relevant_features(df, vars_to_involve):
    df = df.copy()
    df = df[vars_to_involve]
    return df


def calculate_euclidean_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Euclidean distance between two points given their latitude and longitude.

    Parameters:
    lat1, lon1: Latitude and longitude of the first point.
    lat2, lon2: Latitude and longitude of the second point.

    Returns:
    float: Euclidean distance between the two points.
    """
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


# First, we format those variables, that are relevant to missing data. Then we format all variables that are relevant to the model.
def missing_data_preprocess(df):
    # 1. host_location -> San Francisco, not San Francisco, other
    # df["host_location"] = df["host_location"].apply(
    #     lambda x: "San Francisco" if "San Francisco" in str(x) else "Other"
    # )
    # df["host_is_local"] = df["host_location"].str.contains("San Francisco").astype(int)

    # 2. host_response_rate -> formatted in percentages as string, back to decimals. Also needs imputing
    df = df.copy()
    df["host_response_rate"] = (
        df["host_response_rate"].str.rstrip("%").astype("float") / 100
    )
    df["host_response_rate"].fillna(df["host_response_rate"].median(), inplace=True)

    # 3. host_acceptance_rate -> formatted in percentages as string, back to decimals. Also needs imputing
    df["host_acceptance_rate"] = (
        df["host_acceptance_rate"].str.rstrip("%").astype("float") / 100
    )
    df["host_acceptance_rate"].fillna(df["host_acceptance_rate"].median(), inplace=True)

    # Impute numerical columns with median
    numerical_columns = [
        "review_scores_location",
        "review_scores_value",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_cleanliness",
        "review_scores_accuracy",
        "review_scores_rating",
        "reviews_per_month",
        # "beds", inpute instead with accommmodates
        "bathrooms",
        "host_response_rate",
        "host_acceptance_rate",
        "bedrooms",
    ]

    for column in numerical_columns:
        df[column].fillna(df[column].median(), inplace=True)

    # Impute beds with accommodates
    df["beds"].fillna(df["accommodates"], inplace=True)

    # Impute categorical columns with specified methods
    df["host_response_time"].fillna(
        df["host_response_time"].mode()[0], inplace=True
    )  # filled with the mode of the column
    df["host_neighbourhood"].fillna(
        df["host_neighbourhood"].mode()[0], inplace=True
    )  # filled with the mode of the column
    df["has_availability"] = (
        df["has_availability"].fillna("f").apply(lambda x: x == "t").astype(int)
    )
    df["host_is_superhost"] = (
        df["host_is_superhost"].fillna("f").apply(lambda x: x == "t").astype(int)
    )
    df["host_identity_verified"] = (
        df["host_identity_verified"].fillna("f").apply(lambda x: x == "t").astype(int)
    )
    df["host_has_profile_pic"] = (
        df["host_has_profile_pic"].fillna("f").apply(lambda x: x == "t").astype(int)
    )
    df["instant_bookable"] = (
        df["instant_bookable"].fillna("f").apply(lambda x: x == "t").astype(int)
    )

    # dropping all missing prices
    df = df.dropna(subset=["price"])
    df.reset_index(drop=True, inplace=True)

    return df


# Here, we define the entire preprocessing pipeline that we used above for later use.
def preprocess_data(df, vars_to_involve, reference_lat, reference_lon):
    # Append dummy columns for amenities
    df = append_amenity_dummies(df, "amenities", AMENITY_PATTERNS)

    # Aggregate amenity categories
    df = aggregate_amenity_categories(df, AMENITY_CATEGORIES)

    # Calculate the distance to the center
    df["distance_to_center"] = calculate_euclidean_distance(
        df["latitude"], df["longitude"], reference_lat, reference_lon
    )

    # Preprocess missing data
    df = missing_data_preprocess(df)

    # Keep only relevant features
    df = keep_only_relevant_features(df, vars_to_involve)

    # # drop room types of hotel rooms and shared rooms, because they are too little in numbers
    # df = df[df["room_type"] != "Hotel room"]
    # df = df[df["room_type"] != "Shared room"]

    # Apply the function to the dataframe
    df = format_datetime_and_categorical(df)
    # this can possibly be different for each subset, so preprocessing is not including this step implicitly.
    df["price"] = df["price"].replace("[\$,€]", "", regex=True).astype(float)

    # Calculate the 99th percentile value for the price - to reduce OLS instability
    price_cutoff = df["price"].quantile(0.99)

    # Filter the dataframe to include only rows with price less than or equal to the 99th percentile value
    df = df[df["price"] <= price_cutoff]

    return df


def format_datetime_and_categorical(df):
    """
    Format datetime and categorical variables in the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: The dataframe with formatted datetime and categorical variables.
    """
    categorical_columns = ["host_verifications", "room_type"]

    # Convert 'host_since' to datetime
    df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce")
    df["host_since"] = df["host_since"].max() - df["host_since"]
    df["host_since"] = df["host_since"].dt.days

    # Convert specified columns to categorical
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    return df
