import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
pd.set_option('future.no_silent_downcasting', True)
def filling_HomePlanet(df):
    """
    Fill missing values in the 'HomePlanet' column with the mode (most frequent value).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a 'HomePlanet' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing 'HomePlanet' values filled.
    """
    mode = df['HomePlanet'].value_counts().index[0]
    df['HomePlanet'] = df['HomePlanet'].fillna(mode)
    return df

def filling_CryoSleep(df):
    """
    Fill missing values in the 'CryoSleep' column with False.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a 'CryoSleep' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing 'CryoSleep' values filled as False.
    """
    df["CryoSleep"] = df['CryoSleep'].fillna(False)
    return df
def filling_Cabin(df):
    df['Deck'] = df['Deck'].fillna('F')
    mode = df[df.Deck=='F']['Side'].value_counts().index[0]
    df['Side'] =mode
    df['Num'] = df['Num'].astype(float)
    df['Num'] = df['Num'].fillna(1796/2)
    return df

def filling_Destination(df):
    mode = df['Destination'].value_counts().index[0]
    df['Destination'] = df['Destination'].fillna(mode)
    return df

def filling_VIP(df):
    df['VIP'] = df['VIP'].fillna(False)
    return df

def filling_Name(df):
    df['Name'] = df['Name'].fillna('None')
    return df

def filling_categorical(df):
    df = filling_Cabin(df)
    df = filling_CryoSleep(df)
    df = filling_HomePlanet(df)
    df = filling_Destination(df)
    df = filling_Name(df)
    df = filling_VIP(df)
    return df

def filling_Age(df):
    """
    Fill missing values in the 'Age' column with the median age.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing an 'Age' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing 'Age' values replaced by the median.
    """
    median = df['Age'].median()
    df['Age'] = df['Age'].fillna(median)
    return df

def filling_luxury(df):
    """
    Fill missing values in luxury feature columns with 0.0.

    The following columns are considered luxury features:
    ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'].

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing luxury feature columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing luxury feature values replaced by 0.0.
    """
    luxury_features = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    df[luxury_features] = df[luxury_features].fillna(0.0)
    return df

def filling_numerical(df):
    df = filling_Age(df)
    df = filling_luxury(df)
    return df

def filling_missing(df):
    df = filling_categorical(df)
    df = filling_numerical(df)
    return df

def process_data(input_path=None, output_path=None):
    """
        Process raw Titanic dataset: clean, engineer features, and save to processed folder.

        Steps:
        1. Read raw CSV file.
        2. Split 'Cabin' into 'Deck', 'Num', and 'Side'.
        3. Fill missing values.
        4. Create 'Luxury' feature from luxury expenditure.
        5. Drop irrelevant columns ('Name', 'PassengerId').
        6. Label encode selected categorical variables.
        7. Apply one-hot encoding to categorical features.
        8. Save processed data to disk.

        Parameters
        ----------
        input_path : str, optional
            Path to the raw input CSV file. Default is "../data/raw/train.csv".
        output_path : str, optional
            Path to save the processed CSV file. Default is "../data/processed/train_processed.csv".

        Returns
        -------
        pd.DataFrame
            The cleaned and feature-engineered dataframe.
        """
    if input_path is None:
        input_path = os.path.join(BASE_DIR, "data", "raw", "train.csv")
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")
    df = pd.read_csv(input_path)

    if "Cabin" in df.columns:
        cabin = df['Cabin'].copy()
        cabin_info = cabin.str.split("/", expand=True)
        cabin_info.columns = ['Deck','Num','Side']
        df = pd.concat([df, cabin_info], axis=1)
        df.drop("Cabin", axis=1, inplace=True)

    df = filling_missing(df)

    df['Luxury'] = df['VRDeck'] + df['ShoppingMall'] + df['Spa'] + df['FoodCourt'] + df['RoomService']
    df['Luxury'] = pd.qcut(df['Luxury'], 6, duplicates='drop')
    for col in ['Name','PassengerId']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df_numeric = df.select_dtypes(include=[np.number])
    for feature in df_numeric.columns:
        if df_numeric[feature].max() > 10 * df_numeric[feature].median() and df_numeric[feature].max() > 10:
            df[feature] = np.where(df[feature] < df[feature].quantile(0.95), df[feature],
                                         df[feature].quantile(0.95))

    df_numeric = df.select_dtypes(include=[np.number])
    for feature in df_numeric.columns:
        if df_numeric[feature].nunique() > 50:
            if df_numeric[feature].min() == 0:
                df[feature] = np.log(df[feature] + 1)
            else:
                df[feature] = np.log(df[feature])

    le = LabelEncoder()
    for col in ['CryoSleep','VIP','Transported']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, drop_first=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

    return df

if __name__ == "__main__":
    process_data()
    process_data(input_path=os.path.join(BASE_DIR, "data", "raw", "test.csv"),output_path=os.path.join(BASE_DIR, "data", "processed", "test_processed.csv"))