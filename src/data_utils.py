import pandas as pd
import os
import matplotlib.pyplot as plt

def get_target_name() -> str:
    """Return pre-defined name of target variable.

    :return str: target variable name
    """
    return "Outcome"

def explore_data(data: pd.DataFrame):
    """Print auto-generated summaries and plots for dataframe.

    :param pd.DataFrame data: dataset to summarize
    """
    symbol = "="
    print(f"{symbol*12} Data exploration {symbol*12}")
    print("\nColumn types:\n")
    print(data.info())
    print("\nStatistics:\n")
    print(data.describe())
    print("\nCorrelations:")
    print(data.corr())
    data.hist()
    plt.show()
    print(f"{symbol*42}")

def load_data() -> pd.DataFrame:
    """Loads the PIMA Indians diabetes dataset from data/

    :return pd.DataFrame: raw dataset
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = f"{dir_path}/../data/diabetes.csv"
    data = pd.read_csv(data_path)
    print(data.head())
    return data

def get_predictor_names(data: pd.DataFrame, target_name: str) -> list[str]:
    """Returns all other column names from data except target.

    :param pd.DataFrame data: dataframe
    :param str target_name: target feature name
    :return list[str]: names of predictors
    """
    return data.loc[:, data.columns != target_name].columns.values

def get_col_names_with_missing_values(sample: pd.DataFrame) -> list[str]:
    """Returns column names where the value is missing.

    :param pd.DataFrame sample: dataframe with one row
    :return list[str]: names of columns with missing values
    """
    return [col_name for col_name in sample.columns if pd.isnull(sample.loc[0, col_name])]

def get_test_input_1() -> dict:
    """Test input for which we expect classification outcome=0.

    :return dict: test input dict
    """
    return {"Pregnancies": 2, 
            "Glucose": 50, 
            "BloodPressure": 50, 
            "SkinThickness": 20, 
            "Insulin": 0, 
            "BMI": 23, 
            "DiabetesPedigreeFunction": 0.356,
            "Age": 34}

def get_test_input_2() -> dict:
    """Test input for which we expect classification outcome=1.

    :return dict: test input dict
    """
    return {"Pregnancies": 6, 
            "Glucose": 150, 
            "BloodPressure": 70, 
            "SkinThickness": 30, 
            "Insulin": 140, 
            "BMI": 29, 
            "DiabetesPedigreeFunction": 0.90,
            "Age": 50}