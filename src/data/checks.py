import pandas as pd

def basic_checks(df: pd.DataFrame) -> None:
    """This is a Data Validation Function

    Args:
        df (pd.DataFrame): Takes the Pandas DataFrame
    """
    assert df.isnull().sum().sum() == 0, "Unexpected missing values"
    assert "Class" in df.columns, "Target column missing"
