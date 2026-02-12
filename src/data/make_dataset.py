import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    """Loads the raw dataset from disk

    Args:
        path (str): Data Path to be loaded from

    Returns:
        pd.DataFrame: Returns the loaded data as a Panadas DataFrame
    """
    return pd.read_csv(path)

def save_processed_data(df: pd.DataFrame, path: str) -> None:
    """Saving the Dataset after Preprocessing

    Args:
        df (pd.DataFrame): The DataFrame to be Saved
        path (str): The Saving Path for the DataFrame
    """
    df.to_csv(path, index=False)

if __name__ == "__main__":
    raw_df = load_raw_data("../data/raw/creditcard.csv")
    save_processed_data(raw_df, "../data/processed/creditcard_processed.csv")