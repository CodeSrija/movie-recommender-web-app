import pandas as pd

def load_movie_data(movies_path: str, credits_path: str) -> pd.DataFrame:
    """
    Loads and merges movie and credits data from CSV files.

    Parameters:
    movies_path (str): Path to the CSV file containing movie data.
    credits_path (str): Path to the CSV file containing credits data.

    Returns:
    pd.DataFrame: A cleaned DataFrame containing selected columns from the merged dataset,
                  with rows containing any missing values removed.
    """
    # Load movie data from the specified CSV file
    movies = pd.read_csv(movies_path)
    
    # Load credits data from the specified CSV file
    credits = pd.read_csv(credits_path)
    
    # Merge the two datasets on the 'title' column to combine related movie and credit information
    merged = movies.merge(credits, on='title')
    
    # Select only relevant columns and remove rows with missing values
    return merged[['title', 'genres', 'keywords', 'overview', 'cast', 'crew']].dropna()
