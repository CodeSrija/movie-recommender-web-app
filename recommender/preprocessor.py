import ast

def parse(obj):
    """
    Parses a stringified list of dictionaries and extracts 'name' values,
    removing any spaces in the names.
    
    Parameters:
    obj (str): A string representation of a list of dictionaries.
    
    Returns:
    list: A list of names with spaces removed.
    """
    return [i['name'].replace(" ", "") for i in ast.literal_eval(obj)]

def get_top_cast(obj):
    """
    Extracts the top 3 cast member names from a stringified list of dictionaries.
    
    Parameters:
    obj (str): A string representation of a list of cast dictionaries.
    
    Returns:
    list: A list of the top 3 cast member names with spaces removed.
    """
    return [i['name'].replace(" ", "") for i in ast.literal_eval(obj)[:3]]

def get_director(obj):
    """
    Extracts the director's name from a stringified list of crew member dictionaries.
    
    Parameters:
    obj (str): A string representation of a list of crew dictionaries.
    
    Returns:
    list: A list containing the director's name with spaces removed (at most one item).
    """
    return [i['name'].replace(" ", "") for i in ast.literal_eval(obj) if i['job'] == 'Director'][:1]

def build_tags(df):
    """
    Transforms the movie dataset by extracting and combining relevant textual features
    into a single 'tags' column for each movie.

    Parameters:
    df (pd.DataFrame): DataFrame containing movie metadata.

    Returns:
    pd.DataFrame: A simplified DataFrame with 'title' and processed 'tags' columns.
    """
    # Apply parsing functions to extract structured data from stringified JSON columns
    df['genres'] = df['genres'].apply(parse)
    df['keywords'] = df['keywords'].apply(parse)
    df['cast'] = df['cast'].apply(get_top_cast)
    df['crew'] = df['crew'].apply(get_director)
    
    # Split overview into a list of words
    df['overview'] = df['overview'].apply(lambda x: x.split())

    # Combine overview, genres, keywords, cast, and crew into a single list of tags
    df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
    
    # Convert list of tags into a lowercase space-separated string
    df['tags'] = df['tags'].apply(lambda x: " ".join(x).lower())
    
    # Return a simplified DataFrame with only the title and tags columns
    return df[['title', 'tags']]
