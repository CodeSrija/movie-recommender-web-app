from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_matrix(vectors):
    """
    Computes the cosine similarity matrix from a feature vector array.

    Parameters:
    vectors (np.ndarray): 2D array where each row is a vector representing a movie.

    Returns:
    np.ndarray: A square matrix where each element [i][j] is the cosine similarity between movie i and movie j.
    """
    return cosine_similarity(vectors)

def recommend(title: str, data, similarity_matrix, top_n=5):
    """
    Recommends movies similar to the given title based on cosine similarity.

    Parameters:
    title (str): The title of the movie to find recommendations for.
    data (pd.DataFrame): DataFrame containing at least a 'title' column.
    similarity_matrix (np.ndarray): Precomputed cosine similarity matrix.
    top_n (int): Number of similar movies to return (default is 5).

    Returns:
    list of tuples: Each tuple contains the recommended movie title and its similarity score.

    Raises:
    ValueError: If the input movie title is not found in the dataset.
    """
    # Normalize input title for case-insensitive matching
    title = title.lower()
    data_titles = data['title'].str.lower()
    
    # Check if the title exists in the dataset
    if title not in data_titles.values:
        raise ValueError("Movie title not found in dataset.")
    
    # Find index of the movie in the dataset
    idx = data_titles[data_titles == title].index[0]
    
    # Get similarity scores for the given movie
    distances = list(enumerate(similarity_matrix[idx]))
    
    # Sort by similarity score in descending order and exclude the input movie itself
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    # Return a list of tuples with recommended movie titles and their similarity scores
    return [(data.iloc[i[0]].title, i[1]) for i in distances]
