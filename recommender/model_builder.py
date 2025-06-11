from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

# Initialize a Porter Stemmer for reducing words to their root form
ps = PorterStemmer()

def stem(text: str) -> str:
    """
    Applies stemming to each word in the input text.

    Parameters:
    text (str): A space-separated string of words.

    Returns:
    str: A space-separated string of stemmed words.
    """
    return " ".join(ps.stem(word) for word in text.split())

def vectorize(df):
    """
    Transforms the 'tags' column of the DataFrame into a numerical feature matrix
    using the Bag-of-Words model with stemming and stop-word removal.

    Parameters:
    df (pd.DataFrame): DataFrame containing a 'tags' column with preprocessed text.

    Returns:
    np.ndarray: A 2D array where each row represents a movie and each column represents
                the frequency of a term (up to 5000 most common terms).
    """
    # Apply stemming to the 'tags' text
    df['tags'] = df['tags'].apply(stem)
    
    # Initialize CountVectorizer with a limit of 5000 features and built-in English stopword removal
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    
    # Fit the vectorizer to the 'tags' column and transform the text into a feature matrix
    vectors = vectorizer.fit_transform(df['tags']).toarray()
    
    return vectors
