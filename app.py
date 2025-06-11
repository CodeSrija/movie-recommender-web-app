import streamlit as st
from recommender.data_loader import load_movie_data
from recommender.preprocessor import build_tags
from recommender.model_builder import vectorize
from recommender.engine import compute_similarity_matrix, recommend

# Set page title and icon for the web app
st.set_page_config(page_title="Movie Recommender", page_icon="🎥")

@st.cache_resource
def setup():
    """
    Loads, processes, and prepares the movie dataset for recommendations.
    Caches the result to avoid reloading and recomputing on every run.

    Returns:
    df (pd.DataFrame): Processed movie DataFrame.
    similarity_matrix (np.ndarray): Cosine similarity matrix for all movies.
    """
    # Load raw data
    df = load_movie_data('data/tmdb_5000_movies.csv', 'data/tmdb_5000_credits.csv')
    
    # Process data to extract and combine relevant textual features
    df = build_tags(df)
    
    # Vectorize text data and compute similarity matrix
    vectors = vectorize(df)
    similarity_matrix = compute_similarity_matrix(vectors)
    
    return df, similarity_matrix

# Initialize data and similarity matrix
df, sim = setup()

# Set app title
st.title("🎬 Content-Based Movie Recommender")

# Movie selection dropdown
selected_movie = st.selectbox("Choose a movie", sorted(df['title'].values))

# When user clicks the recommend button
if st.button("Recommend"):
    try:
        # Generate recommendations
        results = recommend(selected_movie, df, sim)
        
        # Display top recommended movies
        st.subheader("Top Recommendations:")
        for idx, (title, _) in enumerate(results, start=1):
            st.write(f"{idx}. {title}")
    
    # Handle case where selected title is invalid (shouldn't occur with dropdown)
    except ValueError:
        st.error("Movie not found. Please choose from dropdown only.")
