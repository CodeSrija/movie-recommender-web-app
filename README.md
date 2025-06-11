# 🎬 Movie Recommender – Content-Based Filtering

A modular **content-based movie recommendation system** built using metadata from the [TMDB 5000 dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). It recommends movies based on **overviews, genres, cast, crew**, and **keywords**, using **cosine similarity** on stemmed textual features. Built with **Python**, **Pandas**, **Scikit-learn**, and **NLTK**, and deployed via **Streamlit**. Access the web app at: [http://localhost:8501](http://localhost:8501).


## 📂 Project Structure

```bash
movie-recommender-web-app/
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── recommender/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── model_builder.py
│   └── engine.py
├── app.py
├── requirements.txt
└── README.md
```

## 🚀 Features

* Content-based recommendation using metadata
* Preprocessing pipeline with NLTK stemming
* Cosine similarity based on combined movie tags
* Streamlit UI with dropdown movie selection
* Easily extendable (e.g., genre filters, posters, API integration)

## 🛠 Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/movie-recommender-app.git
cd movie-recommender-web-app
```

### Step 2: Set Up a Virtual Environment

**2.1. Using Conda (Recommended)**

```bash
conda create -n mrec python=3.10 -y
conda activate mrec
```

**2.2. Using `venv`**

* **Windows**:

  ```bash
   python -m venv mrec
   .\mrec\Scripts\activate
  ```

* **macOS/Linux**:

  ```bash
   python3 -m venv mrec
   source mrec/bin/activate
  ```

### Step 3: Install Dependencies

* Using `pip`:

  ```bash
  pip install -r requirements.txt
  ```

* Or using `conda` (for Anaconda/Minconda):

  ```bash
  conda install --file requirements.txt
  ```

### Step 4: Run the App Locally

```bash
streamlit run app.py
```

## 🧠 How It Works

1. **Data Ingestion**
   Merges `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` on the movie title.
2. **Feature Engineering**
   Combines `overview`, `genres`, `keywords`, `cast`, and `crew` into a unified `tags` column.
3. **Text Normalization**
   Tokenizes and stems the text using `PorterStemmer`.
4. **Vectorization**
   Uses `CountVectorizer` to convert text into feature vectors.
5. **Similarity Computation**
   Applies cosine similarity to find similar movies.
6. **Recommendation Engine**
   Outputs top-N movies most similar in content.

### 📌 Example

> Select **"The Dark Knight Rises"** — get recommendations based on plot, actors, and keywords.

## 📈 Possible Extensions

* Integrate TMDB API for posters and descriptions
* Add genre or year filters
* Switch to TF-IDF or transformer-based embeddings
* Dockerize for containerized deployment
* Add benchmark comparisons with collaborative or hybrid models

## 🎓 Attribution

Inspired by:  
*YouTube Tutorial by Nitish Singh (CampusX): [Build Movie Recommendation System](https://youtu.be/1xtrIEwY_zY)*

## 👤 Author

**Duruseti Srija**  
© 2025. All rights reserved.

## 📜 License

This project is licensed under the **MIT License**.
