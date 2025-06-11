"""
Movie Recommender - Content-Based Filtering
-------------------------------------------
This package implements a content-based movie recommendation system using
metadata from the TMDB 5000 dataset. It includes modular components for:

- Data loading and merging
- Feature extraction (genres, cast, crew, keywords, overview)
- Tag construction and preprocessing
- Vectorization with stemming
- Cosine similarity-based recommendation

The structure and methodology are inspired by educational content.

Original Source of Inspiration:
- YouTube Tutorial by Nitish Singh (CampusX): "Build Movie Recommendation System"
  https://youtu.be/1xtrIEwY_zY

Modules:
- data_loader:     Loads and merges TMDB datasets
- preprocessor:    Extracts structured metadata and builds tag fields
- model_builder:   Applies vectorization and text normalization
- engine:          Computes cosine similarity and retrieves top matches
"""

__version__ = "1.0.0"
__author__ = "Duruseti Srija"
__inspired_by__ = "CampusX (YouTube)"