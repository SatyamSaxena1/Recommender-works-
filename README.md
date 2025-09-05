# Recommender-works-

This notebook builds a KNN-based item-item recommender on the MovieLens data. It now downloads the dataset automatically using KaggleHub, so you dont need to hardcode local paths.

## Setup

1. Install Python 3.9+ and pip.
2. Create a virtual environment (optional but recommended).
3. Install dependencies:

```
pip install -r requirements.txt
```

## Data download with KaggleHub

We use KaggleHub to pull `rounakbanik/the-movies-dataset`. On first run, you may be prompted to authenticate with Kaggle. See https://github.com/Kaggle/kagglehub for details.

The notebook picks available files automatically, preferring the small variants for quick runs:
- ratings_small.csv or ratings.csv
- movies.csv (MovieLens) or movies_metadata.csv + links.csv pair

## Run the notebook

Open `This one works.ipynb` and run all cells.

If you want to try a different dataset (e.g., `grouplens/movielens-20m-dataset`), set `DATASET_SLUG` in the first cell accordingly.

## Notes
- The recommender expects `ratings` with columns: userId, movieId, rating.
- For `movies_metadata.csv`, we map MovieLens movieId via `links.csv` to the TMDB metadata and extract the `title` and `genres`.
