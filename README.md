# Movie Recommender System

This project implements a movie recommender system using K-Nearest Neighbors algorithm with collaborative filtering.

## Features

- Movie recommendation based on user ratings
- Uses cosine similarity to find similar movies
- Supports MovieLens dataset with automatic download via KaggleHub
- Interactive Jupyter notebook implementation

## Setup Instructions

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Recommender-works-
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up Kaggle credentials for automatic dataset download:
   - Create a Kaggle account and generate API token
   - Place `kaggle.json` in `~/.kaggle/` directory
   - Or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables

### Running the Project

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `This one works.ipynb`

3. Run all cells sequentially - the notebook will:
   - Automatically download MovieLens dataset using KaggleHub
   - Process the data and create a user-movie rating matrix
   - Train a K-Nearest Neighbors model
   - Provide movie recommendations

## Dataset

The project uses the MovieLens dataset, specifically:
- **Primary**: `ratings_small.csv` and `movies.csv` from MovieLens small dataset
- **Fallback**: Standard MovieLens `ratings.csv` and `movies.csv`

The dataset is automatically downloaded using KaggleHub. No manual dataset setup required.

## Usage

After running all notebook cells, use the recommendation function:

```python
# Get movie recommendations
recommendations = get_movie_recommendation('toy story')
print(recommendations)
```

## How it Works

1. **Data Loading**: Downloads and loads MovieLens movies and ratings data
2. **Data Processing**: Creates a user-movie rating matrix
3. **Filtering**: Removes users/movies with insufficient ratings
4. **Model Training**: Trains KNN model with cosine similarity
5. **Recommendations**: Finds similar movies based on user rating patterns

## Troubleshooting

- **Network Issues**: If KaggleHub fails to download, check internet connection and Kaggle credentials
- **Memory Issues**: Large datasets may require more RAM - consider using smaller dataset versions
- **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`