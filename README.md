# Movie Recommendation System

A collaborative filtering-based movie recommendation system using K-Nearest Neighbors (KNN) algorithm with cosine similarity.

## Overview

This project implements a movie recommendation system that:
- Downloads the movies dataset from Kaggle using KaggleHub
- Builds a comprehensive movies database by joining links.csv with movies_metadata.csv 
- Uses ratings_small.csv for faster processing
- Creates recommendations based on user-item collaborative filtering
- Employs cosine similarity to find movies with similar rating patterns

## Features

- **Automated Dataset Download**: Uses KaggleHub to automatically download the rounakbanik/the-movies-dataset
- **Data Integration**: Combines multiple CSV files to create a rich movie database
- **Efficient Processing**: Uses ratings_small.csv for improved performance
- **Sparse Matrix Optimization**: Leverages scipy's csr_matrix for memory-efficient computation
- **Interactive Recommendations**: Provides a simple function to get movie recommendations
- **Fallback Mode**: Includes sample data generation when dataset download fails

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Internet connection for dataset download

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SatyamSaxena1/Recommender-works-.git
   cd Recommender-works-
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the installation** (optional but recommended):
   ```bash
   python test_notebook.py
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

5. **Open and run the notebook**:
   - Open `This one works.ipynb` in Jupyter
   - Run all cells to download the dataset and build the recommendation system

### Required Packages

The system requires the following Python packages (automatically installed via requirements.txt):
- `pandas >= 1.3.0` - Data manipulation and analysis
- `numpy >= 1.21.0` - Numerical computing
- `scipy >= 1.7.0` - Scientific computing (sparse matrices)
- `scikit-learn >= 1.0.0` - Machine learning (KNN algorithm)
- `matplotlib >= 3.4.0` - Plotting and visualization
- `seaborn >= 0.11.0` - Statistical data visualization
- `kagglehub >= 0.3.0` - Kaggle dataset downloads
- `jupyter >= 1.0.0` - Jupyter notebook environment

## Usage

### Basic Usage

1. **Run the notebook**: Execute all cells in `This one works.ipynb`

2. **Get recommendations**: Use the recommendation function with a movie title:
   ```python
   recommendations = get_movie_recommendation('toy story')
   print(recommendations)
   ```

### Dataset Information

The system uses the **rounakbanik/the-movies-dataset** from Kaggle, which includes:
- **movies_metadata.csv**: Comprehensive movie information
- **links.csv**: Links between different movie databases
- **ratings_small.csv**: User ratings (subset for performance)

The movies table is built by joining:
- `links.csv` (movieId, tmdbId) with `movies_metadata.csv` (id, title, genres)
- This provides richer movie metadata while maintaining compatibility with ratings

### How It Works

1. **Data Loading**: Downloads and processes the Kaggle dataset
2. **Data Preparation**: Creates a user-item rating matrix
3. **Filtering**: Removes users and movies with insufficient data
4. **Model Training**: Fits KNN model with cosine similarity
5. **Recommendations**: Finds similar movies based on rating patterns

### Customization

You can modify the recommendation parameters:
- `n_neighbors`: Number of similar movies to consider (default: 20)
- `n_movies_to_recommend`: Number of recommendations to return (default: 10)
- Similarity metric: Currently uses cosine similarity
- Filtering thresholds: Minimum ratings per user/movie

## Dataset Structure

After processing, the data includes:
- **Movies**: movieId, title, genres
- **Ratings**: userId, movieId, rating, timestamp
- **User-Item Matrix**: Sparse matrix for efficient similarity computation

## Project Structure

```
Recommender-works-/
├── This one works.ipynb    # Main Jupyter notebook with recommendation system
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── test_notebook.py       # Test script to validate functionality
├── .gitignore            # Git ignore file
└── .git/                 # Git repository data
```

## Troubleshooting

### Quick Testing
Run the test script to check if everything is working:
```bash
python test_notebook.py
```

This will validate:
- All dependencies are installed correctly
- Data generation and processing works
- Recommendation system functions properly

### Dataset Download Issues
- Ensure you have internet connectivity
- The system includes fallback sample data if download fails
- Check Kaggle API credentials if needed

### Memory Issues
- The system uses ratings_small.csv for better performance
- Sparse matrices optimize memory usage
- Consider further filtering for very large datasets

### Performance Optimization
- Adjust user/movie filtering thresholds
- Use smaller neighbor counts for faster recommendations
- Consider using approximate nearest neighbor algorithms for very large datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset: rounakbanik/the-movies-dataset from Kaggle
- Built using scikit-learn's NearestNeighbors implementation
- Uses collaborative filtering techniques for recommendation generation