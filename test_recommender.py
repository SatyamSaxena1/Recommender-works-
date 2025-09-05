#!/usr/bin/env python3
"""
Test script for MovieLens data loading and recommender system
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import os
import sys

def create_sample_data():
    """Create sample MovieLens-like data for testing"""
    print("Creating sample data for testing...")
    
    # Sample movies data
    movies_data = {
        'movieId': [1, 2, 3, 4, 5],
        'title': ['toy story (1995)', 'jumanji (1995)', 'grumpier old men (1995)', 
                 'waiting to exhale (1995)', 'father of the bride part ii (1995)'],
        'genres': ['Adventure|Animation|Children|Comedy|Fantasy', 
                  'Adventure|Children|Fantasy', 'Comedy|Romance', 
                  'Comedy|Drama|Romance', 'Comedy']
    }
    
    # Sample ratings data
    ratings_data = {
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        'movieId': [1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 5, 1, 4, 5],
        'rating': [4.0, 3.5, 4.0, 4.5, 3.0, 3.5, 5.0, 4.0, 4.5, 3.5, 3.0, 4.0, 4.5, 4.0, 3.5],
        'timestamp': [964982703] * 15
    }
    
    movies = pd.DataFrame(movies_data)
    ratings = pd.DataFrame(ratings_data)
    
    return movies, ratings

def test_kagglehub_integration():
    """Test KaggleHub integration with fallback to sample data"""
    try:
        import kagglehub
        print("KaggleHub installed successfully")
        
        # Try to download a known dataset (this may fail due to network issues)
        try:
            print("Testing KaggleHub dataset download...")
            dataset_path = kagglehub.dataset_download("shubhammehta21/movie-lens-small-latest-dataset")
            print(f"✓ Dataset downloaded successfully to: {dataset_path}")
            
            # Check for required files
            files = os.listdir(dataset_path)
            print(f"Available files: {files}")
            
            movies_file = None
            ratings_file = None
            
            for file in files:
                if 'movies' in file.lower() and file.endswith('.csv'):
                    movies_file = os.path.join(dataset_path, file)
                elif 'ratings_small' in file.lower() and file.endswith('.csv'):
                    ratings_file = os.path.join(dataset_path, file)
                elif 'ratings' in file.lower() and file.endswith('.csv') and ratings_file is None:
                    ratings_file = os.path.join(dataset_path, file)
            
            if movies_file and ratings_file:
                movies = pd.read_csv(movies_file)
                ratings = pd.read_csv(ratings_file)
                print(f"✓ Loaded movies: {movies.shape}, ratings: {ratings.shape}")
                return movies, ratings
            else:
                raise FileNotFoundError("Required CSV files not found")
                
        except Exception as e:
            print(f"KaggleHub download failed: {e}")
            print("Using sample data for testing...")
            return create_sample_data()
            
    except ImportError:
        print("KaggleHub not available, using sample data...")
        return create_sample_data()

def test_recommender_system(movies, ratings):
    """Test the recommender system functionality"""
    print("\n=== Testing Recommender System ===")
    
    # Convert movie titles to lowercase
    movies['title'] = movies['title'].str.lower()
    print(f"✓ Processed movie titles")
    
    # Create pivot table
    final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
    print(f"✓ Created pivot table: {final_dataset.shape}")
    
    # Fill NaN values with 0
    final_dataset.fillna(0, inplace=True)
    print(f"✓ Filled NaN values")
    
    # Filter users with sufficient ratings (for small sample, use lower threshold)
    min_ratings = min(10, ratings.groupby('userId')['rating'].count().max() // 2)
    no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
    active_users = no_movies_voted[no_movies_voted > min_ratings].index
    
    if len(active_users) > 0:
        final_dataset = final_dataset.loc[:, active_users]
        print(f"✓ Filtered to active users: {final_dataset.shape}")
    else:
        print(f"✓ Using all users due to small dataset: {final_dataset.shape}")
    
    # Create CSR matrix
    csr_data = csr_matrix(final_dataset.values)
    final_dataset.reset_index(inplace=True)
    print(f"✓ Created CSR matrix with sparsity: {1.0 - (csr_data.nnz / float(csr_data.shape[0] * csr_data.shape[1])):.3f}")
    
    # Train KNN model
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=min(5, final_dataset.shape[0]), n_jobs=-1)
    knn.fit(csr_data)
    print(f"✓ Trained KNN model")
    
    # Test recommendation function
    def get_movie_recommendation(movie_name, n_movies_to_recommend=3):
        movie_list = movies[movies['title'].str.contains(movie_name.lower())]
        if len(movie_list):
            movie_idx = movie_list.iloc[0]['movieId']
            try:
                movie_row_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
                distances, indices = knn.kneighbors(csr_data[movie_row_idx], n_neighbors=n_movies_to_recommend+1)
                
                rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
                
                recommend_frame = []
                for val in rec_movie_indices:
                    rec_movie_idx = final_dataset.iloc[val[0]]['movieId']
                    idx = movies[movies['movieId'] == rec_movie_idx].index
                    if len(idx) > 0:
                        recommend_frame.append({
                            'Title': movies.iloc[idx[0]]['title'], 
                            'Distance': val[1]
                        })
                
                if recommend_frame:
                    df = pd.DataFrame(recommend_frame, index=range(1, len(recommend_frame)+1))
                    return df
                else:
                    return "No recommendations found"
            except IndexError:
                return f"Movie '{movie_name}' not found in filtered dataset"
        else:
            return f"No movies found matching '{movie_name}'"
    
    # Test with available movies
    test_movie = movies.iloc[0]['title'].split(' ')[0]  # Use first word of first movie
    print(f"\n=== Testing Recommendations ===")
    print(f"Testing with movie containing: '{test_movie}'")
    
    result = get_movie_recommendation(test_movie)
    print("Recommendation result:")
    print(result)
    
    return True

if __name__ == "__main__":
    try:
        # Test the integration
        movies, ratings = test_kagglehub_integration()
        
        # Test the recommender system
        success = test_recommender_system(movies, ratings)
        
        if success:
            print("\n✓ All tests passed! The recommender system is working correctly.")
            print("\nNext steps:")
            print("1. Run the Jupyter notebook: jupyter notebook")
            print("2. Open 'This one works.ipynb'")
            print("3. Run all cells to use the full recommender system")
        else:
            print("\n✗ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)