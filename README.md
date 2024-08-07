# Movie Recommender System using PySpark

This project implements a sophisticated movie recommender system using various collaborative filtering techniques and machine learning algorithms. The system is built using Apache Spark and PySpark, leveraging the power of distributed computing for handling large-scale movie rating data.

## Features

1. **Alternating Least Squares (ALS) Model**
   - Implements collaborative filtering using the ALS algorithm
   - Utilizes cross-validation for hyperparameter tuning
   - Evaluates performance using RMSE, MSE, and MAP metrics

2. **Item-Item Collaborative Filtering**
   - Computes movie similarities using Pearson correlation
   - Generates recommendations based on item similarities

3. **Hybrid Recommender System**
   - Combines ALS and Item-Item CF predictions
   - Incorporates a supervised learning component (Random Forest Regressor)
   - Optimizes weights for different model components

4. **Evaluation Metrics**
   - Root Mean Square Error (RMSE)
   - Mean Square Error (MSE)
   - Mean Average Precision (MAP)

## Dataset

The project uses the MovieLens 25M dataset, which includes:

- User ratings for movies
- Movie metadata (title, genres)

## Implementation Details

- **Data Preparation**: The dataset is split into training (80%) and test (20%) sets.
- **ALS Model**: Implemented using Spark's MLlib with hyperparameter tuning.
- **Item-Item CF**: Custom implementation of similarity computation and prediction generation.
- **Hybrid Model**: 
  - First version combines ALS and Item-Item CF
  - Second version adds a Random Forest Regressor trained on movie genres
- **Evaluation**: Comprehensive evaluation using various metrics, with a focus on RMSE for the hybrid model.

## Results

The project demonstrates the effectiveness of hybrid approaches in recommendation systems:

- ALS Model RMSE: 0.845
- Hybrid Model (ALS + Item-Item CF) RMSE: 0.16
- Best Hybrid Model (ALS + Item-Item CF + Random Forest) RMSE: 0.43

## Technologies Used

- Apache Spark
- PySpark
- PySpark MLlib
- Python

## How to Run

1. Ensure you have Apache Spark and PySpark installed.
2. Download the MovieLens 25M dataset.
3. Update the file paths in the `movie_recommender_system.py` script to point to your dataset location.
4. Run the script using spark-submit or in a PySpark environment:

## Future Improvements

- Incorporate more features like user demographics and movie metadata
- Experiment with deep learning models for recommendation
- Implement real-time recommendation updates

This project showcases the power of combining multiple recommendation techniques to create a robust and accurate movie recommender system.
