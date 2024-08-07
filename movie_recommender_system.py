# Install and import packages
"""## Import packages and initiate spark session"""

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.functions import count, expr, collect_list, col, sqrt, when, lit, rank, split, explode, sum as sql_sum
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator, RankingEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

spark = SparkSession.builder.appName("MovieRecommenderSystem").getOrCreate()

ratings_df = spark.read.csv("hdfs:///user/apathak2/input/ml-25m/ratings.csv", header=True, inferSchema=True).repartition(12)
ratings_df = ratings_df.cache()
ratings_df.repartition(12)

""" Create Train and Test sets"""

(train, test) = ratings_df.randomSplit([0.8, 0.2], seed=0)
train = train.cache()
test = test.cache()

""" ALS Model"""

als = ALS(seed=0, coldStartStrategy="drop", nonnegative=True)
als.setUserCol("userId")
als.setItemCol("movieId")
als.setRatingCol("rating")

param_grid = ParamGridBuilder()\
            .addGrid(als.rank, [5, 10])\
            .addGrid(als.maxIter, [6, 12])\
            .addGrid(als.regParam, [.17])\
            .build()

rmse_evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')

cv = CrossValidator(
    estimator=als,
    estimatorParamMaps=param_grid,
    evaluator=rmse_evaluator,
    numFolds=3
)

""" Train ALS model"""

print("Traning recommendation model...")
model = cv.fit(train)
print("Training finished successfully!")

best_model = model.bestModel

""" Generate predictions and print evaluation metrics"""

predictions = best_model.transform(test)
predictions.withColumn("prediction", expr("CASE WHEN prediction < 1 THEN 1 WHEN prediction > 5 THEN 5 ELSE prediction END"))

rmse = rmse_evaluator.evaluate(predictions)

mse_evaluator = RegressionEvaluator(metricName='mse', labelCol='rating', predictionCol='prediction')
mse = mse_evaluator.evaluate(predictions)

# Generate top N movie recommendations for each user
user_recommendations = best_model.recommendForAllUsers(10)

# Prepare predictions for RankingEvaluator
user_predicted_movies = user_recommendations.selectExpr("userId", "recommendations.movieId as predicted_movies")

# Prepare actual data for RankingEvaluator
# Group by userId and collect all movieIds in a list
user_actual_movies = test.groupBy("userId").agg(expr("collect_list(movieId) as actual_movies"))

# Transform each integer in predicted_movies to a double
user_predicted_movies = user_predicted_movies.select(
    col("userId"),
    expr("transform(predicted_movies, movie -> cast(movie as double)) as predicted_movies")
)
# Transform each integer in actual_movies to a double
user_actual_movies = user_actual_movies.select(
    col("userId"),
    expr("transform(actual_movies, movie -> cast(movie as double)) as actual_movies")
)

# Join the actual and predicted data
evaluator_data = user_actual_movies.join(user_predicted_movies, ["userId"]).select("userId", "actual_movies", "predicted_movies")

# Create and use the RankingEvaluator
ranking_evaluator = RankingEvaluator(
    metricName="meanAveragePrecision",
    labelCol="actual_movies",
    predictionCol="predicted_movies"
)

map = ranking_evaluator.evaluate(evaluator_data)

""" Print ALS metrics"""

print(f"RMSE = {rmse}")
print(f"MSE = {mse}")
print(f"MAP = {map}")
print("---Best Model---")
print(f" Rank: {best_model.rank}")
print(f" MaxIter: {best_model._java_obj.parent().getMaxIter()}")
print(f" RegParam: {best_model._java_obj.parent().getRegParam()}")


""" Item-Item CF"""

# Calculate mean rating for each movie
mean_ratings = ratings_df.groupBy("movieId").agg(F.avg("rating").alias("mean_rating"))

# Normalize ratings by subtracting mean rating
normalized_ratings_df = ratings_df.join(mean_ratings, "movieId")
normalized_ratings_df = normalized_ratings_df.withColumn("norm_rating", F.col("rating") - F.col("mean_rating"))

def compute_similarity(df):
    # Self-join to find pairs of movies rated by the same user
    joined_df = df.alias("df1").join(df.alias("df2"), "userId")
    joined_df = joined_df.filter("df1.movieId < df2.movieId")  # Remove duplicates

    # Compute the components for Pearson similarity
    joined_df = joined_df.groupBy("df1.movieId", "df2.movieId").agg(
        count(col("df1.movieId")).alias("numPairs"),
        sql_sum(col("df1.norm_rating") * col("df2.norm_rating")).alias("sum_xy"),
        sql_sum(col("df1.norm_rating") * col("df1.norm_rating")).alias("sum_xx"),
        sql_sum(col("df2.norm_rating") * col("df2.norm_rating")).alias("sum_yy")
    )

    # Calculate Pearson similarity
    result_df = joined_df.withColumn("numerator", col("sum_xy"))
    result_df = result_df.withColumn("denominator", sqrt(col("sum_xx")) * sqrt(col("sum_yy")))
    result_df = result_df.withColumn("similarity",
                                    when(col("denominator") != 0, col("numerator") / col("denominator"))
                                    .otherwise(lit(0)))

    return result_df.select(col("df1.movieId").alias("movieId1"), col("df2.movieId").alias("movieId2"), "similarity")

movie_similarity_df = compute_similarity(normalized_ratings_df)

# Specify the movie ID for which you want to find similar movies
target_movie_id = 1  # Toy Story

# Filter for similarities involving the target movie and sort by similarity score
top_similar_movies = movie_similarity_df.filter(
    (col("movieId1") == target_movie_id) | (col("movieId2") == target_movie_id)
).orderBy(col("similarity").desc(), col("movieId1"), col("movieId2"))

# Limit to top 10 similar movies
top_5_similar_movies = top_similar_movies.limit(5)


def calculate_item_cf_predictions(user_movie_pairs, item_similarity_df, user_ratings_df, N=10):
    """
    Calculate predictions for user-item pairs based on item-item CF.

    :param user_movie_pairs: DataFrame of user-item pairs
    :param item_similarity_df: DataFrame of item-item similarities
    :param user_ratings_df: DataFrame of user's historical ratings
    :param N: Number of top similar items to consider for prediction
    :return: DataFrame with predictions
    """
    # Alias for clarity
    similarities = item_similarity_df.alias("sims")
    ratings = user_ratings_df.alias("ratings")

    # Join similarities with user ratings
    user_item_sims = user_movie_pairs.alias("pairs").join(
        similarities, col("pairs.movieId1") == col("sims.movieId1")
    ).join(
        ratings, (col("ratings.userId") == col("pairs.userId")) & (col("ratings.movieId") == col("sims.movieId2"))
    )

    # Select only relevant columns and rename for clarity
    user_item_sims = user_item_sims.select(
        col("pairs.userId"),
        col("pairs.movieId1").alias("target_movieId"),
        col("sims.movieId2").alias("similar_movieId"),
        col("sims.similarity"),
        col("ratings.rating").alias("similar_movie_rating")
    )

    # Filter to top N similar movies for each target movie
    windowSpec = Window.partitionBy("userId", "target_movieId").orderBy(col("similarity").desc())
    top_n_similar = user_item_sims.withColumn("rank", rank().over(windowSpec)).filter(col("rank") <= N)

    # Calculate the weighted sum of ratings and sum of similarities
    weighted_ratings = top_n_similar.groupBy("userId", "target_movieId").agg(
        sql_sum(col("similarity") * col("similar_movie_rating")).alias("weighted_sum"),
        sql_sum("similarity").alias("similarity_sum")
    )

    # Compute the final prediction
    predictions = weighted_ratings.withColumn(
        "prediction", col("weighted_sum") / col("similarity_sum")
    ).select("userId", "target_movieId", "prediction")

    return predictions

""" Creating a hybrid algorithm"""

# Generate all user-item pairs (assuming you have a DataFrame of all unique movie IDs)
user_movie_pairs = test.select("userId").distinct().crossJoin(movie_similarity_df.select("movieId1").distinct())

item_cf_predictions = calculate_item_cf_predictions(user_movie_pairs, movie_similarity_df, train)

als_predictions = predictions.withColumnRenamed("prediction", "als_prediction")
item_cf_predictions = item_cf_predictions.withColumnRenamed("prediction", "cf_prediction")
item_cf_predictions = item_cf_predictions.withColumnRenamed("target_movieId", "movieId")
combined_predictions = als_predictions.join(item_cf_predictions, ["userId", "movieId"], "inner")
combined_predictions = combined_predictions.withColumn("hybrid_prediction",
                                                        (col("als_prediction") + col("cf_prediction")) / 2)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="actual_rating", predictionCol="hybrid_prediction")
actual_ratings_df = test.withColumnRenamed("rating", "actual_rating")


# Join the actual ratings with the hybrid predictions
evaluation_df = actual_ratings_df.join(combined_predictions, ["userId", "movieId"], "inner")

# Evaluate the model
h1_rmse = evaluator.evaluate(evaluation_df)
print(f"Hybrid Model RMSE: {h1_rmse}")

# Load the movies data
movies_df = spark.read.csv("hdfs:///user/apathak2/input/ml-25m/movies.csv", header=True, inferSchema=True)
movies_df = movies_df.cache()

# Split the genres string into an array
movies_df = movies_df.withColumn("genres_array", split(col("genres"), "\|"))
# Explode the genres_array into multiple rows
movies_df_exploded = movies_df.withColumn("genre", explode(col("genres_array")))
# Drop the original genres column and genres_array
movies_df_exploded = movies_df_exploded.drop("genres", "genres_array")


indexer = StringIndexer(inputCol="genre", outputCol="genreIndex")
encoder = OneHotEncoder(inputCols=["genreIndex"], outputCols=["genreVec"])

# Pipeline
pipeline = Pipeline(stages=[indexer, encoder])
genre_model = pipeline.fit(movies_df_exploded)
movies_df_transformed = genre_model.transform(movies_df_exploded)

# Drop the original genre and genreIndex columns
movies_df_transformed = movies_df_transformed.drop("genre", "genreIndex")
# Group by movieId and aggregate the genre vectors
movies_df_final = movies_df_transformed.groupBy("movieId", "title").agg(
    collect_list("genreVec").alias("genresVec")
)

# Repartition DataFrames based on the join key
ratings_df = ratings_df.repartition("movieId")
movies_df_final = movies_df_final.repartition("movieId")

combined_df = ratings_df.join(movies_df_final, "movieId")
(combined_train, combined_test) = combined_df.randomSplit([0.8, 0.2], seed=0)

""" Adding a third component to the hybrid algorithm"""

feature_columns = ['movieId', 'genresVec']

# Assemble features and train the model
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
rf = RandomForestRegressor(featuresCol="features", labelCol="rating")
pipeline = Pipeline(stages=[assembler, rf])
rf_model = pipeline.fit(combined_train)

supervised_predictions = rf_model.transform(combined_test)

# Join the ALS, item-item CF, and supervised model predictions
combined_predictions = als_predictions.join(item_cf_predictions, ["userId", "movieId"], "inner")
combined_predictions = combined_predictions.join(supervised_predictions, ["userId", "movieId"], "inner")


# Define the weight values to try
weight_values = [
    (0.2, 0.3, 0.5),
    (0.3, 0.3, 0.4),
    (0.4, 0.3, 0.3)
]

# Create a list to store the RMSE values for each weight combination
rmse_values = []

# Iterate over the weight combinations
for weight_als, weight_cf, weight_supervised in weight_values:
    # Calculate the hybrid prediction as a weighted average of all three predictions
    combined_predictions = combined_predictions.withColumn(
        "hybrid_prediction",
        (col("als_prediction") * weight_als + col("cf_prediction") * weight_cf + col("supervised_prediction") * weight_supervised)
    )

    # Join the actual ratings with the hybrid predictions
    evaluation_df = actual_ratings_df.join(combined_predictions.select("userId", "movieId", "hybrid_prediction"), ["userId", "movieId"], "inner")

    # Evaluate the model using RMSE
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="actual_rating", predictionCol="hybrid_prediction")
    h2_rmse = evaluator.evaluate(evaluation_df)
    rmse_values.append(h2_rmse)

# Find the index of the weight combination with the lowest RMSE
best_index = rmse_values.index(min(rmse_values))
best_weights = weight_values[best_index]

print(f"Best weights: {best_weights}")
print(f"Best Hybrid Model RMSE: {min(rmse_values)}")

# Writing all metrics to files
metrics_data = [
    Row(metric="ALS RMSE", value=float(rmse)),
    Row(metric="ALS MSE", value=float(mse)),
    Row(metric="ALS MAP", value=float(map)),
    Row(metric="Best Model Rank", value=float(best_model.rank)),
    Row(metric="Best Model MaxIter", value=float(best_model._java_obj.parent().getMaxIter())),
    Row(metric="Best Model RegParam", value=float(best_model._java_obj.parent().getRegParam())),
    Row(metric="Hybrid Model RMSE", value=float(h1_rmse)),
    Row(metric="Best Hybrid Model RMSE", value=float(rmse_values)),
    Row(metric="Best Weights", value=float(best_weights))
]

metrics_df = spark.createDataFrame(metrics_data)
metrics_df.write.format("csv").option("header", "true").coalesce(1).save("hdfs:///user/apathak2/output/metrics.csv")

