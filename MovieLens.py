import os
import sys

os.environ['JAVA_HOME'] = r"C:\Users\marko\AppData\Local\Programs\Eclipse Adoptium\jdk-17.0.17.10-hotspot"
os.environ['PATH'] = os.environ['JAVA_HOME'] + r'\bin;' + os.environ.get('PATH', '')
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

print(f"✓ JAVA_HOME: {os.environ['JAVA_HOME']}")
print(f"✓ Python: {sys.executable}")

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import pandas as pd

def create_spark_session():
    spark = SparkSession.builder \
        .appName("MovieLens-ALS-Recommendations") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_movielens_data(spark, data_path="ml-100k/u.data"):
    schema = ["userId", "movieId", "rating", "timestamp"]

    ratings = spark.read.csv(
        data_path,
        sep="\t",
        schema="userId INT, movieId INT, rating DOUBLE, timestamp LONG"
    )

    print("\n✓ Податоците се успешно вчитани")
    print(f"Вкупно рејтинзи: {ratings.count()}")
    print(f"Вкупно корисници: {ratings.select('userId').distinct().count()}")
    print(f"Вкупно филмови: {ratings.select('movieId').distinct().count()}")

    print("\nПример на податоци:")
    ratings.show(5)

    return ratings

def load_movie_titles(spark, movies_path="ml-100k/u.item"):
    movies_pd = pd.read_csv(
        movies_path,
        sep="|",
        encoding='latin-1',
        header=None,
        usecols=[0, 1],
        names=['movieId', 'title']
    )
    movies = spark.createDataFrame(movies_pd)
    return movies

def analyze_data(ratings):
    print("\n" + "="*60)
    print("ЕКСПЛОРАТИВНА АНАЛИЗА НА ПОДАТОЦИ")
    print("="*60)

    ratings.describe(['rating']).show()

    print("\nРаспределба на рејтинзи:")
    ratings.groupBy('rating').count().orderBy('rating').show()

    print("\nТоп 5 корисници со најмногу рејтинзи:")
    ratings.groupBy('userId').count() \
        .orderBy(col('count').desc()).limit(5).show()

    print("\nТоп 5 најрејтирани филмови:")
    ratings.groupBy('movieId').count() \
        .orderBy(col('count').desc()).limit(5).show()

def split_data(ratings, train_ratio=0.8, seed=42):
    train, test = ratings.randomSplit([train_ratio, 1-train_ratio], seed=seed)

    print("\n" + "="*60)
    print("ПОДЕЛБА НА ПОДАТОЦИ")
    print("="*60)
    print(f"Тренинг сет: {train.count()} рејтинзи ({train_ratio*100}%)")
    print(f"Тест сет: {test.count()} рејтинзи ({(1-train_ratio)*100}%)")

    return train, test

def train_als_model(train_data, rank=10, maxIter=10, regParam=0.1):
    print("\n" + "="*60)
    print("ТРЕНИРАЊЕ НА ALS МОДЕЛ")
    print("="*60)
    print(f"Параметри: rank={rank}, maxIter={maxIter}, regParam={regParam}")

    als = ALS(
        rank=rank,
        maxIter=maxIter,
        regParam=regParam,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        seed=42
    )

    model = als.fit(train_data)

    return model

def evaluate_model(model, test_data):
    print("\n" + "="*60)
    print("ЕВАЛУАЦИЈА НА МОДЕЛ")
    print("="*60)

    predictions = model.transform(test_data)

    evaluator_rmse = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator_rmse.evaluate(predictions)

    evaluator_mae = RegressionEvaluator(
        metricName="mae",
        labelCol="rating",
        predictionCol="prediction"
    )
    mae = evaluator_mae.evaluate(predictions)

    print(f"\n📊 Метрики на точност:")
    print(f"RMSE (Root Mean Square Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"\nПојасно: моделот греши во просек за ±{mae:.2f} ѕвездички")

    return predictions, rmse, mae

def generate_recommendations(model, movies, n=10):
    print("\n" + "="*60)
    print(f"ГЕНЕРИРАЊЕ НА ТОП-{n} ПРЕПОРАКИ")
    print("="*60)

    user_recs = model.recommendForAllUsers(n)

    movie_recs = model.recommendForAllItems(n)

    return user_recs, movie_recs

def show_user_recommendations(user_recs, movies, user_id=1, n=10):
    print(f"\n{'='*60}")
    print(f"ТОП-{n} ПРЕПОРАКИ ЗА КОРИСНИК {user_id}")
    print('='*60)

    user_rec = user_recs.filter(col("userId") == user_id).collect()

    if user_rec:
        recommendations = user_rec[0]['recommendations']

        print(f"\n{'Ред':<5} {'Филм ID':<10} {'Предвиден рејтинг':<20} {'Наслов'}")
        print("-" * 80)

        for i, rec in enumerate(recommendations, 1):
            movie_id = rec['movieId']
            score = rec['rating']

            movie_title = movies.filter(col("movieId") == movie_id).collect()
            title = movie_title[0]['title'] if movie_title else "Непознат"

            print(f"{i:<5} {movie_id:<10} {score:<20.2f} {title}")

def hyperparameter_tuning(train_data, test_data):
    print("\n" + "="*60)
    print("ХИПЕРПАРАМЕТАРСКО ТУНИРАЊЕ")
    print("="*60)

    ranks = [5, 10, 15]
    reg_params = [0.01, 0.1, 0.5]

    best_rmse = float('inf')
    best_params = {}
    results = []

    for rank in ranks:
        for reg in reg_params:
            print(f"\nТестирање: rank={rank}, regParam={reg}")

            model = train_als_model(train_data, rank=rank, regParam=reg, maxIter=10)
            _, rmse, _ = evaluate_model(model, test_data)

            results.append({
                'rank': rank,
                'regParam': reg,
                'rmse': rmse
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'rank': rank, 'regParam': reg}

    print("\n" + "="*60)
    print("РЕЗУЛТАТИ ОД ТУНИРАЊЕ")
    print("="*60)
    print(f"\n✓ Најдобри параметри: {best_params}")
    print(f"✓ Најдобар RMSE: {best_rmse:.4f}")

    return best_params, results

def main():

    print("\n" + "="*60)
    print("SPARK ALS СИСТЕМ ЗА ПРЕПОРАКИ НА ФИЛМОВИ")
    print("MovieLens 100K Dataset")
    print("="*60)

    spark = create_spark_session()

    ratings = load_movielens_data(spark)
    movies = load_movie_titles(spark)

    analyze_data(ratings)

    train, test = split_data(ratings, train_ratio=0.8)

    model = train_als_model(train, rank=10, maxIter=15, regParam=0.1)

    predictions, rmse, mae = evaluate_model(model, test)

    user_recs, movie_recs = generate_recommendations(model, movies, n=10)

    show_user_recommendations(user_recs, movies, user_id=1, n=10)
    show_user_recommendations(user_recs, movies, user_id=100, n=10)

    print("\n" + "="*60)
    print("✓ АНАЛИЗАТА Е ЗАВРШЕНА")
    print("="*60)

    spark.stop()

if __name__ == "__main__":
    main()