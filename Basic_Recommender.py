#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Collaborative Filtering for Final Project

Usage:

    $ spark-submit Basic_Recommender.py cf_train_encode.parquet cf_val_encode.parquet ALS_Model > results.txt 2>&1

'''


# We need sys to get the command line arguments
import sys

# Other packages
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

import itertools
from math import inf
import pandas as pd


def main(spark, train_file, val_file, model_file):

    # Create dataframes
    train_df = spark.read.parquet(train_file)
    train_df.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    
    val_df = spark.read.parquet(val_file)
    val_df.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    
    
    # Declare evaluation model object
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
    
    # Variables for hyperparameter tuning
    bestRmse = 7.651545231817701
    
    ranks = [i for i in range(9,15,3)]
    regParams = [10**i for i in range(-5,-3)]
    alphas = [10*i for i in range(1,5)]
    
    # results_df = pd.DataFrame(columns=['Rank', 'Regularization', 'Alpha', 'RMSE'])
    
    # Model training 
    for i,j,k in itertools.product(ranks, regParams, alphas):
        als = ALS(userCol="user_num", itemCol="track_num", ratingCol="count", coldStartStrategy="drop",\
                  implicitPrefs=True, rank=i, regParam=j, alpha=k)
        model = als.fit(train_df)
        prediction = model.transform(val_df)
        rmse = evaluator.evaluate(prediction)
        # results_df = results_df.append({'Rank': i, 'Regularization': j, 'Alpha': k, 'RMSE':rmse}, ignore_index=True)
        print("RMSE (validation) =", rmse,"for the model trained with rank =", i, "regParam = ",j,"and alpha = ", k)
        if rmse < bestRmse:
            bestModel = model
            bestRmse = rmse
            bestRank = i
            bestReg = j
            bestAlpha = k
            bestModel.write().overwrite().save(model_file)

    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender').getOrCreate()

    # Get the filename from the command line
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    
    # And the location to store the trained model
    model_file = sys.argv[3]

    # Call our main routine
    main(spark, train_file, val_file, model_file)
