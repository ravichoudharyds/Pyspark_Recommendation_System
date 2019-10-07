#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Collaborative Filtering for Final Project

Usage:
    Train File cf_train_encode.parquet
    Val File cf_val_encode.parquet
    Test File cf_test_encode.parquet
    $ spark-submit --driver-memory 8g --executor-memory 8g Extension_Drop_Low_Count.py cf_train_encode.parquet cf_val_encode.parquet ALS_drop_Model > results_ext.txt 2>&1

'''


# We need sys to get the command line arguments
import sys
import pyspark
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
# Other packages
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

import itertools
from math import inf
import pandas as pd
from pyspark import SparkConf

conf = SparkConf()


def main(spark, train_file, val_file, model_file):
    '''Main routine for extension implementation

    Parameters
    ----------
    spark : SparkSession object

    data_file : string, path to the parquet file to load

    model_file : string, path to store the serialized model file
    '''

    # Create dataframes by dropping low count(1)
    train_df = spark.read.parquet(train_file).filter('count > 1').select('count','track_num','user_num')
    train_df.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
        
    val_df = spark.read.parquet(val_file).select("count","track_num","user_num")
    val_df.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    
    # Declare evaluation model object
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
    
    # Variables for hyperparameter tuning
    bestRmse = np.Inf
    
    ranks = [i for i in range(9,15,3)]
    regParams = [10**i for i in range(-5,-3)]
    alphas = [10*i for i in range(1,5)]
    
    # results_df = pd.DataFrame(columns=['Rank', 'Regularization', 'Alpha', 'RMSE'])
    
    # Model training 
    for i,j,k in itertools.product(ranks, regParams, alphas):
        als = ALS(userCol="user_num", itemCol="track_num", ratingCol="count", coldStartStrategy="drop", \
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

    # spark_df = spark.createDataFrame(results_df)
    # spark_df.repartition(1).write.overwrite().option("header", "true").save("Extension_ALS.csv")
    
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('extension_drop_low_count').getOrCreate()

    # Get the filename from the command line
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    
    # And the location to store the trained model
    model_file = sys.argv[3]

    # Call our main routine
    main(spark, train_file, val_file, model_file)
