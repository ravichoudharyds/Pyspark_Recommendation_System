#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''String Indexer for Final Project

Usage:

    $ spark-submit String_Indexer.py hdfs:/user/bm106/pub/project/metadata.parquet hdfs:/user/bm106/pub/project/cf_train.parquet hdfs:/user/bm106/pub/project/cf_validation.parquet hdfs:/user/bm106/pub/project/cf_test.parquet

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer


def main(spark, track_file, train_file, val_file, test_file):

    # Create dataframes for all the input files
    df_track = spark.read.parquet(track_file)
    df_train = spark.read.parquet(train_file)
    df_val = spark.read.parquet(val_file)
    df_test = spark.read.parquet(test_file)
    
    # Model training
    user_id_encode = StringIndexer(inputCol="user_id", outputCol="user_num")
    track_id_encode = StringIndexer(inputCol="track_id", outputCol="track_num")
    encoding1 = user_id_encode.fit(df_train)
    encoding2 = track_id_encode.fit(df_track)
    
    # Train
    df_train_1 = encoding1.transform(df_train)
    df_train_2 = encoding2.transform(df_train_1)
    df_train_2.repartition(5000,'user_num')
    df_train_2.write.parquet('cf_train_encode.parquet')
    
    # Validation
    df_val_1 = encoding1.transform(df_val)
    df_val_2 = encoding2.transform(df_val_1)
    df_val_2.repartition('user_num')
    df_val_2.write.parquet('cf_val_encode.parquet')
    
    # Test
    df_test_1 = encoding1.transform(df_test)
    df_test_2 = encoding2.transform(df_test_1)
    df_test_2.repartition('user_num')
    df_test_2.write.parquet('cf_test_encode.parquet')
    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('stringindexer').getOrCreate()

    # Get the filenames from the command line
    track_file = sys.argv[1]
    train_file = sys.argv[2]
    val_file = sys.argv[3]
    test_file = sys.argv[4]

    # Call our main routine
    main(spark, track_file, train_file, val_file, test_file)
