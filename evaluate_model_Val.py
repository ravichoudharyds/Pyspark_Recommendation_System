#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Collaborative Filtering for Final Project

Usage:
    Train File cf_train_encode.parquet
    Val File cf_val_encode.parquet
    Test File cf_test_encode.parquet
    $ spark-submit --driver-memory 8g --executor-memory 16g --deploy-mode cluster --executor-cores 4 --driver-cores 2 --num-executors 50 evaluate_model_Val.py cf_val_encode.parquet > results_val.txt 2>&1

'''


# We need sys to get the command line arguments
import sys
import pyspark
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
# Other packages
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import Window
from pyspark.sql.functions import col, expr
import pyspark.sql.functions as F


def main(spark, test_file):
    '''Main routine for supervised training

    Parameters
    ----------
    spark : SparkSession object

    test_file : string, path to the parquet file to load

    '''

    # Variable declaration
    num_tracks = 500
    test_df = spark.read.parquet(test_file).select("count","track_num","user_num")
    
    model_list = ["ALS_Model","ALS_drop_Model","ALS_log_Model"]
    results_dict = {"ALS_Model":"Results_ALS_val", "ALS_log_Model":"Results_ALS_Log_val", "ALS_drop_Model":"Results_ALS_Drop_val"}
    
    # Performance Evaluation
    for model_file in model_list:
        model = ALSModel.load(model_file)
        test_users = test_df.select("user_num").distinct()
        prediction = model.recommendForUserSubset(test_users, num_tracks)
        
        order_track = Window.partitionBy('user_num').orderBy(col('count').desc())
        ActualTop500Tracks = test_df \
        .withColumn('rank',F.rank().over(order_track)) \
        .where('rank <= {0}'.format(num_tracks)) \
        .groupBy('user_num') \
        .agg(expr('collect_list(track_num) as track_list_act'))
        
        PredAndLabelRDD = ActualTop500Tracks.join(prediction, 'user_num') \
        .rdd \
        .map(lambda row: ([rec.track_num for rec in row["recommendations"]],row["track_list_act"]))
        rankingMetrics = RankingMetrics(PredAndLabelRDD)
        
        mavpr = rankingMetrics.meanAveragePrecision
        pat_num = rankingMetrics.precisionAt(num_tracks)
        ndcg_num = rankingMetrics.ndcgAt(num_tracks)
        results = spark.createDataFrame([(pat_num, ndcg_num, mavpr)], ["Precision_At", "NDCG_At","mean_Av_Prec"])
        results.write.mode("overwrite").parquet(results_dict[model_file])
    
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('Evaluate_Models').getOrCreate()

    # Get the filename from the command line
    test_file = sys.argv[1]

    # Call our main routine
    main(spark, test_file)
