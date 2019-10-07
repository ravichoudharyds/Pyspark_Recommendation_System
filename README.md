# DSGA1004 - BIG DATA
## Final project - Recommendation System using PySpark

# Overview

In the final project, we have implemented Recommendation System through Collaborative Filtering in PySpark and evaluated it. 

## The data set
  
The data was user-song interaction data. It was split into three files contain training, validation, and testing data for the collaborative filter.  Specifically, each file contains a table of triples `(user_id, count, track_id)` which measure implicit feedback derived from listening behavior.  

The training data contains full histories for approximately 1M users, and partial histories for 110,000 users, located at the end of the table. Validation data has remaining history for 10k users. It  is used to tune the hyperparameters of the model.

Test Data was used for final evaluation. It contains the remaining history for 100K users.

## Basic recommender system

We used Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.  This model has some hyper-parameters that we tuned to optimize performance on the validation set, notably: 

  - the *rank* (dimension) of the latent factors,
  - the *regularization* parameter, and
  - *alpha*, the scaling parameter for handling implicit feedback (count) data.

The choice of evaluation criteria for hyper-parameter tuning was MSE. Once the model was trained, it was evaluated on the test set using the ranking metrics provided by spark.  Evaluations were based on predictions of the top 500 items for each user.

We transformed the user and item identifiers (strings) into numerical index representations for the ease of Spark's ALS model.

We conducted a thorough evaluation of different modification strategies (e.g., log compression, or dropping low count values) and their impact on overall accuracy.
