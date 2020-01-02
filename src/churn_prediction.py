
"""Churn prediction on new data"""

### 1. IMPORTS

import os
import yaml
from pyspark.ml import Pipeline
import logging
from logging import info

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Churn Classification').getOrCreate()


### 2. LOAD EXTERNAL LIBRARIES

from utils.ml_helper import FeatureEngineering, MachineLearningTest


### 3. SET LOG

logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s %(levelname)s %(name)s %(message)s',
                    filename = 'logs_test.txt')


### 4. LOAD CONFIG

with open('..\\config\\config.yaml', 'r', encoding='utf-8') as yml:
    config = yaml.load(yml, Loader=yaml.SafeLoader)

### 5. MAIN:

if __name__ == '__main__':
	
	### LOAD DATA
	df_test = spark.read.csv(os.path.join(config['paths']['DATA_DIR'],config['paths']['TEST_DATA']),
							inferSchema = True, 
							header = True)
	
	### CONSTANTS
	LIST_PIPELINE_FEATURES = config['LIST_PIPELINE_FEATURES']
	FEATURE_VEC = config['FEATURE_VEC']
	LABEL_VEC = config['LABEL_VEC']
	path_model = os.path.join(config['paths']['MODEL_DIR'],config['paths']['MODEL_DATA'])
	path_predictions = os.path.join(config['paths']['RESULTS_DIR'],config['paths']['RESULTS_DATA'])
	list_columns = df_test.columns

	### FUNCTIONS WRAPPER

	# feature engineering
	info('Feature engineering of test data starts.')
	feature_engineering = FeatureEngineering()
	pipeline = feature_engineering.pipeline_build(in_list_pipeline_features = LIST_PIPELINE_FEATURES, 
													in_feature_name = FEATURE_VEC)
	pipeline_model = feature_engineering.pipeline_build_model(in_pipeline = pipeline, in_df = df_test)
	feature_engineered_data = feature_engineering.pipeline_run(in_pipeline_model = pipeline_model, in_df = df_test)
	info('Feature engineering of training data completed.')

	# machine learning
	info('Prediction on test data starts.')
	machine_learning = MachineLearningTest()
	model = machine_learning.model_load(in_path_model = path_model)
	df_predicted = machine_learning.model_predict(in_model = model, in_df = feature_engineered_data)
	machine_learning.model_predictions_save(in_path_predictions = path_predictions, 
											in_df_predicted = df_predicted,
											in_list_columns = list_columns + [FEATURE_VEC])
	info('Prediction is completed.')