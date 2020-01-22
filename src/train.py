
"""Train the churn prediction models"""

### 1. IMPORTS

import os
import yaml
from pyspark.ml import Pipeline
import logging
from logging import info

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Churn Classification').getOrCreate()


### 2. LOAD EXTERNAL LIBRARIES

from utils.ml_helper import FeatureEngineering, MachineLearningTrain


### 3. SET LOG

logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s %(levelname)s %(name)s %(message)s',
                    filename = 'logs_train.txt')


### 4. LOAD CONFIG

with open('..\\config\\config.yaml', 'r', encoding='utf-8') as yml:
    config = yaml.load(yml, Loader=yaml.SafeLoader)

### 5. MAIN:

if __name__ == '__main__':
	
	### LOAD DATA
	df_train = spark.read.csv(os.path.join(config['paths']['DATA_DIR'],config['paths']['TRAINING_DATA']),
							inferSchema = True, 
							header = True)
	
	### CONSTANTS
	LIST_PIPELINE_FEATURES = config['LIST_PIPELINE_FEATURES']
	FEATURE_VEC = config['FEATURE_VEC']
	LABEL_VEC = config['LABEL_VEC']
	path_model = os.path.join(config['paths']['MODEL_DIR'],config['paths']['MODEL_DATA'])
	path_model_parameter = os.path.join(config['paths']['MODEL_DIR'],config['paths']['MODEL_PARAMS'])
	ml_model_param = {}
	ml_model_param['MAXDEPTH'] = config['Machine Learning Model Parameters']['MAXDEPTH']
	ml_model_param['NUMTREES'] = config['Machine Learning Model Parameters']['NUMTREES']
	ml_model_param['MAXBINS'] = config['Machine Learning Model Parameters']['MAXBINS']
	ml_model_param['MININSTANCESPERNODE'] = config['Machine Learning Model Parameters']['MININSTANCESPERNODE']

	
	### FUNCTIONS WRAPPER

	# feature engineering
	info('Feature engineering of training data starts.')
	feature_engineering = FeatureEngineering()
	pipeline = feature_engineering.pipeline_build(in_list_pipeline_features = LIST_PIPELINE_FEATURES, 
													in_feature_name = FEATURE_VEC)
	pipeline_model = feature_engineering.pipeline_build_model(in_pipeline = pipeline, in_df = df_train)
	feature_engineered_data = feature_engineering.pipeline_run(in_pipeline_model = pipeline_model, in_df = df_train)
	info('Feature engineering of training data completed.')

	# machine learning
	info('Machine learning on training data starts.')
	# Random Forest Classifier parameters found by Grid Search
	machine_learning = MachineLearningTrain()
	model = machine_learning.model_build(in_feature_vec = FEATURE_VEC, in_label_vec = LABEL_VEC, in_ml_model_param = ml_model_param)
	model_train = machine_learning.model_train(in_model = model, in_data = feature_engineered_data)
	machine_learning.model_train_results_save()
	machine_learning.model_save(in_model_train = model_train, in_path_model = path_model)
	info('Training is completed. Machine learning model completed.')