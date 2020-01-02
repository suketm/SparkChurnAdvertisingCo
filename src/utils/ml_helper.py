
### 1. IMPORTS
import os
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.classification import RandomForestClassifier

### 2. SET LOG

### 3. CLASSES

class FeatureEngineering(object):

	def __init__(self):
		pass

	def pipeline_build(self, in_list_pipeline_features, in_feature_name):
		""" 
		build pipeline
		: in_list_pipeline_features	: list of features
		: in_feature_name			: feature name in final data
		: out_pipeline       		: pipeline object
		"""
		assembler = VectorAssembler(
					inputCols = in_list_pipeline_features,
    				outputCol = "featuresToScale")
		scaler = StandardScaler(
					inputCol = "featuresToScale", 
					outputCol = in_feature_name
					,withStd=True)
		out_pipeline = Pipeline(stages = [assembler, scaler])
		return out_pipeline

	def pipeline_build_model(self, in_pipeline,  in_df):
		""" 
		build pipeline model
		: in_pipeline 			: pipeline object
		: in_df					: spark dataframe for machine learning
		: out_pipeline_model    : pipeline model for feature engineering
		"""
		out_pipeline_model = in_pipeline.fit(in_df)
		return out_pipeline_model

	def pipeline_run(self, in_pipeline_model, in_df):
		""" 
		create spark dataframe with final features
		: in_pipeline_model				: pipeline model for feature engineering
		: in_df							: spark dataframe for machine learning
		: out_feature_engineered_data	: spark dataframe with final features
		"""
		out_feature_engineered_data = in_pipeline_model.transform(in_df)
		return out_feature_engineered_data


class MachineLearningTrain(object):

	def __init__(self):
		pass

	def model_build(self, in_feature_vec, in_label_vec):
		""" 
		build machine learning model
		: in_feature_vec	:	feacture vector name
		: in_label_vec		:	label vector name
		: out_model 	    :	machine learning model
		"""
		# fit model paramters + pass model parameters as fucntions attributes
		out_model = LogisticRegression(labelCol= in_label_vec, featuresCol = in_feature_vec)
		return out_model

	def model_train(self, in_model, in_data):
		""" 
		train machine learning model
		: in_model 			:	machine learning model
		: in_data 			:	training data
		: out_model_train  	:	trained machine learning model
		"""
		# ensembling
		out_model_train = in_model.fit(in_data)
		return out_model_train

	def model_save(self, in_model_train, in_path_model):
		""" 
		save machine learning model
		: in_model_train 	:	trained machine learning model
		: in_path_model		:	directory to store machine learning model
		"""
		if os.path.exists(in_path_model):
			in_model_train.write().overwrite().save(in_path_model)
		else:
			in_model_train.save(in_path_model)
		return None

	def model_train_results_save(self):
		return None


class MachineLearningTest(object):

	def __init__(self):
		pass

	def model_load(self, in_path_model):
		""" 
		load trained machine learning model
		: in_path_model		:	directory to store machine learning model
		: out_model 		:	trained machine learning model
		"""
		out_model = LogisticRegressionModel.load(in_path_model)
		return out_model

	def model_predict(self, in_model, in_df):
		""" 
		make prediction using ml model
		: in_model			:	directory to store machine learning model
		: in_df 			:	spark dataframe - test data for machine learning
		: out_df_predicted 	:	predicted data
		"""
		out_df_predicted = in_model.transform(in_df)
		return out_df_predicted

	def model_predictions_save(self, in_df_predicted, in_path_predictions, in_list_columns):
		""" 
		save predictions
		: in_df_predicted			:	predicted data
		: in_list_columns 			:	list of columns of input data
		: in_path_predictions 		:	directory to store predicted data
		"""
		in_df_predicted.select( in_list_columns + ['rawPrediction', 'probability', 'prediction']) \
						.write.orc(in_path_predictions)
		return None