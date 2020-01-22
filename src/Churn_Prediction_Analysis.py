#!/usr/bin/env python
# coding: utf-8

import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Churn Classification").getOrCreate()

from pyspark.sql import functions as F
from pyspark.ml.stat import Correlation


# load config file

with open('..\\config\\config.yaml', 'r', encoding='utf-8') as yml:
    config = yaml.load(yml, Loader=yaml.SafeLoader)
    
df = spark.read.csv(config['paths']['DATA_DIR']+config['paths']['TRAINING_DATA'], inferSchema = True, header = True)

# constants
path_FIGURE_DIR = config['paths']['FIGURE_DIR']
path_MODEL_DIR = config['paths']['MODEL_DIR']
file_MODEL_PARAMS = config['paths']['MODEL_PARAMS']
lr_MAXITER = config['ml_model_optimizer']['logistic_regression']['MAXITER']
lr_REGUPARAM = config['ml_model_optimizer']['logistic_regression']['REGUPARAM']
lr_ALPHA = config['ml_model_optimizer']['logistic_regression']['ALPHA']
lr_THRESHOLD = config['ml_model_optimizer']['logistic_regression']['THRESHOLD']
rf_MAXDEPTH = config['ml_model_optimizer']['random_forest']['MAXDEPTH']
rf_NUMTREES = config['ml_model_optimizer']['random_forest']['NUMTREES']
rf_MAXBINS = config['ml_model_optimizer']['random_forest']['MAXBINS']
rf_MININSTANCESPERNODE = config['ml_model_optimizer']['random_forest']['MININSTANCESPERNODE']

num_params_lr = len(lr_MAXITER)*len(lr_REGUPARAM)*len(lr_ALPHA)*len(lr_THRESHOLD)
num_params_rf = len(rf_MAXDEPTH)*len(rf_NUMTREES)*len(rf_MAXBINS)*len(rf_MININSTANCESPERNODE)


df.printSchema()


df.show(3)


for row in df.head(3):
    print (row,'\n')


print (df.columns)


# Churn
df.groupBy('Churn').count().show()
# since data is imbalanced, we will consider AUC to evaluate our model

# Churn distribution
objects = ('Churn', 'not Churn')
y_pos = np.arange(len(objects))
performance = [df.groupBy('Churn').count().collect()[0][1], df.groupBy('Churn').count().collect()[1][1]]

plt.bar(y_pos, performance, align='center', alpha=0.3, color = ['grey','g'])
plt.xticks(y_pos, objects)
plt.ylabel('# of users')
plt.title('Churn Distribution')
plt.savefig(path_FIGURE_DIR+'Churn Distribution')
plt.show()


# Age
df.select('Age').printSchema()
a = df.filter(df.Churn == 1).select('Age').collect()
b = df.filter(df.Churn == 0).select('Age').collect()
a = [value[0] for value in a]
b = [value[0] for value in b]
plt.title('Age Distribution')
plt.hist(b, label = 'Not Churn', color = 'green', alpha = 0.3)
plt.hist(a, label = 'Churn', color = 'grey', alpha = 0.7)
plt.xlabel('Age')
plt.ylabel('# of users')
plt.legend()
plt.savefig(path_FIGURE_DIR+'Age Distribution')
plt.show()
# Feature Engineering: Age - normalization


# Total_Purchase
a = df.filter(df.Churn == 1).select('Total_Purchase').collect()
b = df.filter(df.Churn == 0).select('Total_Purchase').collect()
a = [value[0] for value in a]
b = [value[0] for value in b]
plt.title('Purchase Distribution')
plt.hist(b, label = 'Not Churn', color = 'green', alpha = 0.3)
plt.hist(a, label = 'Churn', color = 'grey', alpha = 0.7)
plt.xlabel('Total Purchases')
plt.ylabel('# of users')
plt.legend()
plt.savefig(path_FIGURE_DIR+'Purchase Distribution')
plt.show()
# Feature Engineering: Total_Pruchase - normalization


# Account_Manager
acc_mngr_Vect = df.select('Account_Manager')
acc_mngr_Vect.printSchema()
acc_mngr_Vect.distinct().show()
# number of classes : 2
acc_mngr_Vect.groupBy('Account_Manager').count().show()
# Feature Engineering: Not required

# Account_Manger Distribution
num_churn_acc = df.filter((df.Churn == 1) & (df.Account_Manager == 1)).count()
num_churn_no_acc = df.filter((df.Churn == 1) & (df.Account_Manager == 0)).count()
num_no_churn_acc = df.filter((df.Churn == 0) & (df.Account_Manager == 1)).count()
num_no_churn_no_acc = df.filter((df.Churn == 0) & (df.Account_Manager == 0)).count()

print ("Churn and Account Manager:",num_churn_acc,
       "\nChurn and No Account Manager:",num_churn_no_acc,
       "\nNo Churn and Account Manager:",num_no_churn_acc,
       "\nNo Churn and No Account Manager:",num_no_churn_no_acc,)

objects = ('Yes', 'No')
y_pos = np.arange(len(objects))
performance_Churn = [num_churn_acc/(num_churn_acc+num_no_churn_acc)*100,
					num_churn_no_acc/(num_churn_no_acc+num_no_churn_no_acc)*100]
performance_not_Churn = [100,100] 

plt.bar(y_pos, performance_not_Churn, align='center', alpha=0.3, color = 'g')
plt.bar(y_pos, performance_Churn, align='center', alpha=0.7, color = 'grey', label = 'Churn %')
plt.xticks(y_pos, objects)
plt.ylabel('% of user')
plt.xlabel('Account Manager')
plt.legend()
plt.title('Account Manager vs. Churn')
plt.savefig(path_FIGURE_DIR+'Account Manager_vs_Churn')


# Years
a = df.filter(df.Churn == 1).select('Years').collect()
b = df.filter(df.Churn == 0).select('Years').collect()
a = [value[0] for value in a]
b = [value[0] for value in b]
plt.title('Years Distribution')
plt.hist(b, label = 'Not Churn', color = 'green', alpha = 0.3)
plt.hist(a, label = 'Churn', color = 'grey', alpha = 0.7)
plt.xlabel('Years')
plt.ylabel('# of users')
plt.legend()
plt.savefig(path_FIGURE_DIR+'Years Distribution')
plt.show()
# Feature Engineering: Years - normalization


# Num_Sites
a = df.filter(df.Churn == 1).select('Num_Sites').collect()
b = df.filter(df.Churn == 0).select('Num_Sites').collect()
a = [value[0] for value in a]
b = [value[0] for value in b]
plt.title('Number of Sites Distribution')
plt.hist(b, label = 'Not Churn', color = 'green', alpha = 0.3)
plt.hist(a, label = 'Churn', color = 'grey', alpha = 0.7)
plt.xlabel('Years')
plt.ylabel('# of users')
plt.legend()
plt.savefig(path_FIGURE_DIR+'Number of Sites Distribution')
plt.show()
# Feature Engineering: Num_Sites - normalization


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml import Pipeline

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# Correlation Matrix

# Location Indexer
indexer_location = StringIndexer(inputCol="Location", outputCol="LocationIndex")

# Company Indexer
indexer_Co = StringIndexer(inputCol="Company", outputCol="CompanyIndex")

# Assembler: to generate Scale Vector
assembler_corr = VectorAssembler(
    inputCols=['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites','LocationIndex','CompanyIndex', 'Churn' ],
    outputCol="features_Corr")

# Feature Engineering Pipeline
pipeline_corr = Pipeline(stages = [indexer_location, indexer_Co, assembler_corr])


pipepline_corr_2 = pipeline_corr.fit(df)
data_complete_corr = pipepline_corr_2.transform(df)
data_corr = data_complete_corr.select('features_Corr')

r1 = Correlation.corr(data_corr, "features_Corr").head()
print("Pearson correlation matrix:\n" + str(r1[0]))


list_features = ['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites']


# Feature Engineering
# Features: 'Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites'


# Assembler: to generate Scale Vector
assembler = VectorAssembler(
    inputCols=list_features,
    outputCol="featuresToScale")

# Feature Standardizer: Age, Total_Pruchase, Years, Num_Sites
scaler = StandardScaler(inputCol="featuresToScale", outputCol="features",withStd=True)

# Feature Engineering Pipeline
pipeline = Pipeline(stages = [assembler, scaler])


feature_engineering_pipepline = pipeline.fit(df)
feature_engineered_data = feature_engineering_pipepline.transform(df)
data = feature_engineered_data.select('features','Churn')


lr = LogisticRegression(labelCol = 'Churn')
grid_lr = ParamGridBuilder()             .addGrid(lr.threshold, lr_THRESHOLD)            .addGrid(lr.regParam, lr_REGUPARAM)            .addGrid(lr.maxIter, lr_MAXITER)            .addGrid(lr.elasticNetParam, lr_ALPHA)            .build()


rf = RandomForestClassifier(labelCol = 'Churn')
grid_rf = ParamGridBuilder()            .addGrid(rf.maxDepth, rf_MAXDEPTH)            .addGrid(rf.maxBins, rf_MAXBINS)            .addGrid(rf.numTrees, rf_NUMTREES)            .addGrid(rf.minInstancesPerNode, rf_MININSTANCESPERNODE)            .build()


# since distribution of Churn is not balance, we will use AUC matrix to make the predictions
evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='Churn')


# build cross validator
cv_lr = CrossValidator(estimator = lr, estimatorParamMaps = grid_lr, evaluator = evaluator, numFolds = 3)
cv_rf = CrossValidator(estimator = rf, estimatorParamMaps = grid_rf, evaluator = evaluator, numFolds = 3)


# fit models
cv_model_lr = cv_lr.fit(data)
cv_model_rf = cv_rf.fit(data)


best_model_lr = cv_model_lr.bestModel
best_model_rf = cv_model_rf.bestModel


# Feature Importance
n_features = 5
plt.barh(range(n_features), best_model_rf.featureImportances, align='center', color = 'green', alpha = 0.3)
plt.yticks(np.arange(n_features), list_features)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.ylim(-1, n_features)
plt.savefig(path_FIGURE_DIR+'Feature Importance')


auc_vec_lr = cv_model_lr.avgMetrics
auc_vec_rf = cv_model_rf.avgMetrics


# Result Visual: Logistic Regression
print ('\nLogistic Regression Results')
dict_log_reg_results = {'AUC':[]}
for param_idx in range(num_params_lr):
    list_param_value = list(grid_lr[param_idx].keys())
    for param_value in list_param_value:
        if param_value.name not in dict_log_reg_results:
            dict_log_reg_results[param_value.name] = []
        dict_log_reg_results[param_value.name].append(grid_lr[param_idx][param_value])
    dict_log_reg_results['AUC'].append(round(auc_vec_lr[param_idx],2))
pd.DataFrame(dict_log_reg_results)


# Result Visual: Random Forest
print ('\nRandom Forest Results')
dict_random_forest_results = {'AUC':[]}
for param_idx in range(num_params_rf):
    list_param_value = list(grid_rf[param_idx].keys())
    for param_value in list_param_value:
        if param_value.name not in dict_random_forest_results:
            dict_random_forest_results[param_value.name] = []
        dict_random_forest_results[param_value.name].append(grid_rf[param_idx][param_value])
    dict_random_forest_results['AUC'].append(round(auc_vec_rf[param_idx],2))
pd.DataFrame(dict_random_forest_results)


def gain_chart(predicted_data):
    predicted_data_sort = predicted_data.orderBy('probability')
    num_users = predicted_data_sort.count()
    gain_analysis = {'decile_num':list(range(1,11)),
                     'num_cust': [0]*10,
                     'actual_churn':[0]*10,
                     'predicted_churn':[0]*10,
                     'Precission': [0]*10,
                     'Recall': [0]*10,
                     'F1_Score': [0]*10,
                     'Lift': [0]*10
                      }
    list_churn = [churn_value[0] for churn_value in predicted_data_sort.select('Churn').collect() ]
    list_predicted = [prd_churn_value[0] for prd_churn_value in predicted_data_sort.select('prediction').collect()]

    for decile in range(10):
        gain_analysis['num_cust'][decile] = gain_analysis['num_cust'][decile-1] + num_users/10
        gain_analysis['actual_churn'][decile] = sum(list_churn[:int((decile+1)*num_users/10)])/sum(list_churn)*100
        gain_analysis['predicted_churn'][decile] = sum(list_predicted[:int((decile+1)*num_users/10)])/sum(list_predicted)*100
        list_temp_predicted = list_predicted[:int((decile+1)*num_users/10)]
        list_temp_actual = list_churn[:int((decile+1)*num_users/10)]
        true_pos = len([idx for idx in range(int((decile+1)*num_users/10))                         if list_temp_predicted[idx] == 1 and list_temp_actual[idx] == 1])
        false_pos = len([idx for idx in range(int((decile+1)*num_users/10))                         if list_temp_predicted[idx] == 1 and list_temp_actual[idx] == 0])
        false_neg = len([idx for idx in range(int((decile+1)*num_users/10))                         if list_temp_predicted[idx] == 0 and list_temp_actual[idx] == 1])
        gain_analysis['Precission'][decile] = true_pos/(true_pos+false_pos)
        gain_analysis['Recall'][decile] = true_pos/(true_pos+false_neg)
        gain_analysis['F1_Score'][decile] = 2*(gain_analysis['Recall'][decile] * gain_analysis['Precission'][decile])/                            (gain_analysis['Recall'][decile] + gain_analysis['Precission'][decile])
        gain_analysis['Lift'][decile] = gain_analysis['actual_churn'][decile]/gain_analysis['num_cust'][decile]*10
        
    return gain_analysis


# Gain Analysis: Logistic Regression
predicted_model_lr = best_model_lr.transform(data)
gain_analysis_lr = gain_chart(predicted_model_lr)


# Gain Analysis: Random Forest
predicted_model_rf = best_model_rf.transform(data)
gain_analysis_rf = gain_chart(predicted_model_rf)


# Model Evaluation
plt.plot(range(10,110,10),gain_analysis_lr['actual_churn'], label = 'Logistic Regression', color = 'green', alpha = 0.9)
plt.plot(range(10,110,10),gain_analysis_rf['actual_churn'], label = 'Random Forest', color = 'green', alpha = 0.3)
plt.legend()
plt.title('Gain Chart')
plt.xlabel('% of Dataset')
plt.ylabel('% of Churn')
plt.savefig(path_FIGURE_DIR+'Gain Chart')
plt.show()


# Precision, Recall and F1 Score: Random Forest
pd.DataFrame(gain_analysis_rf)


# Precision, Recall and F1 Score: Logistic Regression
pd.DataFrame(gain_analysis_lr)


# Model Paramters: 
model_params = best_model_rf.extractParamMap()
model_parameters = {}
model_parameters['MAX_DEPTH'] = model_params[best_model_rf.maxDepth]
model_parameters['MAX_BINS'] = model_params[best_model_rf.maxBins]
model_parameters['NUM_TREES'] = model_params[best_model_rf.numTrees]
model_parameters['INSTANCES_PER_NODE'] = model_params[best_model_rf.minInstancesPerNode]

with open(os.path.join(config['paths']['MODEL_DIR'],config['paths']['MODEL_PARAMS']), 'w', encoding = 'utf-8') as file:
    yaml.dump(model_parameters, file, default_flow_style = False)

