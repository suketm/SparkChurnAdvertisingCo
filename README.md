
# Advertising Company Churn Modelling

A marketing agency has many customers that use their service to produce ads for the client/customer websites. They've noticed that they have quite a bit of churn in clients.
*This porject is to help Advertising Co develope a churn propensity model and find a suitable business solution for the churn problem.*


## File structure
**Configuration file:** /config/config.yaml

**Python virtual environment setting file:** environment.yml

**Python files**
1. Library for preparing data for modeling and inference
    ```
    /src/utils/ml_helper.py
    ```
2. Library for modelling
    ```
    /src/utils/ml_helper.py
    ```
3. Script to train models
    ```
    /src/train.py
    ```
4. Script to make predictions
    ```
    /src/churn_predict.py
    ```

## Flow to execute code for modelling:
1. Create a virtual environment and activate the environment
    ```
    conda env create -f environment.yml
    activate advertiseco
    ```
2. Train models on the training dataset (run under '/src/' directory)
    ```
    python train.py ../config/config.yaml
    ```
3. Make predictions using the models built in the previous step, output file is stored in **'/reports/new_customers_predictions/'**
    ```
    python churn_predict.py ../config/config.yaml
    ```