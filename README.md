# Build an ML Pipeline for Short-Term Rental Prices in NYC

This project builds the ML pipeline for bettter prediction accuracy using random forest model. This pipeline is built by Weights & Biases (wandb). 

The Weights & Biases project is under https://wandb.ai/vickyliau/nyc_airbnb_updated

## ML pipeline:
![alt text](https://github.com/vickyting0910/nd0821-c2-build-model-workflow-starter/blob/master/images/pipeline.JPG)

## Data Cleaning:
1. test_column_presence_and_type
2. test_class_names
3. test_column_ranges
4. test_kolmogorov_smirnov
5. test_row_count
6. test_price_range

## MAE of Hyperparameter Tuning
### Training
![alt text](https://github.com/vickyting0910/nd0821-c2-build-model-workflow-starter/blob/master/images/MAE_dataset1_training.JPG)
### Testing
![alt text](https://github.com/vickyting0910/nd0821-c2-build-model-workflow-starter/blob/master/images/MAE_dataset1_testing.JPG)

## Furture Extensions for future release
### Test with different models, such gradient boosting model: 

Instead of reducing variances, reducing bias may work better. 

### Apply Bayesian Optimization for hyperparameter tuning

### Apply more data cleaning steps, such as more imputation methods

## Refereences
The code is revised from two sources:
1. nd0821-c2-build-model-workflow-exercises/ https://github.com/udacity/nd0821-c2-build-model-workflow-exercises

2. nd0821-c2-build-model-workflow-starter/ https://github.com/udacity/nd0821-c2-build-model-workflow-starter

