# Credit_Risk_Analysis
## Overview
Using a credit card dataset from LendingClub, we will be applying different machine learning algorithms to give predictions on classifying a client as being a high-risk or low-risk loan. Because this dataset contains unbalanced classifications, we will evaluate which of these models and sampling techniques performs best by looking at their confusion matrices and balanced accuracy scores. 

## Results
### Logistic Regression
The classification imbalance within the data set is 68,470 low-risk and 347 high-risk loans. When the training set is produced, there are 51,340 low-risk loans and 272 high-risk loans. We will apply four different sampling techniques in a logistic regression model and compare the results.

- Random Oversampling {'low_risk': 51340, 'high_risk': 51340}

![random_oversampling.png](https://github.com/rptseng/Credit_Risk_Analysis/blob/main/images/random_oversample.png)

- SMOTE oversampling {'low_risk': 51340, 'high_risk': 51340}

![smote.png](https://github.com/rptseng/Credit_Risk_Analysis/blob/main/images/smote.png)

- ClusterCentroids Undersampling {'high_risk': 272, 'low_risk': 272}

![clustercentroids.png](https://github.com/rptseng/Credit_Risk_Analysis/blob/main/images/clustercentroids.png)

- SMOTEENN resampling {'high_risk': 68458, 'low_risk': 62022}

![smoteenn.png](https://github.com/rptseng/Credit_Risk_Analysis/blob/main/images/smoteenn.png)

The results of Logistic Regression after these various resampling techniques do not produce very high accuracy scores. The best result is the SMOTEENN resampling with an accuracy score of 63%, while the worst performing result is ClusterCentroids undersampling with an accuracy score of 52%.

### Ensemble Learners
We will assess two Ensemble classifiers to see whether they more accurately predict credit risk than Logistic regression:

- Balanced Random Forest Classifier

![balanced_random_forest.png](https://github.com/rptseng/Credit_Risk_Analysis/blob/main/images/balanced_random_forest.png)

- Easy Ensemble Ada Boost Classifier

![easy_ensemble.png](https://github.com/rptseng/Credit_Risk_Analysis/blob/main/images/easy_ensemble.png)

The Random Forest model produces an accuracy score of 79%, while the Easy Ensemble Ada Boost Classifier produces an accuracy score of 91%. This would suggest the best model for predicting credit risk is the Easy Ensemble Ada Boost Classifier.

## Summary
Summary of the accuracy results:

- Logistic Regression
    - Random Oversampling - 61%
    - SMOTE - 60%
    - ClusterCentroid Undersampling - 52%
    - SMOTEENN - 63%
- Balanced Random Forest - 79%
- Ensemble Ada Boost - 91%

The recommended model for predicting credit risk is the Ensemble Ada Boost Classifier. It tested with 91% accuracy with an F-score of 0.97, meaning there is no large difference between precision and recall within this model.
