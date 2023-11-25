# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Logistic Regression were trained.

* Model version: 1.0.0
* Model date: 26 November 2023

## Intended Use
The model is capable of predicting income classes in census data, specifically distinguishing between two categories: those earning over 50K and those earning 50K or less. This task involves binary classification.

## Training Data
The UCI Census Income Data Set served as the training data. 
Additional details about the dataset can be accessed at https://archive.ics.uci.edu/ml/datasets/census+income. In the training phase, 80% of the 32,561 rows, equivalent to 26,561 instances, were utilized for the training set.

## Evaluation Data
During the evaluation phase, 20% of the 32,561 rows, totaling 6,513 instances, were employed for the test set.

## Metrics
Three metrics were used for model evaluation (performance on test set):
* precision: 0.6803418803418804
* recall: 0.25810635538262
* fbeta: 0.3742360131640809

## Ethical Considerations
Given that the dataset comprises publicly available data featuring highly aggregated census information, there is no need to address concerns regarding harmful unintended use of the data.
## Caveats and Recommendations
Conducting hyperparameter optimization would be valuable to enhance the performance of the model.