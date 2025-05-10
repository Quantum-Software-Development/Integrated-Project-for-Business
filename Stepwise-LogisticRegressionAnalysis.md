

Stepwise Logistic Regression Analysis

This repository demonstrates the application of stepwise logistic regression for binary classification tasks. It encompasses data loading, feature selection, model fitting, evaluation, and interpretation.

Table of Contents
	•	Overview
	•	Dataset
	•	Stepwise Feature Selection
	•	Model Fitting
	•	Model Evaluation
	•	Model Interpretation
	•	Conclusion

Overview

Stepwise regression is a method for building a regression model by adding or removing predictors based on their statistical significance. This approach helps in identifying a subset of variables that contribute most to the predictive power of the model.

Dataset

Ensure that your dataset (e.g., binary.csv) is placed in the same directory as the notebook or script. The dataset should contain both the independent variables (features) and the dependent variable (target).

import pandas as pd

# Load the dataset
data = pd.read_csv('binary.csv')

Stepwise Feature Selection

We apply the stepwise selection method to identify significant features for the logistic regression model.

# Apply the stepwise selection method
selected_features = stepwise_selection(X_train, y_train)
print("\nSelected variables by Stepwise method:")
print(selected_features)

Note: Implement the stepwise_selection function or use existing libraries that support stepwise regression.

Model Fitting

Using the selected features, we fit the logistic regression model.

import statsmodels.api as sm

# Prepare the training and testing data with selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Fit the logistic regression model
model_final = sm.Logit(y_train, X_train_selected).fit()

# Display the summary of the final model
print("\nFinal Model Summary:")
print(model_final.summary())

Model Evaluation

Evaluate the model’s performance using various metrics.

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Predict probabilities on the test set
y_pred = model_final.predict(X_test_selected)
y_pred_class = (y_pred > 0.5).astype(int)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_class))

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot the ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='cyan', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.style.use('dark_background')  # Dark mode theme
plt.show()

Model Interpretation

Final Model Equation

The logistic regression equation is as follows:

# Display the final model equation
print("\nFinal Model Equation:")
print(f"logit(P) = {model_final.params[0]:.4f} ", end="")
for i, (var, coef) in enumerate(model_final.params[1:].items(), start=1):
    print(f"+ ({coef:.4f} * {var}) ", end="")
print()

Where:
	•	logit(P) is the log-odds of the probability of the event.
	•	Coefficients (coef) indicate the change in the log-odds for a one-unit change in the predictor variable.

Coefficients (β)
	•	Positive Coefficient: Indicates that as the predictor increases, the probability of the event (e.g., admission) increases.
	•	Negative Coefficient: Indicates that as the predictor increases, the probability of the event decreases.

P-values (P>|z|)
	•	P-value < 0.05: The predictor is statistically significant.
	•	P-value ≥ 0.05: The predictor is not statistically significant and may be considered for removal from the model.

Pseudo R-squared

Pseudo R-squared values provide a measure of model fit for logistic regression models. Common types include:
	•	McFadden’s R²: Values between 0.2 and 0.4 indicate excellent fit.
	•	Cox and Snell R²: Adjusts the likelihood ratio to mimic the R² in linear regression.
	•	Nagelkerke R²: A modification of Cox and Snell R² that adjusts the scale to cover the full range from 0 to 1.

Note: Pseudo R-squared values are not directly comparable to R² in linear regression and should be interpreted with caution.

Conclusion

This analysis demonstrates the application of stepwise logistic regression for feature selection and model building. By evaluating model performance through various metrics, we ensure the reliability and validity of the predictive model.

