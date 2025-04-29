

## Step 1: Import Necessary Libraries

# Importing essential libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Setting dark mode for plots

plt.style.use('dark_background')
sns.set_palette("deep")

## Step 2: Load the Dataset

# Loading the dataset

# Replace the file path with the correct one if necessary

dados = pd.read_excel('/Users/fabicampanari/Desktop/class_6-Lasso Regression/project_6-Lasso-Roidge-Elastic-Regression/Imoveis (1).xlsx')

# Displaying the first few rows of the dataset

print(dados.head())

## Step 3: Preprocess the Data

# Separating predictors (X) and the target variable (y)

X = dados.drop(columns=['Valor'])  \# Replace 'Value' with the actual column name for the target variable
y = dados['Valor']

# Splitting the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the predictors

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Step 4: Lasso Regression

# Fitting a Lasso Regression model

lasso = Lasso(alpha=0.1, random_state=42)  \# Adjust alpha as needed
lasso.fit(X_train_scaled, y_train)

# Making predictions

y_pred_lasso = lasso.predict(X_test_scaled)

# Evaluating the model

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# Printing results

print(f"Lasso Regression - MSE: {mse_lasso}, R2: {r2_lasso}")

## Step 5: Ridge Regression

# Fitting a Ridge Regression model

ridge = Ridge(alpha=0.1, random_state=42)  \# Adjust alpha as needed
ridge.fit(X_train_scaled, y_train)

# Making predictions

y_pred_ridge = ridge.predict(X_test_scaled)

# Evaluating the model

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Printing results

print(f"Ridge Regression - MSE: {mse_ridge}, R2: {r2_ridge}")

## Step 6: Elastic Net Regression

# Fitting an Elastic Net model

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)  \# Adjust alpha and l1_ratio as needed
elastic_net.fit(X_train_scaled, y_train)

# Making predictions

y_pred_elastic = elastic_net.predict(X_test_scaled)

# Evaluating the model

mse_elastic = mean_squared_error(y_test, y_pred_elastic)
r2_elastic = r2_score(y_test, y_pred_elastic)

# Printing results

print(f"Elastic Net Regression - MSE: {mse_elastic}, R2: {r2_elastic}")

## Step 7: Compare Results

# Comparing the results of the three models

results = pd.DataFrame({
'Model': ['Lasso', 'Ridge', 'Elastic Net'],
'MSE': [mse_lasso, mse_ridge, mse_elastic],
'R2': [r2_lasso, r2_ridge, r2_elastic]
})

print(results)

## Step 8: Visualize Results

# Plotting the comparison of MSE

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MSE', data=results)
plt.title('Comparison of MSE Across Models', fontsize=16)
plt.xlabel('Regression Model', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.show()

# Plotting the comparison of R2

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2', data=results)
plt.title('Comparison of R2 Across Models', fontsize=16)
plt.xlabel('Regression Model', fontsize=14)
plt.ylabel('R2 Score', fontsize=14)
plt.show()

# Lasso, Ridge, and Elastic Net Regression Comparison

This project demonstrates how to apply and compare three popular linear regression techniques-**Lasso**, **Ridge**, and **Elastic Net**-using Python. The models are evaluated using a real estate dataset to predict property values.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Code Structure](#code-structure)
- [Results \& Visualization](#results--visualization)
- [License](#license)

---

## Overview

The goal of this project is to:

- Load and preprocess a real estate dataset.
- Train and evaluate Lasso, Ridge, and Elastic Net regression models.
- Compare their performance using Mean Squared Error (MSE) and R² score.
- Visualize the results for easy interpretation.

---

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```


---

## Dataset

- The dataset should be in Excel format (`.xlsx`) and must contain a column named `Valor` (the target variable).
- Update the file path in the code as needed:

```python
dados = pd.read_excel('path/to/your/Imoveis.xlsx')
```


---

## How to Run

1. **Clone this repository or copy the code.**
2. **Ensure your dataset is available and the path is correct.**
3. **Install the required libraries.**
4. **Run the script:**

```bash
python your_script_name.py
```


---

## Code Structure

The script follows these steps:

### 1. Import Libraries

Imports all necessary Python libraries for data manipulation, modeling, and visualization.

### 2. Load the Dataset

Reads the Excel dataset and displays the first few rows.

### 3. Preprocess the Data

- Separates predictors (`X`) and the target variable (`y`).
- Splits data into training and testing sets.
- Standardizes the features.


### 4. Lasso Regression

- Trains a Lasso regression model.
- Makes predictions and evaluates performance.


### 5. Ridge Regression

- Trains a Ridge regression model.
- Makes predictions and evaluates performance.


### 6. Elastic Net Regression

- Trains an Elastic Net model.
- Makes predictions and evaluates performance.


### 7. Compare Results

- Collects and prints the MSE and R² scores for all models.


### 8. Visualize Results

- Plots bar charts comparing the MSE and R² scores of the three models.

---

## Results \& Visualization

The script outputs:

- The MSE and R² scores for each model.
- Bar plots comparing the performance of Lasso, Ridge, and Elastic Net regression.

---

## License

This project is provided for educational purposes.

---

## Full Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

plt.style.use('dark_background')
sns.set_palette("deep")

# Load the dataset
dados = pd.read_excel('/Users/fabicampanari/Desktop/class_6-Lasso Regression/project_6-Lasso-Roidge-Elastic-Regression/Imoveis (1).xlsx')
print(dados.head())

# Preprocess the data
X = dados.drop(columns=['Valor'])
y = dados['Valor']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso Regression
lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f"Lasso Regression - MSE: {mse_lasso}, R2: {r2_lasso}")

# Ridge Regression
ridge = Ridge(alpha=0.1, random_state=42)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"Ridge Regression - MSE: {mse_ridge}, R2: {r2_ridge}")

# Elastic Net Regression
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X_train_scaled, y_train)
y_pred_elastic = elastic_net.predict(X_test_scaled)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)
r2_elastic = r2_score(y_test, y_pred_elastic)
print(f"Elastic Net Regression - MSE: {mse_elastic}, R2: {r2_elastic}")

# Compare Results
results = pd.DataFrame({
    'Model': ['Lasso', 'Ridge', 'Elastic Net'],
    'MSE': [mse_lasso, mse_ridge, mse_elastic],
    'R2': [r2_lasso, r2_ridge, r2_elastic]
})
print(results)

# Visualize Results
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MSE', data=results)
plt.title('Comparison of MSE Across Models', fontsize=16)
plt.xlabel('Regression Model', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2', data=results)
plt.title('Comparison of R2 Across Models', fontsize=16)
plt.xlabel('Regression Model', fontsize=14)
plt.ylabel('R2 Score', fontsize=14)
plt.show()
```



