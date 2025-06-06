<br>


<div align="center">
  <h1 style="font-size:2.5em;">🌟 Integrated Business Project – 3rd Semester at PUC-SP: Bachelor's in Humanistic AI & Data Science</h1>
  <h3 style="font-size:0.9em;">
    Under the guidance of <a href="https://www.linkedin.com/in/eric-bacconi-423137/" target="_blank" style="color:inherit; text-decoration:underline;">Professor Dr. Eric Bacconi</a>, Coordinator of the Bachelor's Program in Humanistic AI & Data Science at PUC-SP.
  </h3>
</div>


<br><br><br>

<h2 align="center">  $$\Huge {\textbf{\color{DodgerBlue} GOOD DECISIONS = GOOD RESULTS}}$$ 

<br><br><br>

### <p align="center"> [![Sponsor Quantum Software Development](https://img.shields.io/badge/Sponsor-Quantum%20Software%20Development-brightgreen?logo=GitHub)](https://github.com/sponsors/Quantum-Software-Development)

<br><br><br>


# Table of Contents  
1. [Linear Regression and Data Scaling Analysis - Normalization - Scaling](#linear-regressio-data-scalinganalysis-normalization-scaling)  
2. [Practical Example for Calculating this Normalized Value in Python](#practical-example-calculating-normalized-value-python)  
3. [Pratical Example for Calculating this Normalized Value in Excel](#pratical-example-Calculating-normalized-value-excel)
4. [Range Standardization](#range-standardization)
5. [Pearson Correlation](#pearson-correlation)  
6. [Linear Regression: Price Prediction Case Study](#linear-regression-price-prediction-case-study-)  
   - [I. Use Case Implementation](#i-use-case-implementation)  
   - [Dataset Description](#dataset-description)  
   - [II. Methodology](#ii-methodology)  
   - [Stepwise Regression Implementation](#stepwise-regression-implementation)  
   - [III. Statistical Analysis](#iii-statistical-analysis)  
   - [Key Metrics Table](#key-metrics-table)  
   - [Correlation Matrix](#correlation-matrix)  
   - [IV. Full Implementation Code](#iv-full-implementation-code)  
   - [Model Training & Evaluation](#model-training--evaluation)  
   - [ANOVA Results](#anova-results)  
   - [V. Visualization](#v-visualization)  
   - [Actual vs Predicted Prices](#actual-vs-predicted-prices)  
   - [VI. How to Run](#vi-how-to-run)  
7. [Multiple Linear Regression Analysis Report](#multiple-linear-regression-analysis-report)  
   - [Dataset Overview](#dataset-overview)  
   - [Key Formulas](#key-formulas)  
   - [Statistical Results](#statistical-results)  
   - [Code Implementation](#code-implementation)  
   - [Stepwise Regression](#stepwise-regression)
8. [Logistic Regression](#logistic-regression)
9. [Discriminant Analysis](#discriminant-analysis)
10. [Lasso, Ridge and Elastic Net Regression](#lasso-ridge-elastic-net-regression)
    - [Compare Results: Lasso, Ridge and Elastic Net Regression](#compare-results-lasso-ridge-elastic-net-regression)
11. [Brains Made of Code: Regression Training with Gradient Descent and Stochastic Optimization Algorithms](#brains-made-code-regression-training-gradient-descent-stochastic-optimization-algorithms)
    - [Build a Brain NVIDEA](build-a-brain-nvidea)
12. [Bayesian- KNN Regression - Model Persistence](#bayesian-knn-regression-model-Persistence)
13. [Projects](#projects)

<br><br>


# [1.]() - Linear Regression and Data Scaling Analysis - Normalization - Scaling

<br>

### [Project Overview]()

This project demonstrates a complete machine learning workflow for price prediction using:
- **Stepwise Regression** for feature selection  
- Advanced statistical analysis (ANOVA, R² metrics)  
- Full model diagnostics  
- Interactive visualization integration

➢➣ [Get access](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/tree/c751c1eb34ccfe30e451954c9a5ae5ea3f52e70d/Projects) to all the projects + dataset developed in this discipline.

<br>



#### [Standardization of a Range of Values]()

It's describes the process of scaling or normalizing data within a specific range, typically to a standardized scale, for example, from 0 to 1. This is a common technique in data analysis and machine learning.

<br>

###  <p align="center"> [Mathematical Formula]()

<br>

$$X_{normalized} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

<br>

#### <p align="center"> [Where]():

 $$X_{\max} - X_{\min} = \text{Amplitude}$$ 

 <br>

####  <p align="center"> Is the `amplitude`, a way to represent the range of data values before normalization.

<br>

### [Explanation]():

To calculate the standardization of the variables salario, n_filhos, and idade using both the Z-Score and Range methods, and to evaluate the mean, standard deviation, maximum, and minimum before and after standardization, we can follow these steps:


### [Before Standardization]():

Compute the mean, standard deviation, maximum, and minimum for each of the variables (n_filhos, salario, idade).

### [Z-Score Standardization]():

We standardize the variables using the Z-Score method, which is computed as:


$Z$ = $\frac{X - \mu}{\sigma}$

```latex
Z = \frac{X - \mu}{\sigma}
```

Where:
- $\( \mu \)$ is the mean,
- $\( \sigma \)$ is the standard deviation.

  <br>

### [Range Standardization (Min-Max Scaling)]():

We scale the data using the Min-Max method, which scales the values to a [0, 1] range using:

$X'$ = $\frac{X - \min(X)}{\max(X) - \min(X)}$

```latex
X' = \frac{X - \min(X)}{\max(X) - \min(X)}
```
  
Where:
- X is the original value,
- min(X) is the minimum value,
- max(X) is the maximum value.

<br>

### [After Standardization]():

Compute the mean, standard deviation, maximum, and minimum of the standardized data for both Z-Score and Range methods.

The output will provide the descriptive statistics before and after each standardization method, allowing you to compare the effects of Z-Score and Range standardization on the dataset.

 <br>

## [2.]() Practical Example for Calculating this Normalized Value in Python:

#### Use this [dataset](https://github.com/Quantum-Software-Development/Integrated_Project-Business/blob/f2d7abe6ee5853ae29c750170a01e429334f6fe5/HomeWork/1-Z-Score-Range/cadastro_funcionarios.xlsx)

The code demonstrates how to apply Z-Score and Range (Min-Max) standardization to the variables salario, n_filhos, and idade in a dataset. It also evaluates and compares the mean, standard deviation, minimum, and maximum values before and after the standardization methods are applied.

 <br>

#### Cell 1: [Import necessary libraries]()

```python
# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
```

<br>

#### Cell 2: [Load the dataset from the Excel file]()

```python
# Load the data from the Excel file
# df = pd.read_excel('use-your-own-dataset.xlsx') - optional
df = pd.read_excel('cadastro_funcionarios.xlsx')
df.head()  # Displaying the first few rows of the dataset to understand its structure
```

<br>

#### Cell 3: [Evaluate the statistics before standardization]()

```python
# Step 1: Evaluate the mean, std, max, and min before standardization
before_std_stats = {
    'mean_n_filhos': df['n_filhos'].mean(),
    'std_n_filhos': df['n_filhos'].std(),
    'min_n_filhos': df['n_filhos'].min(),
    'max_n_filhos': df['n_filhos'].max(),
    
    'mean_salario': df['salario'].mean(),
    'std_salario': df['salario'].std(),
    'min_salario': df['salario'].min(),
    'max_salario': df['salario'].max(),
    
    'mean_idade': df['idade'].mean(),
    'std_idade': df['idade'].std(),
    'min_idade': df['idade'].min(),
    'max_idade': df['idade'].max(),
}

# Display the statistics before standardization
before_std_stats
```

<br>

#### Cell 4: [Apply Z-Score standardization]()

```python
# Step 2: Z-Score Standardization
df_zscore = df[['n_filhos', 'salario', 'idade']].apply(lambda x: (x - x.mean()) / x.std())

# Display the standardized data
df_zscore.head()
```

<br>

#### Cell 5: [Evaluate the statistics after Z-Score standardization]()

```python
# Step 3: Evaluate the mean, std, max, and min after Z-Score standardization
after_zscore_stats = {
    'mean_n_filhos_zscore': df_zscore['n_filhos'].mean(),
    'std_n_filhos_zscore': df_zscore['n_filhos'].std(),
    'min_n_filhos_zscore': df_zscore['n_filhos'].min(),
    'max_n_filhos_zscore': df_zscore['n_filhos'].max(),
    
    'mean_salario_zscore': df_zscore['salario'].mean(),
    'std_salario_zscore': df_zscore['salario'].std(),
    'min_salario_zscore': df_zscore['salario'].min(),
    'max_salario_zscore': df_zscore['salario'].max(),
    
    'mean_idade_zscore': df_zscore['idade'].mean(),
    'std_idade_zscore': df_zscore['idade'].std(),
    'min_idade_zscore': df_zscore['idade'].min(),
    'max_idade_zscore': df_zscore['idade'].max(),
}

# Display the statistics after Z-Score standardization
after_zscore_stats
```

<br>

#### Cell 6: [Apply Range Standardization]() (Min-Max Scaling)

```python
# Step 4: Range Standardization (Min-Max Scaling)
scaler = MinMaxScaler()
df_range = pd.DataFrame(scaler.fit_transform(df[['n_filhos', 'salario', 'idade']]), columns=['n_filhos', 'salario', 'idade'])

# Display the scaled data
df_range.head()
```

<br>

#### Cell 7: [Evaluate the statistics after Range standardization]()

```python
# Step 5: Evaluate the mean, std, max, and min after Range standardization
after_range_stats = {
    'mean_n_filhos_range': df_range['n_filhos'].mean(),
    'std_n_filhos_range': df_range['n_filhos'].std(),
    'min_n_filhos_range': df_range['n_filhos'].min(),
    'max_n_filhos_range': df_range['n_filhos'].max(),
    
    'mean_salario_range': df_range['salario'].mean(),
    'std_salario_range': df_range['salario'].std(),
    'min_salario_range': df_range['salario'].min(),
    'max_salario_range': df_range['salario'].max(),
    
    'mean_idade_range': df_range['idade'].mean(),
    'std_idade_range': df_range['idade'].std(),
    'min_idade_range': df_range['idade'].min(),
    'max_idade_range': df_range['idade'].max(),
}

# Display the statistics after Range standardization
after_range_stats
```

<br>

## [3.]()  Pratical Example for Calculating this Normalized Value in [Excel

#### Use this [dataset](https://github.com/Quantum-Software-Development/Integrated_Project-Business/blob/f2d7abe6ee5853ae29c750170a01e429334f6fe5/HomeWork/1-Z-Score-Range/cadastro_funcionarios.xlsx)

To standardize the variables (salary, number of children, and age) in Excel using the Z-Score and Range methods, you can follow these steps:

 <br>

### I. [Z-Score Standardization]()

### Steps for Z-Score in Excel:

##### 1. [Find the Mean (µ)]():

Use the AVERAGE function to calculate the mean of the column. For example, to find the mean of the salary (column E), use:

```excel
=AVERAGE(E2:E351)
```

<br>

#### 2. [Find the Standard Deviation (σ)]():
   
Use the STDEV.P function to calculate the standard deviation of the column. For example, to find the standard deviation of the salary (column E), use:

```excel
=STDEV.P(E2:E351)
```

<br>

#### 3. [Apply the Z-Score Formula]():

For each value in the column, apply the Z-Score formula. In the first row of the new column, use:

```excel
=(E2 - AVERAGE(E$2:E$351)) / STDEV.P(E$2:E$351)
```

<br>

#### 4.[Drag the formula down to calculate the Z-Score for all the rows]():

Example for Salary:

In cell H2 (new column for standardized salary), write

```excel
=(E2 - AVERAGE(E$2:E$351)) / STDEV.P(E$2:E$351)
```

Then, drag it down to the rest of the rows.

Repeat the same steps for the variables n_filhos (column D) and idade (column F).


<br>

## [4.]() Range Standardization

Steps for Range Standardization in Excel:

#### 1. [Find the Min and Max]():

Use the MIN and MAX functions to find the minimum and maximum values of the column. For example, to find the min and max of salary (column E), use:

```excel
=MIN(E2:E351)
=MAX(E2:E351)
```

<br>

#### 2. [Apply the Range Formula]():

For each value in the column, apply the range formula. In the first row of the new column, use:

```excel
=(E2 - MIN(E$2:E$351)) / (MAX(E$2:E$351) - MIN(E$2:E$351))
```

<br>

#### 3.[Drag the formula down to calculate the range standardized values for all the rows]():

Example for Salary:

In cell I2 (new column for range standardized salary), write:

```excel
=(E2 - MIN(E$2:E$351)) / (MAX(E$2:E$351) - MIN(E$2:E$351))
```

Then, drag it down to the rest of the rows.
Repeat the same steps for the variables n_filhos (column D) and idade (column F).

<br>


### [Summary of the Process]():

[Z-Score Standardization]() centers the data around [zero]() and scales it based on the [standard deviation]().

[Range Standardization (Min-Max Scaling)]() rescals the data to a [[0, 1] range]().

Both techniques were applied (given dataset)  to the [columns n_filhos](), [salario](), and [idade](), and the statistics (mean, std, min, max) were calculated before and after the standardization methods.


<br>

### [Important Notes]():

- **Correlation does not imply causation**: Correlation between two variables does not necessarily mean that one causes the other. For example, there may be a correlation between the number of salespeople in a store and increased sales, but that does not imply that having more salespeople directly causes higher sales.

- **Using regressions we don’t need to worry about standardization**: When using regressions, there is no need to worry about data standardization. Unlike other methods like k-NN or neural networks, where the scale of the data can impact performance, regression models can be applied directly without the need for scaling adjustments.

<br>

## [5.]() Pearson Correlation

**Pearson Correlation** is a statistical measure that describes the strength and direction of a linear relationship between two variables. The Pearson correlation value ranges from -1 to 1:

- **1**: Perfect positive correlation (both variables increase together).
- **-1**: Perfect negative correlation (one variable increases while the other decreases).
- **0**: No linear correlation.

For example, if we're analyzing the correlation between the area of a house and its price, a Pearson value close to 1 would indicate that as the area increases, the price also tends to increase.

<br>

## [6.]() Simple Linear Regression

**Simple Linear Regression** is a statistical model that describes the relationship between a dependent variable (response) and an independent variable (predictor). The model is represented by the formula:

$$
y = \beta_0 + \beta_1 \cdot x
$$

Where:
- \(y\) is the dependent variable (the one we want to predict),
- \(x\) is the independent variable (the one used to make predictions),
- \(\beta_0\) is the intercept (the value of \(y\) when \(x = 0\)),
- \(\beta_1\) is the coefficient (representing the change in \(y\) when \(x\) increases by one unit).

Simple linear regression is widely used for predicting a value based on a linear relationship between variables.

### Steps to Perform Linear Regression:

1. **Data Collection**: Gather the data related to the dependent and independent variables.
2. **Exploratory Data Analysis (EDA)**: Explore the data to identify trends, patterns, and check correlation.
3. **Model Fitting**: Fit the linear regression model to the data using a method like Ordinary Least Squares (OLS).
4. **Model Evaluation**: Evaluate the model performance using metrics like Mean Squared Error (MSE) and the Coefficient of Determination (\(R^2\)).
5. **Prediction**: Use the fitted model to make predictions with new data.

Simple linear regression is a great starting point for predictive problems where a linear relationship between variables is expected.


### 1- Example Code - [Correlation Vendas Gjornal]()

### Use This Dataset - [BD Gerais.xlsx](https://github.com/Quantum-Software-Development/Integrated_Project-Business/blob/4331d9227118d2025a6c167a3cefd99bf7404939/class_2-Linear%20Regression/BD%20Gerais.xlsx)



### Step 1: [Install Required Libraries]()

If you don't have the required libraries installed, you can install them with pip:

<br>

```python
pip install pandas numpy matplotlib scikit-learn openpyxl
```

<br>

### Step 2: [Run rhis Script]()

<br>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from the Excel file
file_path = 'BD Gerais.xlsx'
df = pd.read_excel(file_path)

# Display the first few rows of the dataset
print(df.head())

# Let's assume the columns are: 'Vendas', 'Gjornal', 'GTV', 'Gmdireta'
# Compute the correlation matrix
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Perform linear regression: Let's use 'Vendas' as the target and 'Gjornal', 'GTV', 'Gmdireta' as features
X = df[['Gjornal', 'GTV', 'Gmdireta']]  # Features
y = df['Vendas']  # Target variable

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Print the regression coefficients
print("\nRegression Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Make predictions
y_pred = model.predict(X)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the actual vs predicted values
plt.scatter(y, y_pred)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.title('Actual vs Predicted Vendas')
plt.xlabel('Actual Vendas')
plt.ylabel('Predicted Vendas')
plt.show()

# Plot the regression line for each feature vs 'Vendas'
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot for 'Gjornal'
axs[0].scatter(df['Gjornal'], y, color='blue')
axs[0].plot(df['Gjornal'], model.intercept_ + model.coef_[0] * df['Gjornal'], color='red')
axs[0].set_title('Gjornal vs Vendas')
axs[0].set_xlabel('Gjornal')
axs[0].set_ylabel('Vendas')

# Plot for 'GTV'
axs[1].scatter(df['GTV'], y, color='blue')
axs[1].plot(df['GTV'], model.intercept_ + model.coef_[1] * df['GTV'], color='red')
axs[1].set_title('GTV vs Vendas')
axs[1].set_xlabel('GTV')
axs[1].set_ylabel('Vendas')

# Plot for 'Gmdireta'
axs[2].scatter(df['Gmdireta'], y, color='blue')
axs[2].plot(df['Gmdireta'], model.intercept_ + model.coef_[2] * df['Gmdireta'], color='red')
axs[2].set_title('Gmdireta vs Vendas')
axs[2].set_xlabel('Gmdireta')
axs[2].set_ylabel('Vendas')

plt.tight_layout()
plt.show()
````

#

### 2- Example Code - [Correlation Vendas -  GTV]()

### Use This Dataset - [BD Gerais.xlsx](https://github.com/Quantum-Software-Development/Integrated_Project-Business/blob/4331d9227118d2025a6c167a3cefd99bf7404939/class_2-Linear%20Regression/BD%20Gerais.xlsx)

To compute the correlation between the Vendas and GTV columns in your dataset using Python, you can follow this code. This will calculate the correlation coefficient and visualize the relationship between these two variables using a scatter plot.

```python
import pandas as pd
import matplotlib.pyplot as plt
```
  
<br>

## [7.]() - Multiple Linear Regression with 4 variable

- Vendas as the dependent variable (Y)
  
- Jornal, GTV, and Gmdireta as independent variables (X)
  
This code will also calculate the correlation matrix, fit the multiple linear regression model, and display the regression results.

#### Python Code for Multiple Linear Regression and Correlation

<br>

### 1- Install Required Libraries (if you don't have them yet)

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn
```

<br>

### 2- Python Code

<br>

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from the Excel file
file_path = 'BD Gerais.xlsx'  # Adjust the file path if needed
df = pd.read_excel(file_path)

# Display the first few rows of the dataset to verify the data
print(df.head())

# Calculate the correlation matrix for the variables
correlation_matrix = df[['Vendas', 'Gjornal', 'GTV', 'Gmdireta']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Define the independent variables (X) and the dependent variable (Y)
X = df[['Gjornal', 'GTV', 'Gmdireta']]  # Independent variables
y = df['Vendas']  # Dependent variable (Vendas)

# Add a constant (intercept) to the independent variables
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Display the regression results
print("\nRegression Results:")
print(model.summary())

# Alternatively, using sklearn's LinearRegression to calculate the coefficients and R-squared
model_sklearn = LinearRegression()
model_sklearn.fit(X[['Gjornal', 'GTV', 'Gmdireta']], y)

# Coefficients and intercept
print("\nLinear Regression Coefficients (sklearn):")
print("Intercept:", model_sklearn.intercept_)
print("Coefficients:", model_sklearn.coef_)

# Predicting with the model
y_pred = model_sklearn.predict(X[['Gjornal', 'GTV', 'Gmdireta']])

# Calculating R-squared and Mean Squared Error (MSE)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"\nR-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# Plotting the actual vs predicted Vendas
plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k', color='red')  # line of perfect prediction
plt.xlabel('Actual Vendas')
plt.ylabel('Predicted Vendas')
plt.title('Actual vs Predicted Vendas')
plt.show()
```

<br>

## [Code Explanation]():

Loading Data:

The dataset is loaded from BD Gerais.xlsx using pandas.read_excel(). The file path is adjusted based on your actual file location.
Correlation Matrix:

We calculate the correlation matrix for the four variables: Vendas, Gjornal, GTV, and Gmdireta. This gives us an overview of the relationships between the variables.

### [Multiple Linear Regression]():

We define the independent variables (Gjornal, GTV, Gmdireta) as X and the dependent variable (Vendas) as y.
We add a constant term (intercept) to X using sm.add_constant() for proper regression.
We use the statsmodels.OLS method to fit the multiple linear regression model and print the regression summary, which includes coefficients, R-squared, p-values, and more.
Alternative Model (sklearn):

We also use sklearn.linear_model.LinearRegression() for comparison, which calculates the coefficients and R-squared.
We then use the trained model to predict the Vendas values and calculate Mean Squared Error (MSE) and R-squared.
Plotting:

The actual values of Vendas are plotted against the predicted values from the regression model in a scatter plot. A red line of perfect prediction is also added (this line represents the ideal case where actual values equal predicted values).

### [Output of the Code]():

### Correlation Matrix:

Displays the correlation between Vendas, Gjornal, GTV, and Gmdireta. This helps you understand the relationships between these variables.
Regression Results (from statsmodels):

The regression summary will include:
[Coefficients](): The relationship between each independent variable and the dependent variable (Vendas).
[R-squared](): Measures how well the model fits the data.
[P-values](): For testing the statistical significance of each coefficient.

<br>

### [Linear Regression Coefficients]():

<br>

- The model's intercept and coefficients are printed for comparison.

- R-squared and Mean Squared Error (MSE):

- These two metrics evaluate the performance of the regression model.

- R-squared tells you how well the model explains the variance in the dependent variable.
  
- MSE gives an idea of the average squared difference between the predicted and actual values.

  #
  
### [Plot]():

The plot shows how well the model's predicted Vendas values match the actual values.


### Example Output (Model Summary from statsmodels):

<b>

```plaintext
                            OLS Regression Results
==============================================================================
Dep. Variable:                 Vendas   R-squared:                       0.982
Model:                            OLS   Adj. R-squared:                  0.980
Method:                 Least Squares   F-statistic:                     530.8
Date:                Thu, 10 Mar 2025   Prob (F-statistic):           2.31e-14
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         12.1532      3.001      4.055      0.001       5.892      18.414
Gjornal        2.4503      0.401      6.100      0.000       1.638       3.262
GTV            1.2087      0.244      4.948      0.000       0.734       1.683
Gmdireta       0.5003      0.348      1.437      0.168      -0.190       1.191
==============================================================================
```

### [In the summary]():

R-squared of 0.982 indicates that the model explains 98.2% of the variance in `Vendas`.

The coefficients of the independent variables show how each variable affects `Vendas`.

<br>

If the p-value for `Gmdireta` is greater than 0.05, it means that `Gmdireta` is not statistically significant in explaining the variability in the dependent variable `Vendas`. In such a case, it's common practice to remove the variable from the model and perform the regression again with only the statistically significant variables.

#### In this case, [you can exclude Gmdireta and rerun the regression model using only the remaining variables](): `Gjornal` and `GTV`.

<br>

### [Why Remove Gmdireta]():

[P-value](): The `p-value` is used to test the null hypothesis that the coefficient of the variable is equal to zero (i.e., the variable has no effect). If the p-value is greater than `0.05,` it indicates that the variable is not statistically significant at the 5% level and doesn't provide much explanatory power in the model.

[Adjusted R-squared](): By removing variables that are not significant, you often improve the model's explanatory power (in some cases), as it helps reduce multicollinearity and overfitting.

<br>

### [Modified Python Code Without Gmdireta]():

Let’s update the code by removing Gmdireta from the regression model and re-running the analysis with just Gjornal and GTV as the independent variables.

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from the Excel file
file_path = 'BD Gerais.xlsx'  # Adjust the file path if needed
df = pd.read_excel(file_path)

# Display the first few rows of the dataset to verify the data
print(df.head())

# Calculate the correlation matrix for the variables
correlation_matrix = df[['Vendas', 'Gjornal', 'GTV']].corr()  # Excluding 'Gmdireta'
print("\nCorrelation Matrix (without Gmdireta):")
print(correlation_matrix)

# Define the independent variables (X) and the dependent variable (Y)
X = df[['Gjornal', 'GTV']]  # Independent variables (Gjornal and GTV)
y = df['Vendas']  # Dependent variable (Vendas)

# Add a constant (intercept) to the independent variables
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Display the regression results
print("\nRegression Results (without Gmdireta):")
print(model.summary())

# Alternatively, using sklearn's LinearRegression to calculate the coefficients and R-squared
model_sklearn = LinearRegression()
model_sklearn.fit(X[['Gjornal', 'GTV']], y)

# Coefficients and intercept
print("\nLinear Regression Coefficients (sklearn):")
print("Intercept:", model_sklearn.intercept_)
print("Coefficients:", model_sklearn.coef_)

# Predicting with the model
y_pred = model_sklearn.predict(X[['Gjornal', 'GTV']])

# Calculating R-squared and Mean Squared Error (MSE)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"\nR-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# Plotting the actual vs predicted Vendas
plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k', color='red')  # line of perfect prediction
plt.xlabel('Actual Vendas')
plt.ylabel('Predicted Vendas')
plt.title('Actual vs Predicted Vendas')
plt.show()
```
<br>

## [Key Changes]():

### [Removed Gmdireta]():

In the regression model, Gmdireta was excluded as an independent variable.
The correlation matrix is now calculated using only Vendas, Gjornal, and GTV.
Independent Variables (X):

We now use only Gjornal and GTV as the independent variables for the regression analysis.
The variable Gmdireta is no longer included in the model.
Explanation of the Code:
Correlation Matrix:

We calculate the correlation matrix to examine the relationships between Vendas, Gjornal, and GTV only (without Gmdireta).
Multiple Linear Regression (statsmodels):

We perform the Multiple Linear Regression with Gjornal and GTV as independent variables.
The regression summary will now show the coefficients, p-values, R-squared, and other statistics for the model with the reduced set of independent variables.
Linear Regression (sklearn):

We also use `sklearn.linear_model.LinearRegression()` to perform the regression and output the intercept and coefficients for the model without `Gmdireta`.

<br>

### [Prediction and Performance Metrics]():

After fitting the regression model, we calculate the predicted values [y_pred]() for `Vendas` using the new model `(without Gmdireta)`.

We calculate the [R-squared]() and [Mean Squared Error (MSE)]() to evaluate the model's performance.

The [R-squared]() tells us how much of the variance in `Vendas` is explained by `Gjornal` and `GTV`.

The [MSE]() tells us the average squared difference between the predicted and actual values.

<br>


### [Plotting]():

The plot visualizes how well the predicted `Vendas` values match the actual values. The red line represents the ideal case where the predicted values equal the actual values.

<br>

### [Example of Expected Output]() (Updated Model Summary):

<br>


```plaintext

                            OLS Regression Results
==============================================================================
Dep. Variable:                 Vendas   R-squared:                       0.976
Model:                            OLS   Adj. R-squared:                  0.974
Method:                 Least Squares   F-statistic:                     320.3
Date:                Thu, 10 Mar 2025   Prob (F-statistic):           4.23e-10
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         10.6345      2.591      4.107      0.000       5.146      16.123
Gjornal        2.8951      0.453      6.394      0.000       1.987       3.803
GTV            1.3547      0.290      4.673      0.000       0.778       1.931
==============================================================================
```

<br>

### [Interpretation]():

[R-squared](): A value of 0.976 means that the independent variables `Gjornal` and `GTV` explain 97.6% of the variance in `Vendas`, which is a good fit.

<br>

### [Coefficients]():

<br>

The coefficient for Gjornal (2.8951) tells us that for each unit increase in `Gjornal`, `Vendas` increases by approximately 2.90 units.

The coefficient for `GTV` (1.3547) tells us that for each unit increase in `GTV`, `Vendas` increases by approximately 1.35 units.

<br>

### [P-values](): Both `Gjornal` and `GTV` have very small p-values (much smaller than 0.05), indicating that they are statistically significant in predicting `Vendas`.

By removing the variable `Gmdireta` (which had a p-value greater than 0.05), the regression model now focuses on the variables that have a stronger statistical relationship with the dependent variable `Vendas`.


<br>

## [8.]() Logistic Regression]

#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/9cfecdfdd9d43e93089870e78216f1dc629702c0/class_4-Logistic%20Regression/Logistic%20Regression.pdf) to access Theoretical and Pratical Material.

#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/56a09c405eaf4e03922c8dc03c43b6e7adea64db/class_4-Logistic%20Regression/Credito.xlsx) to access the Dataset

#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/3009ac9b2abbb0eddf3ee8a28d539f62e2eb33cf/class_4-Logistic%20Regression/Regressao_Logistica%20(1).ipynb) and access  Logistic Refression Basic Code

#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/tree/9a770fb9430c5c94ea1fe8a75d09d1aed7423a9e/class_5-2nd_Exam-Logistic%20Regression) and access  Logistic Refression Exercices + Codes and Datasets



<br>

## [9.]() Discriminant Analysis - Iris

#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/5ab10f415d91d5165316facc007f06580ee3e4b7/class_6-Discriminant%20Analysis/Discriminant%20Analysis.pdf) to access Theoretical and Pratical Material.

#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/feaa91502841c85ad30ae891aa1f4ebd25927b20/class_6-Discriminant%20Analysis/Discriminant.xlsx) to access the Dataset

#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/a1d6bd1fec9a1603f60495ef27b31b1866934239/class_6-Discriminant%20Analysis/Discriminant.ipynb) and access  Discriminant  Basic Code - Iris

#### [Click here]() and access  Discriminant  Exercices + Codes and 

#### [Click here]() and access  Discriminant  Exercices in Excel

#### [Click here](https://github.com/FabianaCampanari/Iris-DataAnalysis-Seaborn-) and acess Iris Repository

#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/tree/5dfd47488997c5202bcd6fc1f2a902c22e6e1f49/Projects/project_5-Discr[iminant%20Analyze) and access Project_5 - Discriminant Analyzes as demonstrated in the video below ⬇︎

<br>

https://github.com/user-attachments/assets/e1757373-b9a7-4051-afb8-63855576f1b2


<br>

## [10.]() Lasso, Ridge and Ecolastic Net Regression: Complete Overview with Definitions, Regularization Concepts, and Implementation Steps

<br>

#### [Click here]() to access Theoretical and Pratical Material.

#### [Click here]() to access the Dataset

#### [Click here]() and access Lasso Regression Basic Code 

<br>

## 1. [Definitions]():

### Lasso Regression (L1 Regularization)
Lasso regression (**L**east **A**bsolute **S**hrinkage and **S**election **O**perator) is a linear regression technique that applies **L1 regularization** by adding a penalty term equal to the sum of the absolute values of the coefficients to the ordinary least squares (OLS) cost function. This method performs **variable selection** and **regularization** simultaneously, shrinking less important coefficients to zero to reduce overfitting and enhance model interpretability.

Used in genomics, finance, and machine learning for tasks requiring model simplicity, multicollinearity handling, and automated feature selection.

### [**Objective Function:**]():

$\arg\min_{\mathbf{w}} \left( \text{MSE}(\mathbf{w}) + \lambda \|\mathbf{w}\|_1 \right)\$

Where $\(\lambda\)$ controls the regularization strength.

### [Ridge Regression]() (L2 Regularization):

Ridge regression applies **L2 regularization** by adding a penalty term equal to the sum of the squared coefficients to the OLS cost function. It shrinks coefficients toward zero but does not eliminate them, making it effective for handling multicollinearity and improving model stability.

**Objective Function:**

$\arg\min_{\mathbf{w}} \left( \text{MSE}(\mathbf{w}) + \lambda \|\mathbf{w}\|_2^2 \right)\#

[Where](): $\(\lambda\)$ controls the strength of regularization.

<br>

## 2. [Regularization to Prevent Overfitting]():

Regularization techniques add a **penalty term** to the loss function to discourage complex models that overfit the training data by fitting noise instead of underlying patterns. The penalty term controls the magnitude of the coefficients, reducing variance at the cost of introducing some bias (bias-variance tradeoff). This leads to better generalization on unseen data.

- **L1 penalty (Lasso)** encourages sparsity by forcing some coefficients to be exactly zero, effectively performing feature selection.
- **L2 penalty (Ridge)** shrinks coefficients uniformly but keeps all features, improving stability especially when predictors are correlated.

The regularization parameter \(\lambda\) controls the strength of the penalty:  
- Higher \(\lambda\) means stronger shrinkage and simpler models.  
- Optimal \(\lambda\) is usually found via cross-validation.

<br>

### 3. [Penalization Terms Summary]()

| Regularization Type | Penalization Term                  | Effect                         |
|--------------------|----------------------------------|--------------------------------|
| **L1 (Lasso)**     | \(\lambda \sum_i |w_i|\)          | Shrinks some coefficients to zero (feature selection) |
| **L2 (Ridge)**     | \(\lambda \sum_i w_i^2\)          | Shrinks coefficients towards zero without elimination |
| **Elastic Net**    | \(\lambda_1 \sum_i |w_i| + \lambda_2 \sum_i w_i^2\) | Combines L1 and L2 penalties for feature selection and stability |

<br>

### 4. [Comparison Between Lasso and Ridge]()

| Aspect               | Lasso (L1)                              | Ridge (L2)                            |
|----------------------|---------------------------------------|-------------------------------------|
| Penalty Term         | \(\lambda \sum |w_i|\)                 | \(\lambda \sum w_i^2\)               |
| Feature Selection    | Yes (sparse models, some coefficients exactly zero) | No (all coefficients shrunk but retained) |
| Handles Multicollinearity | Partially                          | Yes (stabilizes correlated predictors) |
| Use Cases            | High-dimensional data with sparse signals | Correlated predictors, small sample sizes |
| Coefficients Impact  | Some coefficients become zero         | Coefficients shrink but remain nonzero |

<br>

### 5. [How to Execute Ridge Regression: Step-by-Step]()

### Step 1: Data Preparation
- **Standardize features** to zero mean and unit variance because Ridge regression is sensitive to feature scales.  
- **Split dataset** into training and testing sets (e.g., 80% train, 20% test).

### Step 2: Model Training and Hyperparameter Tuning (Python Example)

```python
from sklearn.linear_model import RidgeCV from sklearn.preprocessing import StandardScaler from sklearn.model_selection import train_test_split from sklearn.metrics import mean_squared_error import numpy as np

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler().fit(X_train) X_train_scaled = scaler.transform(X_train) X_test_scaled = scaler.transform(X_test)

# Define candidate lambdas (alphas)
alphas = 0.001, 0.01, 0.1, 1, 10, 100

# Train Ridge with cross-validation
ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_train_scaled, y_train) best_alpha = ridge_cv.alpha_

# Predictions and evaluation
y_pred = ridge_cv.predict(X_test_scaled) rmse = np.sqrt(mean_squared_error(y_test, y_pred)) print(f”Best lambda: {best_alpha}, Test RMSE: {rmse}”)
```

<br>

## [Comparing Reults]()  -  Lasso, Ridge, and Elastic Net Regression

Here we demonstrates how to apply and compare three popular linear regression techniques-**Lasso**, **Ridge**, and **Elastic Net**-using Python. The models are evaluated using a real estate dataset to predict property values.

<br>

### [Overview]():

The goal of this project is to:

- Load and preprocess a real estate dataset.
- Train and evaluate Lasso, Ridge, and Elastic Net regression models.
- Compare their performance using Mean Squared Error (MSE) and R² score.
- Visualize the results for easy interpretation.

<br>  

### [Requirements]():

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

<br>

### [Install Dependencies with]():

```python
pip install pandas numpy matplotlib seaborn scikit-learn
```

<br>

### [Dataset]()

#### ➢ Get the [Dataset](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/2d91771f1f79bef4d4d635d5bf17564e140cb0c2/Projects/project_6-Lasso-Roidge-Elastic-Regression/Imoveis%20(1).xlsx) 

- The dataset should be in Excel format (`.xlsx`) and must contain a column named `Valor` (the target variable).
- Update the file path in the code as needed:

<br>

```python
dados = pd.read_excel('path/to/your/Imoveis.xlsx')
```

<br>

### [How to Run]()

1. **Clone this repository or copy the code.**
2. **Ensure your dataset is available and the path is correct.**
3. **Install the required libraries.**
4. **Run the script:**

<br>   

```bash
python your_script_name.py
```

<br>   

### [Code Structure]():

### The script follows these steps:

<br>  

## 1. Step 1: [Import Necessary Libraries]()

```python
# Importing essential libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
```

<br>  

#### Setting dark mode for plots

<br>  

```python
plt.style.use('dark_background')
sns.set_palette("deep")
```

<br>  

## Step 2: [Load the Dataset]()

#### Loading the dataset

```python
# Replace the file path with the correct one if necessary

dados = pd.read_excel('/Users/fabicampanari/Desktop/class_6-Lasso Regression/project_6-Lasso-Roidge-Elastic-Regression/Imoveis (1).xlsx')
```
<br>  

#### Replace the file path with the correct one if necessary

```python
dados = pd.read_excel('/Users/fabicampanari/Desktop/class_6-Lasso Regression/project_6-Lasso-Roidge-Elastic-Regression/Imoveis (1).xlsx')
```
<br>  

#### Displaying the first few rows of the dataset

```python
print(dados.head())
```

<br>  

## Step 3: [Preprocess the Data]()

#### Separating predictors (X) and the target variable (y)

```python
X = dados.drop(columns=['Valor'])  \# Replace 'Value' with the actual column name for the target variable
y = dados['Valor']
```

<br>  

#### Splitting the data into training and testing sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

<br>  

#### Standardizing the predictors

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

<br>  

## Step 4: [Lasso Regression]()

#### Fitting a Lasso Regression model

```python
lasso = Lasso(alpha=0.1, random_state=42)  \# Adjust alpha as needed
lasso.fit(X_train_scaled, y_train)
```

<br>  

#### Making predictions

```python
y_pred_lasso = lasso.predict(X_test_scaled)
```

<br>  

#### Evaluating the model

```python
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
```

<br>  

#### Printing results

```python
print(f"Lasso Regression - MSE: {mse_lasso}, R2: {r2_lasso}")
```

<br>

## Step 5: [Ridge Regression]()

#### Fitting a Ridge Regression model

```python
ridge = Ridge(alpha=0.1, random_state=42)  \# Adjust alpha as needed
ridge.fit(X_train_scaled, y_train)
```

<br>

#### Making predictions

```python
y_pred_ridge = ridge.predict(X_test_scaled)
```

<br>

#### Evaluating the model

```python
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
```

<br>

#### Printing results

```python
print(f"Ridge Regression - MSE: {mse_ridge}, R2: {r2_ridge}")
```

<br>

## Step 6: [Elastic Net Regression]()

#### Fitting an Elastic Net model

```python
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)  \# Adjust alpha and l1_ratio as needed
elastic_net.fit(X_train_scaled, y_train)
```

<br>

####  Making predictions

```python
y_pred_elastic = elastic_net.predict(X_test_scaled)
```

<br>

#### Evaluating the model

```python
mse_elastic = mean_squared_error(y_test, y_pred_elastic)
r2_elastic = r2_score(y_test, y_pred_elastic)
```

<br>

#### Printing results

```python
print(f"Elastic Net Regression - MSE: {mse_elastic}, R2: {r2_elastic}")
```

<br>

## Step 7: [Compare Results]()

#### Comparing the results of the three models

```python
results = pd.DataFrame({
'Model': ['Lasso', 'Ridge', 'Elastic Net'],
'MSE': [mse_lasso, mse_ridge, mse_elastic],
'R2': [r2_lasso, r2_ridge, r2_elastic]
})

print(results)
```

<br> 

## Step 8: [Visualize Results]()

<br>

#### Plotting the comparison of MSE

```python
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MSE', data=results)
plt.title('Comparison of MSE Across Models', fontsize=16)
plt.xlabel('Regression Model', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.show()
```

<br>

#### Plotting the comparison of R2

```python
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2', data=results)
plt.title('Comparison of R2 Across Models', fontsize=16)
plt.xlabel('Regression Model', fontsize=14)
plt.ylabel('R2 Score', fontsize=14)
plt.show()
```

<br>


## [11.]() - Brains Made of Code: Regression Training with Gradient Descent and Stochastic Optimization Algorithms

<br>

#### ➢ [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/86fa3289a6e57c991e29c77e7bbbe63b7b74dc2b/class__10-Decreasing%20Gradient-Buildin%20a%20Brain/Descendent%20Gradient.pdf) to access Theoretical and Pratical Material.

#### ➣ [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/ab8c90b8685fa6e3b0ce0aba6df8c07eaa960930/class__10-Decreasing%20Gradient-Buildin%20a%20Brain/Regresao_Lasso_Ridge.xlsx) to access the Dataset

#### ➢ [Click here]() to access the code implementations for [Gradient Descent](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/95466df92a8411959dcf8cff8abba2e84d35ab94/class__10-Decreasing%20Gradient-Buildin%20a%20Brain/1-Gradient_Descendin.ipynb) and [Stochastic Gradient Descent](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/37e25f4452b0b43dc3e47c418616920c185b1f38/class__10-Decreasing%20Gradient-Buildin%20a%20Brain/2-Gradient_Descending_Stochastic.ipynb)

####  ➣ View the complete [Project: Building a Linear Regression and Comparision Model using Batch Gradient Descent and Stochastic Gradient Descent](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/tree/597b04603ea55df97a4b85d0b7845bec141d442e/Projects/project_7-Build-Regression-DescendingGradient-Stochastic-DG), including [full code](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/7b25e83f1eaa91665f0bd312993aa4c6404a4ccd/Projects/project_7-Build-Regression-DescendingGradient-Stochastic-DG/RealEstate_Regression_GradientDescent_Comparison.ipynb), [dataset](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/17df670a5197215a2c6418769e4c23f8a2215717/Projects/project_7-Build-Regression-DescendingGradient-Stochastic-DG/Imoveis.xlsx), and visualization [scatter plots](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/tree/9a6a059212105ede6f89a562a4eadf89b2ee79e6/Projects/project_7-Build-Regression-DescendingGradient-Stochastic-DG/Scatter%20Plots).

<br>

This section provides a concise overview of the fundamental concepts behind Artificial Neural Networks (ANNs) and the Gradient Descent algorithm used for training them. It introduces key ideas such as the perceptron model, multilayer perceptrons (MLPs), error correction learning, and the backpropagation algorithm.

For a complete and detailed explanation, including all theoretical background, mathematical derivations, and implementation code, please refer to the dedicated repository [**Brains Made of Code: Regression Training with Gradient Descent and Stochastic Optimization Algorithms**](https://github.com/Mindful-AI-Assistants/brains-made-of-code-ml-gd-sgd/blob/main/README.md), which contains the full content and examples.

This summary is part of a larger collection of topics related to neural tworks and machine learning, designed to provide both conceptual understanding and practical tools.


<br>

## [12.]() - Bayesian- KNN Regression - Model Persistence


#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/d806a426ad8506ff752152ff7e198a49325b6bbb/class__12-%20Bayesian-KNN%20Regression-Model%20Persistence/dataset/Bayesian%20and%20%20KK%20Regression%20Workbook.pdf) to access Theoretical Material 

#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/3628cc01aa37e3e805352bed917f95d83685237e/class__12-%20Bayesian-KNN%20Regression-Model%20Persistence/dataset/Consumo.xlsx) to access the Dataset

#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/df27891910b5dbbd3ccec122008eec2911f0caaf/class__12-%20Bayesian-KNN%20Regression-Model%20Persistence/dataset/Bayesian_Regression.ipynb) and access Bayesian  Regreesion Code

#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/d806a426ad8506ff752152ff7e198a49325b6bbb/class__12-%20Bayesian-KNN%20Regression-Model%20Persistence/dataset/KNN%20Regression.ipynb) and access KNN  Regreesion  Code


#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/d52ff5f2d3b95933d0630d4ae65e7b6d6866aa3c/class__12-%20Bayesian-KNN%20Regression-Model%20Persistence/dataset/modelo_knn.joblib) and access Model-kNN.joblib

#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/blob/d806a426ad8506ff752152ff7e198a49325b6bbb/class__12-%20Bayesian-KNN%20Regression-Model%20Persistence/dataset/Persistence.ipynb) and access Persistence Code

#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/tree/d52ff5f2d3b95933d0630d4ae65e7b6d6866aa3c/class__12-%20Bayesian-KNN%20Regression-Model%20Persistence/dataset/Saved%20Persistece%20Model) Model Persistence Code

#### [Click here](https://github.com/Quantum-Software-Development/Integrated-Project-for-Business/tree/d52ff5f2d3b95933d0630d4ae65e7b6d6866aa3c/class__12-%20Bayesian-KNN%20Regression-Model%20Persistence/dataset/Saved%20Persistece%20Model) and access Saved Persistece Model

<br>

### <p align="center">  Extra Explanbation for Bayesian Regression by [Professor Eric](https://www.linkedin.com/in/eric-bacconi-423137/)

https://github.com/user-attachments/assets/f806fe77-acd8-4c4e-a2af-44ea32d7c155









<br><br>

#

###### <p align="center"> Copyright 2025 Quantum Software Development. Code released under the [MIT License.](https://github.com/Quantum-Software-Development/Math/blob/3bf8270ca09d3848f2bf22f9ac89368e52a2fb66/LICENSE)




  


