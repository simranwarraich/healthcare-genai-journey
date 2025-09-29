QC Notes: Supervised Learning – Linear Regression

1. Supervised Learning

Definition: Machine learning where the model learns from labeled data (input X, output Y).

Goal: Map inputs → outputs by minimizing error.

Examples: Predicting house prices, salary prediction, medical cost prediction.

2. Linear Regression

Purpose: Predict a continuous (numeric) output.

Equation:   Y=β0​+β1​X+ϵ

Y: dependent variable (target)
X: independent variable (feature)
𝛽0: intercept
𝛽1: slope (effect of X on Y)
ϵ: error term

3. Types

Simple Linear Regression – one independent variable.

Multiple Linear Regression – multiple independent variables.

4. Assumptions

Linearity (relationship between X and Y is linear).

Independence of errors.

Homoscedasticity (constant variance of errors).

Normal distribution of errors.

No multicollinearity (for multiple regression).

5. Model Fitting
Method: Ordinary Least Squares (OLS) → minimizes the sum of squared errors (SSE).
SSE=∑(Yi​−Y^i​)2
6. Performance Metrics

R² (Coefficient of Determination): Proportion of variance explained.

Adjusted R²: Adjusts for number of predictors.

MSE / RMSE: Mean squared error / Root MSE.

MAE: Mean absolute error.

7. Pros

Simple to implement, fast.

Easy to interpret (clear relationship between variables).

Works well with linear relationships.

8. Cons

Sensitive to outliers.

Poor with non-linear data.

Assumptions often violated in real-world data.

9. Applications

Predicting housing prices.

Sales forecasting.

Medical cost estimation.

Risk prediction in insurance.
