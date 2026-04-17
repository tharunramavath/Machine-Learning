## Table of Contents

1. [Linear Regression](#linear-regression)
   - [Definition](#definition)
   - [Analogy](#analogy)
   - [Model Representation (Hypothesis Function)](#model-representation-hypothesis-function)
   - [Simple Linear Regression (Hypothesis Function)](#simple-linear-regression-hypothesis-function)
   - [Formal Definition](#formal-definition)
   - [Objective (Learning Goal)](#objective-learning-goal)
   - [Cost Function (Mean Squared Error)](#cost-function-mean-squared-error)
   - [Intuition of Objective](#intuition-of-objective)
   - [Key Parameters](#key-parameters)
   - [How Learning Happens (Optimization)](#how-learning-happens-optimization)
   - [Intuition of Learning Process](#intuition-of-learning-process)
   - [Summary Insight](#summary-insight)

2. [Ordinary Least Squares (OLS)](#ordinary-least-squares-ols--linear-regression)
   - [Definition](#definition-1)
   - [Analogy](#analogy-1)
   - [Cost Function (Objective)](#cost-function-objective)
   - [Interpretation](#interpretation)
   - [Error Behavior](#error-behavior)
   - [Key Results from OLS](#key-results-from-ols)
   - [Intercept](#intercept)
   - [Slope](#slope)
   - [Formal Definition (Optimization View)](#formal-definition-optimization-view)
   - [Key Properties of OLS](#key-properties-of-ols)

3. [Assumptions of Linear Regression (OLS)](#assumptions-of-linear-regression-ols)
   - [Linearity](#1-linearity)
   - [Independence of Errors](#2-independence-of-errors)
   - [Constant Variance (Homoscedasticity)](#3-constant-variance-homoscedasticity)
   - [Normality of Errors](#4-normality-of-errors)
   - [No Multicollinearity](#5-no-multicollinearity-multiple-regression)
   - [No Autocorrelation](#6-no-autocorrelation)
   - [Additivity](#7-additivity)
   - [Summary Table](#summary-table)

4. [Limitations of Linear Regression (OLS)](#limitations-of-linear-regression-ols)
   - [Assumes Linearity](#1-assumes-linearity)
   - [Sensitivity to Outliers](#2-sensitivity-to-outliers)
   - [Multicollinearity](#3-multicollinearity)

---

## Linear Regression {#linear-regression}

### Definition

Linear regression is a supervised learning algorithm that models the relationship between input variables $$X$$ and a continuous output $$Y$$ by fitting a linear function to the data.

---

### Analogy

Think of trying to predict a student’s exam score based on the number of hours studied. You draw a straight line that best represents how scores increase with study time. That line helps you estimate scores for new students based on their study time.

---

### Model Representation (Hypothesis Function)

$$
y = w^T x + b
$$

- w → weights (coefficients)
- b → bias (intercept)

This represents a linear relationship where the output is a weighted sum of inputs plus a bias.

---

### Simple Linear Regression (Hypothesis Function)

$$
h_\theta(x) = \theta_0 + \theta_1 x
$$

- $$\theta_0$$ → Intercept (bias term)  
- $$\theta_1$$ → Slope (weight of feature $$x$$)  
- $$x$$ → Input feature  
- $$h_\theta(x)$$ → Predicted output  

**Interpretation:**
- $$\theta_0$$ shifts the line up or down  
- $$\theta_1$$ controls how steep the line is
  
---

### Formal Definition

Given training data:

$$
\{(x_1,y_1), (x_2,y_2), ..., (x_n,y_n)\}
$$

Learn a function:

$$
h_\theta(x) = w^T x + b
$$

that best approximates the true relationship between $$X$$ and $$Y$$.

---

### Objective (Learning Goal)

The goal is to find parameters $$w$$ and $$b$$ that minimize the difference between predicted and actual values.

#### Optimization Objective

$$
\min_{\theta_0, \theta_1} \; J(\theta_0, \theta_1)
$$

- Best-fit line = line with minimum cost  


#### Residual Error

$$
\text{Residual} = \text{Actual} - \text{Predicted}
$$

- Coefficients ($$\theta_0, \theta_1$$) indicate feature contribution to output  

---

### Cost Function (Mean Squared Error)
Purpose: Measure how well the model fits the data

$$
J(w,b) = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - (w^T x_i + b)\right)^2
$$

#### For Simple Linear Regression

$$
J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2
$$

- $$m$$ → number of training examples  

- Cost function is **convex (U-shaped)** → guarantees a **global minimum**  
---

### Intuition of Objective

- The model tries to make predictions as close as possible to actual values.  
- Squaring the error penalizes larger mistakes more heavily.
  - Squaring ensures:
    - No negative errors
    - Errors don’t cancel out each other
    - Larger errors penalized more
- The best-fit line is the one that minimizes overall squared error.  

---

### Key Parameters

#### 1. Weights ($$w$$)
- Represent the importance of each feature.  
- Control the slope of the line (or hyperplane in higher dimensions).  

#### 2. Bias ($$b$$)
- Represents the intercept.  
- Shifts the line up or down.  

#### 3. Learning Rate ($$\eta$$)
- Controls how fast the model updates parameters during optimization.  
  - Small → slow convergence(small steps)
  - Large → may overshoot minimum(larger steps)
#### 4. Predictions ($$\hat{y}$$)

$$
\hat{y} = w^T x + b
$$

---

### How Learning Happens (Optimization)

The parameters are typically learned using gradient descent:

$$
w := w - \eta \frac{\partial J}{\partial w}, \quad b := b - \eta \frac{\partial J}{\partial b}
$$

#### Core Idea (Gradient Descent)

- Iteratively update parameters to minimize cost  

  - Start with random $$\theta_0, \theta_1$$  

  - Move towards minimum of the cost function( down the hill, cost slope )
---

### Intuition of Learning Process

- Start with a random line  
- Measure error  
- Adjust slope and intercept to reduce error  
- Repeat until the best-fit line is found  

---

[Back to Table of Contents](#table-of-contents)

---

## Ordinary Least Squares (OLS) {#ordinary-least-squares-ols--linear-regression}

### Definition

Ordinary Least Squares (OLS) is a method used in linear regression to estimate the best-fitting line by minimizing the sum of squared differences (errors) between actual and predicted values.

---

### Analogy

Imagine drawing a straight line through scattered data points such that the total vertical distance from all points to the line is as small as possible. Squaring ensures larger errors are penalized more.

---

### Cost Function (Objective)

$$
S(\beta_0, \beta_1) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2
$$

---

### Interpretation
- Measures average squared error between actual $$y_i$$ and predicted values  
- Goal → Minimize this function to find best parameters  

---

## Error Behavior

### Definition

Error is the difference between actual and predicted values:

$$
e_i = y_i - \hat{y}_i
$$

---

### Key Insights
- As error increases → MSE increases quadratically  
- The cost function forms a convex surface  
- → guarantees a single global minimum  

---

## Key Results from OLS

### Intercept ($$\beta_0$$)

$$
\beta_0 = \bar{y} - \beta_1 \bar{x}
$$

- Ensures regression line passes through mean point $$(\bar{x}, \bar{y})$$  

---

### Slope ($$\beta_1$$)

$$
\beta_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
$$

---

### Interpretation
- Measures change in $$y$$ for a unit change in $$x$$  
- Captures strength and direction of relationship  

---

### Formal Definition (Optimization View)

OLS finds:

$$
(\beta_0, \beta_1) = \arg\min_{\beta_0, \beta_1} S(\beta_0, \beta_1)
$$

---

## Key Properties of OLS

- Closed-form solution (no iterative optimization needed)  
- Equivalent to minimizing Mean Squared Error (MSE)  
- Produces Best Linear Unbiased Estimator (BLUE) under assumptions  

---

[Back to Table of Contents](#table-of-contents)

---

## Assumptions of Linear Regression (OLS) {#assumptions-of-linear-regression-ols}

### Definition

Linear regression relies on a set of assumptions to ensure that the estimated coefficients are valid, unbiased, and reliable, and that statistical inferences (like confidence intervals, p-values) are meaningful.

---

### Analogy

Think of building a bridge:  
If the foundational conditions (assumptions) are not satisfied, the structure may still stand—but it won’t be stable or trustworthy.

---

## 1. Linearity

### Definition
The relationship between input variables $$X$$ and output $$Y$$ is linear.

### Analogy
Like fitting a straight line to data—if the true pattern is curved, the line will miss it.

### Formal Definition

$$
Y = \beta_0 + \beta_1 X + \epsilon
$$

### Key Insight
- Model captures only linear relationships  
- Non-linear patterns → underfitting  

---

## 2. Independence of Errors

### Definition
The residuals (errors) should be independent of each other.

### Analogy
Each mistake should be unrelated—one error shouldn’t influence another.

### Formal Definition

$$
\text{Cov}(\epsilon_i, \epsilon_j) = 0 \quad \text{for } i \ne j
$$

### Key Insight
- Violation common in time-series data  
- Leads to misleading inference  

---

## 3. Constant Variance (Homoscedasticity)

### Definition
The variance of errors should be constant across all values of $$X$$.

### Analogy
Errors should be evenly spread—not increasing or decreasing as values grow.

### Formal Definition

$$
\text{Var}(\epsilon_i) = \sigma^2 \quad \forall i
$$

**Violation: Heteroscedasticity**
- Error variance changes (fan shape)  
- Leads to inefficient estimates  

---

## 4. Normality of Errors

### Definition
Errors should follow a normal distribution.

### Analogy
Most errors are small, few are large—forming a bell-shaped curve.

### Formal Definition

$$
\epsilon \sim \mathcal{N}(0, \sigma^2)
$$

### Key Insight
- Important for statistical inference  
- Less critical for prediction accuracy  

---

## 5. No Multicollinearity (Multiple Regression)

### Definition
Independent variables should not be highly correlated with each other.

### Analogy
Avoid using duplicate or highly similar information.

### Formal Definition

$$
\text{Corr}(X_i, X_j) \approx 0
$$

### Key Insight
- Causes unstable coefficients  
- Hard to interpret feature importance  

---

## 6. No Autocorrelation

### Definition
Errors should not show patterns over time or sequence.

### Analogy
Mistakes should not follow a trend like “high-low-high-low”.

### Formal Definition

$$
\epsilon_t \not\sim \epsilon_{t-1}
$$

### Key Insight
- Common issue in time-series data  
- Violates independence assumption  

---

## 7. Additivity

### Definition
The effect of multiple variables is additive, meaning each contributes independently.

### Analogy
Total score = contribution of each subject added together.

### Formal Definition

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \epsilon
$$

### Key Insight
- No hidden interactions unless explicitly modeled  
- Interaction terms must be added manually  

---

## Final Insight

These assumptions ensure that:

- Estimates are unbiased (correct on average)  
- Predictions are reliable  
- Statistical tests are valid  

---

## Summary Table

| Assumption | What it Ensures | Violation Effect |
|-----------|----------------|------------------|
| Linearity | Correct model form | Underfitting |
| Independence | No dependency in errors | Misleading results |
| Homoscedasticity | Stable variance | Inefficient estimates |
| Normality | Valid inference | Affects confidence intervals |
| No Multicollinearity | Stable coefficients | Unstable estimates |
| No Autocorrelation | No time pattern | Biased inference |
| Additivity | Independent effects | Missed interactions |

---

## One-Line Mental Model

Linear Regression works best when data behaves “clean and simple”
---

[Back to Table of Contents](#table-of-contents)

---

## Limitations of Linear Regression (OLS) {#limitations-of-linear-regression-ols}

### 1. Assumes Linearity

**Definition:**
Model assumes a linear relationship between input and output.

**Impact:**
Fails when real relationship is non-linear.

---

### 2. Sensitivity to Outliers

**Definition:**
Squared error magnifies the impact of extreme values.

**Analogy:**
One extreme data point can “pull” the regression line toward it.

---

### 3. Multicollinearity 

- Highly correlated features can make coefficients unstable  

---

## Final Insight 

OLS is the foundation of linear regression, where:

- Objective → Minimize squared error  
- Output → Best-fit straight line  
- Strength → Simple, interpretable  
- Weakness → Rigid assumptions + sensitive to outliers  

---

## One-Line Mental Model

OLS = Find line that minimizes squared distances from data points  

---


### Summary Insight

Linear regression is essentially about finding the best straight line (or hyperplane) that captures the relationship between inputs and outputs by minimizing prediction error or mean squared error by using optimization techniques like gradient descent. It is simple, interpretable, and forms the foundation for many advanced machine learning models.

---

[Back to Table of Contents](#table-of-contents)