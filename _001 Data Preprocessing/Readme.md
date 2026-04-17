# Data Preprocessing 

## Table of Contents

1. [Conceptual Foundation](#1-conceptual-foundation)
   - [What is Data Preprocessing?](#what-is-data-preprocessing)
   - [Intuition](#intuition)

2. [Core Data Preprocessing Techniques](#2-core-data-preprocessing-techniques)
   - [Handling Missing Values](#1-handling-missing-values)
   - [Handling Outliers](#2-handling-outliers)
   - [Feature Scaling](#3-feature-scaling)
   - [Encoding Categorical Variables](#4-encoding-categorical-variables)
   - [Feature Engineering](#5-feature-engineering)
   - [Feature Selection](#6-feature-selection)

3. [Important Terminologies](#3-important-terminologies)

4. [Hyperparameter Tuning in Preprocessing](#4-hyperparameter-tuning-in-preprocessing)
   - [Techniques](#techniques)

5. [When to Use / When Not to Use Preprocessing](#5-when-to-use--when-not-to-use-preprocessing)

6. [Comparison of Techniques](#6-comparison-of-techniques)

7. [Interview Questions](#7-interview-questions)

8. [Real-World Applications](#8-real-world-applications)

9. [Conclusion](#9-conclusion)

10. [Handling Imbalanced Datasets](#handling-imbalanced-datasets)
   - [Resampling Methods](#1-resampling-methods)
   - [Algorithmic Adjustments](#2-algorithmic-adjustments)
   - [Ensemble Methods](#3-ensemble-methods)
   - [Evaluation Metrics](#4-evaluation-metrics)

---

## 1. Conceptual Foundation

### What is Data Preprocessing?

Data preprocessing is the process of transforming raw data into a clean and structured format so that machine learning models can learn effectively.

Raw data in real-world scenarios is often:

* Incomplete
* Noisy
* Inconsistent
* Unstructured

Preprocessing ensures:

* Better model performance
* Faster convergence
* More reliable predictions

### Intuition

A machine learning model is only as good as the data it receives. If the input is flawed, the output will also be flawed. Preprocessing ensures that the model receives meaningful and standardized inputs.

---

[Back to Table of Contents](#table-of-contents)

---

## 2. Core Data Preprocessing Techniques

We can divide preprocessing into six major categories:

---

### 1. Handling Missing Values

#### Types of Missing Data

* MCAR (Missing Completely At Random)
* MAR (Missing At Random)
* MNAR (Missing Not At Random)


#### Techniques

**Removal**

* Drop rows if few missing values
* Drop columns if too many missing values

**Imputation**

* Mean or Median (numerical data)
* Mode (categorical data)
* Forward fill or backward fill (time series)
* KNN imputation
* Model-based imputation

#### Missing Data Mechanisms Explained


#### MCAR (Missing Completely At Random)
- Missingness is independent of both observed and unobserved data
- Probability of missing data is the same for all observations
- No systematic pattern in missing values
- Data loss is random → does not introduce bias
- Analysis remains unbiased but may lose statistical power
- Example:
  - A sensor randomly fails and misses temperature readings



#### MAR (Missing At Random)
- Missingness depends only on observed variables, not on the missing values themselves
- Given observed data, the missingness is random
- Can be handled using statistical techniques (e.g., imputation, regression)
- Less severe than MNAR but more complex than MCAR
- Example:
  - Income data is missing more often for younger people (age is observed)



#### MNAR (Missing Not At Random)
- Missingness depends on the unobserved (missing) values themselves
- There is a systematic pattern related to the missing data
- Cannot be handled reliably using standard imputation techniques
- Requires modeling the missing data mechanism explicitly
- Most difficult and biased case
- Example:
  - People with very high incomes choose not to disclose their income

#### Missing Data Imputation Techniques Explained

#### Mean or Median (Numerical Data)
- Replace missing values with:
  - Mean → when data is normally distributed
  - Median → when data has outliers or is skewed
- Simple and fast to implement
- Preserves dataset size
- Mean is sensitive to outliers; median is more robust
- Reduces variance in the data
- Example:
  - Missing salary values filled with average/median salary



#### Mode (Categorical Data)
- Replace missing values with the most frequent category
- Suitable for categorical or discrete variables
- Easy to implement
- Can introduce bias if one category dominates
- Does not consider relationships between variables
- Example:
  - Missing "Gender" filled with most frequent category (e.g., Male/Female)



#### Forward Fill / Backward Fill (Time Series)
- Forward Fill (FFill):
  - Replace missing value with the previous known value
- Backward Fill (BFill):
  - Replace missing value with the next known value
- Maintains temporal continuity
- Useful in sequential/time-dependent data
- Can introduce bias if large gaps exist
- Example:
  - Stock price missing at time t → filled using previous timestamp value



#### KNN Imputation
- Uses K-Nearest Neighbors to estimate missing values
- Finds similar data points based on distance (e.g., Euclidean)
- Missing value is filled using average (or majority) of neighbors
- Captures relationships between variables
- Computationally expensive for large datasets
- Sensitive to choice of K and scaling of features
- Example:
  - Predict missing height using similar individuals (age, weight)



#### Model-Based Imputation
- Uses machine learning models to predict missing values
- Models include:
  - Linear Regression
  - Decision Trees
  - Random Forest
- Treats missing feature as target variable
- More accurate than simple methods
- Captures complex relationships in data
- Risk of overfitting if not handled properly
- Example:
  - Predict missing income using features like education, age, occupation


---

### 2. Handling Outliers

Outliers are extreme values that can distort model behavior.

#### Detection Methods

* Z-score method
* IQR method
* Box plots
* Isolation Forest

#### Treatment Methods

* Remove outliers
* Cap values (Winsorization)
* Transform data (log transformation)

**Important Note**

* Tree-based models are robust to outliers
* Linear models are sensitive

### Outlier Detection Techniques Explained

####  Z-Score Method
- Measures how many standard deviations a data point is from the mean
- Formula:
  - Z = (X - Mean) / Standard Deviation
- Assumes data is normally distributed
- Common threshold:
  - |Z| > 3 → considered an outlier
- Simple and fast to compute
- Sensitive to extreme values (since mean and std are affected)
- Example:
  - Detect unusually high/low exam scores

#### Z-Score Formula

The Z-score indicates how many standard deviations a data point ($X$) is from the mean.

$$Z = \frac{X - \mu}{\sigma}$$

**Notations:**
* **$Z$**: Z-score (Standard Score)
* **$X$**: The specific value (raw score)
* **$\mu$ (or $\bar{x}$)**: Mean of the population (or sample)
* **$\sigma$ (or $s$)$**: Standard deviation of the population (or sample)



#### Why use Z-scores?
1. **Standardization:** It allows you to compare data points from different datasets that have different scales or units.
2. **Probability:** In a standard normal distribution, a Z-score tells you the percentage of data points that fall below or above that specific value.
3. **Outlier Detection:** Typically, a Z-score greater than $+3$ or less than $-3$ is considered an outlier.


####  IQR (Interquartile Range) Method
- Based on the spread of the middle 50% of data
- Steps:
  - Q1 = 25th percentile
  - Q3 = 75th percentile
  - IQR = Q3 - Q1
- Outlier boundaries:
  - Lower bound = Q1 - 1.5 × IQR
  - Upper bound = Q3 + 1.5 × IQR
- Does not assume normal distribution
- Robust to outliers
- Widely used in real-world datasets
- Example:
  - Detect abnormal income values in skewed data



#### Box Plots
- Visual method for identifying outliers
- Components:
  - Box → IQR (Q1 to Q3)
  - Line inside box → Median
  - Whiskers → data range within 1.5 × IQR
  - Points outside whiskers → outliers
- Easy to understand and interpret
- Helps visualize data distribution and skewness
- Useful for quick exploratory data analysis (EDA)
- Example:
  - Visualizing salary distribution across departments


#### Isolation Forest
- Machine learning-based outlier detection algorithm
- Works by randomly partitioning data using decision trees
- Key idea:
  - Outliers are easier to isolate → require fewer splits
- Produces an anomaly score for each data point
- Works well with high-dimensional data
- Does not assume any distribution
- Efficient for large datasets
- Example:
  - Fraud detection in financial transactions

### Outlier Treatment Methods explained

#### Remove Outliers
- Delete data points that are identified as outliers
- Suitable when outliers are due to errors or noise
- Helps improve model performance and stability
- Risk:
  - Loss of important information if outliers are meaningful
- Best used when:
  - Data errors, incorrect entries, or irrelevant extreme values exist
- Example:
  - Removing impossible values like negative age


#### Cap Values (Winsorization)
- Replace extreme values with threshold limits instead of removing them
- Common approach:
  - Lower cap → e.g., 5th percentile
  - Upper cap → e.g., 95th percentile
- Preserves dataset size
- Reduces impact of extreme values without deleting data
- Less sensitive to outliers compared to raw data
- Example:
  - Salary above 95th percentile replaced with 95th percentile value



#### Transform Data (Log Transformation)
- Apply mathematical transformations to reduce skewness
- Common transformations:
  - Log: $$x' = \log(x)$$
  - Square root: $$x' = \sqrt{x}$$
- Compresses large values more than small values
- Makes distribution more normal-like
- Helps improve performance of statistical and ML models
- Limitation:
  - Cannot apply log if values are zero or negative (without adjustment)
- Example:
  - Transforming income data to reduce right skew

### Difference Between Outliers and Noise

####  Outliers
- Data points that significantly differ from the majority of observations
- Can be **valid and meaningful** (not always errors)
- Often represent rare or extreme events
- Can provide important insights (e.g., fraud detection, anomalies)
- Usually occur due to:
  - Natural variability
  - Rare events
  - Measurement issues (sometimes)
- Example:
  - A transaction of ₹10,00,000 in a dataset where most are ₹1,000–₹5,000


#### Noise
- Random errors or disturbances in data
- Typically **irrelevant and meaningless**
- Does not follow any pattern
- Always considered undesirable
- Usually caused by:
  - Sensor errors
  - Data entry mistakes
  - Transmission errors
- Example:
  - Random fluctuations in sensor readings due to interference



#### Key Differences

| Aspect            | Outliers                              | Noise                              |
|------------------|---------------------------------------|------------------------------------|
| Nature           | Rare but potentially meaningful       | Random and meaningless             |
| Cause            | Real events or anomalies              | Errors or disturbances             |
| Pattern          | May have a pattern                    | No pattern                         |
| Importance       | Can provide insights                  | Should be removed                  |
| Treatment        | Analyze before removing               | Usually removed/cleaned            |
| Example          | Fraud transaction                     | Random sensor fluctuation          |
---

### 3. Feature Scaling

Many algorithms rely on distance or gradient-based optimization, so scaling is critical.

#### Types of Scaling

* **Min-Max Scaling**
  Scales values to range [0,1]

* **Standardization**
  Mean = 0, Standard deviation = 1

* **Robust Scaling**
  Uses median and IQR (resistant to outliers)

* **Normalization**
  Scales each data point to unit norm

#### When to Use

* Required for KNN, SVM, PCA, Gradient Descent
* Not required for tree-based models

#### Types of Feature Scaling explained

#### Min-Max Scaling
- Scales values to a fixed range → usually [0, 1]
- Formula:

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

- Preserves relative relationships between values
- Sensitive to outliers (extreme values affect scaling)
- Best used when:
  - Data has known bounds
  - No extreme outliers present



#### Standardization (Z-score Normalization)
- Transforms data to have:
  - Mean = 0
  - Standard deviation = 1
- Formula:

$$
x' = \frac{x - \mu}{\sigma}
$$

- Works well with algorithms assuming normal distribution
- Less affected by outliers than Min-Max (but still impacted)
- Commonly used in:
  - Linear Regression
  - Logistic Regression
  - PCA



#### Robust Scaling
- Uses median and IQR (Interquartile Range)
- Formula:

$$
x' = \frac{x - \text{Median}}{\text{IQR}}
$$

- IQR = Q3 - Q1
- Resistant to outliers
- Best for skewed data or datasets with extreme values
- Maintains relative spread without being influenced by outliers



#### Normalization (Vector Scaling)
- Scales each data point (row) to unit norm
- Common norms:
  - L2 norm (most common)

$$
x' = \frac{x}{\|x\|}
$$

- Where:

$$
\|x\| = \sqrt{\sum x_i^2}
$$

- Used when:
  - Direction matters more than magnitude
  - Text data (TF-IDF), cosine similarity
- Ensures all data points lie on a unit sphere

---

### 4. Encoding Categorical Variables

Machine learning models require numerical input.

#### Techniques

* **Label Encoding**
  Suitable for ordinal data

* **One-Hot Encoding**
  Suitable for nominal data

* **Ordinal Encoding**
  Used when categories have order

* **Target Encoding**
  Replace category with mean target value
  Risk of data leakage

* **Frequency Encoding**
  Replace category with its frequency

#### Categorical Encoding Techniques (with Examples)


#### Label Encoding
- Assigns a unique integer to each category
- Suitable for **ordinal data** (where order matters)
- May introduce unintended ordinal relationships if used on nominal data

**Example:**

| Category | Encoded |
|----------|--------|
| Low      | 0      |
| Medium   | 1      |
| High     | 2      |



#### One-Hot Encoding
- Creates binary columns for each category
- Suitable for **nominal data** (no order)
- Increases dimensionality

**Example:**

| Color | Red | Blue | Green |
|------|-----|------|-------|
| Red  | 1   | 0    | 0     |
| Blue | 0   | 1    | 0     |
| Green| 0   | 0    | 1     |



#### Ordinal Encoding
- Similar to label encoding but explicitly used when **order exists**
- Maintains ranking information

**Example:**

| Education Level | Encoded |
|-----------------|--------|
| School          | 0      |
| Bachelor        | 1      |
| Master          | 2      |
| PhD             | 3      |



#### Target Encoding
- Replaces each category with the **mean of the target variable**
- Powerful but prone to **data leakage**

**Example:**
The Raw Data  
Before encoding, we calculate the Global Mean of our target. Out of these 7 rows, 3 people clicked ($Target = 1$).  

Global Mean ($\mu_g$): $3 / 7 \approx \mathbf{0.43}$  

| User ID | City     | Target (Clicked?) |
|--------|----------|-------------------|
| 1      | Kolkata  | 1                 |
| 2      | Kolkata  | 0                 |
| 3      | Mumbai   | 1                 |
| 4      | Delhi    | 0                 |
| 5      | Kolkata  | 1                 |
| 6      | Delhi    | 0                 |
| 7      | Mumbai   | 0                 |



Calculating Category Means  
We group the data by City and find the average success rate for each.

| City     | Total Rows | Sum of Clicks | Mean (Target Encoding) |
|----------|-----------|---------------|------------------------|
| Kolkata  | 3         | 2             | $2/3 = \mathbf{0.66}$  |
| Mumbai   | 2         | 1             | $1/2 = \mathbf{0.50}$  |
| Delhi    | 2         | 0             | $0/2 = \mathbf{0.00}$  |


The Final Transformed Table  
The model no longer sees strings like "Kolkata"; it sees the statistical probability associated with that city.

| User ID | City (Encoded) | Target (Clicked?) |
|--------|----------------|-------------------|
| 1      | 0.66           | 1                 |
| 2      | 0.66           | 0                 |
| 3      | 0.50           | 1                 |
| 4      | 0.00           | 0                 |
| 5      | 0.66           | 1                 |
| 6      | 0.00           | 0                 |
| 7      | 0.50           | 0                 |


Why Smoothing is Necessary  
Look at Delhi in the table above. It has an encoding of 0.00.  

If we only have two examples of Delhi and both happened to be 0, the model might conclude that nobody in Delhi ever clicks. This is likely a sampling error.  

Smoothing would blend that 0.00 with the Global Mean (0.43), resulting in a value like 0.25. This makes the model more robust to outliers and small sample sizes.



### Frequency Encoding
- Replaces category with its **frequency (count or proportion)**

**Example:**

| Category | Frequency |
|----------|----------|
| A        | 3        |
| B        | 2        |
| C        | 1        |


---

### 5. Feature Engineering

Creating new features from existing data.


#### Examples

* Extract day, month from date  
  - Convert a date column into useful components like:
    - Day of week (Mon–Sun)
    - Month
    - Year
  - Helps models capture seasonality and patterns

---

* Create interaction terms  
  - Combine two or more features to capture relationships
  - Example:
  
$$
\text{Interaction} = x_1 \times x_2
$$

  - Useful when the effect of one feature depends on another

---

* Log transformations  
  - Apply log to reduce skewness in data
  
$$
x' = \log(x)
$$

  - Compresses large values and stabilizes variance
  - Common in income, sales, population data

---

* Polynomial features  
  - Create higher-degree features from existing ones
  
$$
x^2,\; x^3,\; x_1 x_2
$$

  - Helps capture non-linear relationships
  - Often used in regression models

**Why Important**

* Better features often matter more than better models

---

### 6. Feature Selection

Selecting only the most relevant features.

#### Methods

**Filter Methods**

* Correlation
* Chi-square test
* Mutual information

**Wrapper Methods**

* Recursive Feature Elimination

**Embedded Methods**

* Lasso Regression
* Tree-based feature importance

#### Benefits

* Reduces overfitting
* Improves model performance
* Reduces computation

#### Feature Selection Techniques Explained


#### **Filter Methods**
- Select features based on statistical properties (independent of model)
- Fast and scalable

* **Correlation**
  - Measures linear relationship between features and target
  - High absolute correlation → more relevant
  - Formula:

$$
r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

* **Chi-square test**
  - Used for categorical features
  - Tests dependence between feature and target
  - Higher value → stronger relationship

$$
\chi^2 = \sum \frac{(O - E)^2}{E}
$$

* **Mutual Information**
  - Measures how much information one variable provides about another
  - Captures non-linear relationships

$$
I(X;Y) = \sum p(x,y) \log \left(\frac{p(x,y)}{p(x)p(y)}\right)
$$


#### **Wrapper Methods**
- Use a machine learning model to evaluate feature subsets
- Computationally expensive but more accurate

* **Recursive Feature Elimination (RFE)**
  - Train model → rank features → remove least important → repeat
  - Continues until desired number of features is reached


#### **Embedded Methods**
- Feature selection happens during model training
- Balance between performance and efficiency

* **Lasso Regression**
  - Adds L1 regularization → shrinks some coefficients to zero

$$
\text{Loss} = \text{RSS} + \lambda \sum |w_i|
$$

  - Automatically performs feature selection

* **Tree-based Feature Importance**
  - Decision trees rank features based on:
    - Information gain
    - Gini importance
  - Features contributing more to splits → higher importance

---

[Back to Table of Contents](#table-of-contents)

---

## 3. Important Terminologies

* Imputation: Filling missing values
* Scaling: Adjusting feature ranges
* Normalization: Scaling data to unit norm
* Standardization: Mean zero, unit variance
* Outlier: Data point far from others
* Encoding: Converting categorical to numerical
* Dimensionality Reduction: Reducing feature count
* Data Leakage: Using future or target information improperly
* Feature Engineering: Creating new meaningful features
* Feature Selection: Choosing important features only
* Model Drift: The degradation of a model's predictive power over time due to changes in digital environments or consumer behavior.
* Concept Drift: A type of drift where the statistical relationship between the input features and the target variable changes (e.g., what defined "fraud" in 2020 is different in 2026).
* Data Drift (Feature Drift): When the underlying distribution of the input data changes over time, even if the relationship with the target stays the same.

* Data Leakage: When information from outside the training dataset (specifically the target) is used to create the model, leading to over-optimistic performance.

* Bias-Variance Tradeoff: The tension between error from erroneous assumptions (Bias) and error from sensitivity to small fluctuations in the training set (Variance).

* Hyperparameter Tuning: The process of optimizing the external configurations of a model (like learning rate) that are not learned directly from the data.

* Cross-Validation: A resampling technique used to evaluate a model's ability to generalize by training and testing it on different subsets of the data.

* Overfitting: When a model learns the "noise" and specific details of the training data too well, failing to perform accurately on new, unseen data.

* Underfitting: When a model is too simple to capture the underlying structure of the data, resulting in poor performance on both training and test sets.

* Inference: The process of using a trained machine learning model to make predictions on new, real-world data points.

* Cold Start Problem: A situation in recommendation systems where the model lacks enough data about new users or items to provide accurate suggestions.

* A/B Testing: A randomized experiment where two versions of a model are compared to determine which one performs better in a live environment.
---

[Back to Table of Contents](#table-of-contents)

---

## 4. Hyperparameter Tuning in Preprocessing

Preprocessing itself involves parameters that need tuning.

### Examples

* Number of PCA components
* Choice of scaler
* Imputation strategy
* Encoding type

### Techniques

* GridSearchCV with pipelines
* RandomizedSearchCV
* Bayesian optimization

### Best Practice

* Always use preprocessing inside a pipeline to avoid data leakage

### Hyperparameters & Hyperparameter Tuning in detail

### What are Hyperparameters?
- Hyperparameters are **external configuration settings** of a model
- They are **not learned from data** during training
- Must be set **before training begins**
- Control model behavior, complexity, and learning process

**Examples:**
- Learning rate (in gradient descent)
- Number of trees (in Random Forest)
- Depth of tree
- Number of neighbors (K in KNN)
- Regularization parameter (λ)



### What is Hyperparameter Tuning?
- The process of **finding the optimal combination of hyperparameters**
- Done by:
  - Training model with different hyperparameter values
  - Evaluating performance (e.g., accuracy, RMSE)
- Goal:
  - Improve model performance on unseen data



### Why is it Needed?
- Default values are **not optimal for all datasets**
- Helps to:
  - Improve accuracy and generalization
  - Avoid overfitting and underfitting
  - Find the best bias-variance balance
- Different datasets require different configurations
- Critical for achieving **production-level performance**



## Techniques

### GridSearchCV (with Pipelines)
- Exhaustively tries **all possible combinations** of hyperparameters
- Uses cross-validation for evaluation
- Pipelines help:
  - Combine preprocessing + model in one workflow
  - Prevent data leakage

**Example Search Space:**
- Learning rate: [0.01, 0.1]
- Max depth: [3, 5, 7]

→ Tries all combinations

**Pros:**
- Finds optimal solution (within given grid)

**Cons:**
- Computationally expensive



### RandomizedSearchCV
- Randomly samples combinations from parameter space
- Does **not try all combinations**
- Faster than Grid Search

**Pros:**
- Efficient for large search spaces
- Good performance with limited computation

**Cons:**
- May miss best combination



### Bayesian Optimization
- Uses probabilistic models to **choose next best hyperparameters**
- Learns from previous results to improve search

**Idea:**
- Balance:
  - Exploration (try new areas)
  - Exploitation (focus on good areas)

**Pros:**
- More efficient than Grid/Random search
- Requires fewer evaluations

**Cons:**
- More complex to implement

---

[Back to Table of Contents](#table-of-contents)

---

## 5. When to Use / When Not to Use Preprocessing

### Use Preprocessing When

* Data is messy or incomplete
* Features are on different scales
* There are categorical variables
* High dimensionality exists

### Be Careful When

* Over-processing removes useful information
* Encoding introduces data leakage
* Scaling is applied unnecessarily

---

[Back to Table of Contents](#table-of-contents)

---

## 6. Comparison of Techniques

| Technique           | Use Case              | Advantage            | Limitation                |
| ------------------- | --------------------- | -------------------- | ------------------------- |
| Scaling             | Distance-based models | Improves convergence | Not needed for trees      |
| Encoding            | Categorical data      | Enables model usage  | Can increase dimensions   |
| Imputation          | Missing values        | Preserves data       | May introduce bias        |
| Feature Selection   | High dimensional data | Reduces overfitting  | May lose information      |
| Feature Engineering | Any dataset           | Improves performance | Requires domain knowledge |

---

[Back to Table of Contents](#table-of-contents)

---

## 7. Interview Questions

### Conceptual Questions

**1. Why is preprocessing important?**
Because models cannot handle raw, inconsistent data effectively.

**2. When should you use standardization vs normalization?**
Standardization for Gaussian-like data, normalization for distance-based methods.

#### Use **Standardization (Z-score Scaling)** when:
- Data follows (or is close to) a **normal distribution**
- Features have **different units or scales**
- Model assumes **Gaussian distribution**
- You are using:
  - Linear Regression
  - Logistic Regression
  - Support Vector Machines (SVM)
  - PCA
- Dataset may contain **outliers** (more robust than Min-Max)

$$
x' = \frac{x - \mu}{\sigma}
$$



#### Use **Normalization (Min-Max Scaling / Unit Norm)** when:
- You need data in a **fixed range** (e.g., [0, 1])
- Data does **not follow normal distribution**
- You are using:
  - K-Nearest Neighbors (KNN)
  - Neural Networks
  - Gradient Descent-based models (faster convergence)
- You care about **relative distances or magnitudes**
- No significant outliers present (sensitive to outliers)

**Min-Max Scaling:**

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$


**Vector Normalization (L2):**

$$x' = \frac{x}{\|x\|}$$


####  Min-Max Scaling (Normalization)
Min-Max scaling transforms features to a fixed range, usually [0, 1].

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Notations:**
* **$x'$**: The normalized value
* **$x$**: The original value
* **$x_{min}$**: Minimum value in the feature column
* **$x_{max}$**: Maximum value in the feature column



####  Vector Normalization (L2 Norm)
L2 Normalization (Unit Norm) scales the vector so that its total length (magnitude) equals 1.

$$x' = \frac{x}{\|x\|_2}$$

**Notations:**
* **$x'$**: The normalized vector
* **$x$**: The original vector
* **$\|x\|_2$**: The L2 norm (Euclidean distance) of the vector

#### Key Difference

| Aspect              | Standardization                     | Normalization                     |
|--------------------|-----------------------------------|-----------------------------------|
| Output Range       | Not bounded                        | Usually [0, 1]                    |
| Distribution       | Centers data (mean = 0, std = 1)   | Rescales data                     |
| Outlier Sensitivity| Less sensitive                     | Highly sensitive                  |
| Use Case           | Statistical models                 | Distance-based models             |




**3. What is data leakage?**


#### Definition
- Data leakage occurs when **information from outside the training dataset** (especially the target variable or future data) is used during model training
- This leads to **unrealistically high performance** during training/testing but poor performance in real-world scenarios


#### Why It is a Problem
- Model learns patterns that **won’t be available in production**
- Results in **over-optimistic evaluation metrics**
- Fails to generalize to unseen data



### Types of Data Leakage

#### Target Leakage
- When features include information that directly or indirectly reveals the target
- Example:
  - Using "final exam score" to predict "pass/fail"



#### Train-Test Contamination
- When information from the test set leaks into training data
- Example:
  - Scaling entire dataset before splitting
  - Using global mean in target encoding without separation



#### Time Leakage
- Using **future data** to predict past events
- Example:
  - Using next month’s sales to predict current sales



#### Examples

| Feature            | Target | Leakage? |
|--------------------|--------|----------|
| Purchase Amount    | Fraud  | No       |
| Transaction Time   | Fraud  | No       |
| Chargeback Status  | Fraud  | Yes      |



#### How to Prevent Data Leakage
- Split data into **train/test before preprocessing**
- Apply transformations (scaling, encoding) **only on training data**
- Use **pipelines** to ensure proper workflow
- Be careful with **target encoding** (use cross-validation)
- Respect **time order** in time-series data



#### Key Insight
- If a feature gives **unfair or future information**, it is likely causing leakage


**4. Why is one-hot encoding not always ideal?**
It increases dimensionality and can cause sparsity. sparse data can lead to overfitting and increased computational cost. In such cases, alternatives like target encoding or embedding layers (in deep learning) may be more effective.

**5. How do you handle missing values?**
Drop, impute, or use model-based approaches depending on context.

---


## 8. Real-World Applications

* Fraud detection
* Healthcare data cleaning
* Recommendation systems
* Financial risk modeling
* NLP preprocessing

---

[Back to Table of Contents](#table-of-contents)

---

[Back to Table of Contents](#table-of-contents)

---

## 9. Conclusion
Data preprocessing is a critical step in the machine learning pipeline. It ensures that models receive clean, relevant, and well-structured data, which is essential for achieving good performance. By understanding and applying various preprocessing techniques, we can significantly enhance the effectiveness of our models and make more accurate predictions in real-world applications. 

---

[Back to Table of Contents](#table-of-contents)

---

## Handling Imbalanced Datasets

### Definition

Handling imbalanced datasets refers to techniques used when the distribution of classes is unequal, where one class (majority) significantly outnumbers the other (minority), causing models to become biased toward the majority class.

---

### Analogy

Imagine a class where 95 students passed and only 5 failed. A model that always predicts “pass” gets 95% accuracy—but completely ignores the failing students. Handling imbalance is about ensuring the model pays attention to the minority (important) cases.

---

### Formal Definition

Given a dataset:

$$
D = \{(x_i, y_i)\}_{i=1}^{n}, \quad y_i \in \{0,1\}
$$

where:

$$
P(y=1) \ll P(y=0)
$$

The goal is to learn a function:

$$
h: X \rightarrow Y
$$

that performs well across all classes, especially the minority class, by minimizing a class-sensitive loss function.

---

## 1. Resampling Methods

### Definition
Resampling modifies the dataset distribution to balance class proportions.

---

### a) Oversampling (Minority Class)

**Definition:**
Increase the number of minority samples by duplicating or generating new ones.

**Analogy:**
Like giving extra practice questions to weaker students so they get equal attention.

#### SMOTE (Synthetic Minority Oversampling Technique)

**Definition:**
SMOTE generates synthetic samples by interpolating between existing minority points.

**Formal Definition:**

$$
x_{\text{new}} = x_i + \lambda (x_j - x_i), \quad \lambda \in [0,1]
$$

- $$x_i$$: minority sample  
- $$x_j$$: one of its nearest neighbors  

**Intuition:**
Instead of copying points, SMOTE creates new realistic data points, improving generalization and reducing overfitting.

---

### b) Undersampling (Majority Class)

**Definition:**
Reduce the number of majority samples to balance the dataset.

**Analogy:**
Like selecting a smaller, representative group from a large crowd to ensure fairness.

**Trade-off:**
- Pros: Faster training  
- Cons: May lose important information  

---

## 2. Algorithmic Adjustments

### Definition
Modify the learning algorithm to give more importance to minority class errors.

---

### a) Class Weighting

**Definition:**
Assign higher penalty to misclassifying minority class samples.

**Formal Definition:**

$$
J = \sum_{i=1}^{n} w_{y_i} \cdot L(y_i, \hat{y}_i)
$$

- $$w_{y_i}$$: higher for minority class  

**Intuition:**
The model is “punished more” for ignoring minority cases.

In machine learning, being “punished more” means the loss function assigns a higher penalty to certain errors, making the model prioritize correcting them. In the weighted loss formula:

$$
J = \sum_{i=1}^{n} w_{y_i} \cdot L(y_i, \hat{y}_i)
$$

the term $$w_{y_i}$$ acts as a punishment multiplier. For imbalanced datasets, errors on minority classes (e.g., sick patients) are given higher weights, so even if the raw error $$L(y_i, \hat{y}_i)$$ is the same, the total penalty becomes much larger.  

During gradient descent, this creates stronger “pressure” to reduce those high-weight errors, forcing the model to focus more on correctly predicting minority cases, even at the cost of slightly lower accuracy on majority classes.  

Inother words:
Think of the loss function as a strict teacher. If the teacher gives a small penalty for a mistake, the student might not care much about fixing it. If the teacher gives a massive penalty for that same mistake, the student will work much harder to avoid it.

---

### b) Cost-Sensitive Learning

**Definition:**
Explicitly define different misclassification costs:

$$
\text{Cost(FN)} > \text{Cost(FP)}
$$

**Analogy:**
Missing a fraud transaction (FN) is worse than flagging a normal one (FP).

---

## 3. Ensemble Methods

### Definition
Combine multiple models to improve performance on imbalanced data.

---

### a) Boosting

**Definition:**
Sequentially trains models focusing more on misclassified samples, often minority ones.

**Analogy:**
A teacher focusing more on students who keep making mistakes.

---

### b) Bagging with Balanced Sampling

**Definition:**
Train multiple models on balanced subsets of data.

**Intuition:**
Each model sees a fair representation of classes.

---

## 4. Evaluation Metrics

### Definition
Use metrics that capture performance on minority class, instead of relying on accuracy.

---

### Why Accuracy Fails

$$
\text{Accuracy} = \frac{TP + TN}{\text{Total}}
$$

High accuracy can still ignore minority class completely.

---

### Important Metrics

#### Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

- How many predicted positives are correct  

---

#### Recall (Sensitivity)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

- How many actual positives are captured  

---

#### F1 Score

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

- Balance between precision and recall  

---

#### ROC-AUC
- Measures model’s ability to separate classes across thresholds  

---

## Final Insight

Handling imbalance is not just a preprocessing step—it’s a modeling strategy decision:

- Data-level solutions → Resampling  
- Algorithm-level solutions → Weighting, cost-sensitive learning  
- Model-level solutions → Ensembles  
- Evaluation-level solutions → Better metrics  

A strong ML practitioner chooses a combination of these depending on the problem (e.g., fraud detection, medical diagnosis).

---

[Back to Table of Contents](#table-of-contents)