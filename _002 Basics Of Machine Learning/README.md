## Basic Of Machine Learning


## Hierarchy & Relationship

Artificial Intelligence (AI) → The broadest field. Goal: make machines simulate human intelligence (reasoning, decision-making, problem-solving).  

Machine Learning (ML) → A subset of AI. Machines learn patterns from data to make predictions or decisions without explicit programming.  

Deep Learning (DL) → A specialized subset of ML. Uses multi-layered neural networks to automatically extract features, especially effective for unstructured data (images, text, audio).  

Data Science (DS) → The overarching discipline. Involves the entire process of collecting, cleaning, analyzing, modeling, and communicating insights from data. Uses AI/ML/DL as tools.  

________________________________________  

## Comparison Table

| Aspect | AI | ML | DL | DS |
|-------|----|----|----|----|
| Definition | Broad goal: simulate human intelligence | Subset of AI: learn from data | Subset of ML: neural networks | Multidisciplinary process of extracting insights |
| Goal | Mimic cognition (rules, logic, or data-driven) | Build predictive models | Handle complex tasks (vision, NLP) | Turn raw data into actionable knowledge |
| Techniques | Rules, expert systems, ML/DL | Regression, classification, clustering | CNNs, RNNs, Transformers | Statistics, ML/DL, visualization |
| Data Needs | Can work with rules + data | Requires structured data | Needs massive unstructured data | Works with all data types |
| Interpretability | High (rules are explicit) | Moderate | Low (black-box models) | High (focus on communication & insights) |
| Applications | Robotics, expert systems, planning | Fraud detection, recommendation engines | Image recognition, speech, NLP | Business analytics, dashboards, decision support |

________________________________________  

## Practical Example

Imagine building a smart healthcare system:

AI: The system mimics a doctor’s reasoning to suggest treatments.  

ML: It learns from patient records to predict disease risk.  

DL: It analyzes MRI scans using CNNs to detect tumors.  

DS: It integrates patient data, builds models, and communicates insights to doctors and hospital management.  

________________________________________  

## Key Takeaway

AI is the vision.  
ML is the method.  
DL is the cutting-edge technique.  
DS is the practice that orchestrates everything to deliver value.  

---

## Types of ML:

- Supervised Learning  
- Unsupervised Learning  
- Semi-Supervised Learning  
- Reinforcement Learning  

---

## Types of ML Explained in detail:

## Supervised Learning 
Supervised learning is a machine learning paradigm where an algorithm learns a mapping function $$f(X) \rightarrow Y$$ from labeled training data — input-output pairs $$(X, y)$$ where $$Y$$ (the label/target) is known during training.  

Analogy: Think of a teacher (the labeled data) guiding a student (the model). The student sees examples with correct answers and learns generalizable patterns.  

Formal definition: Given training data $$\{(x_1,y_1), (x_2,y_2), ..., (x_n,y_n)\}$$, learn a hypothesis $$h: X \rightarrow Y$$ that minimizes prediction error on unseen data.  

---

## Unsupervised Learning

Unsupervised learning is a machine learning paradigm where an algorithm learns patterns, structures, or representations from unlabeled data — input data $$X$$ without corresponding output labels $$Y$$.  

Analogy: Think of a student exploring a new city without a guide. They start grouping places based on similarity (e.g., markets, parks, residential areas) without being told explicitly.  

Formal definition: Given training data $$\{x_1, x_2, ..., x_n\}$$, learn a representation or structure $$h: X \rightarrow Z$$ that captures underlying patterns such as clusters, density, or latent features in the data.  

---

## Semi-Supervised Learning

Semi-supervised learning is a machine learning paradigm where an algorithm learns from a combination of a small amount of labeled data and a large amount of unlabeled data, leveraging both to improve learning performance.  

Analogy: Think of a student who has access to a few solved examples (labeled data) and many unsolved problems (unlabeled data). The student uses the solved ones to infer patterns and apply them to the rest.  

Formal definition: Given a labeled dataset $$\{(x_1,y_1), ..., (x_l,y_l)\}$$ and an unlabeled dataset $$\{x_{l+1}, ..., x_n\}$$, learn a hypothesis $$h: X \rightarrow Y$$ that utilizes both labeled and unlabeled data to minimize generalization error.  

---

## Reinforcement Learning

Reinforcement learning is a machine learning paradigm where an agent learns to make sequential decisions by interacting with an environment and receiving rewards or penalties as feedback.  

Analogy: Think of training a dog. When the dog performs a desired action, it gets a treat (reward); otherwise, it does not. Over time, the dog learns which actions maximize rewards.  

Formal definition: Given an environment modeled as a Markov Decision Process (MDP), the agent learns a policy $$\pi: S \rightarrow A$$ that maximizes the expected cumulative reward:

$$
E\left[\sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

where $$r_t$$ is the reward at time $$t$$ and $$\gamma \in [0,1]$$ is the discount factor.

---

## Further Deep Dive into Learning Approaches

---

## Subdivisions of Supervised Learning

### Definition
Supervised learning can be subdivided based on the type of output variable $$Y$$ the model is trying to predict.

---

### 1. Classification

**Definition:**
A task where the model predicts discrete/categorical labels.

**Analogy:**
Like sorting emails into “spam” or “not spam”.

**Formal definition:**

Given:

$$
Y \in \{1,2,...,K\}
$$

Learn:

$$
h: X \rightarrow Y
$$

---

### 2. Regression

**Definition:**
A task where the model predicts continuous numerical values.

**Analogy:**
Predicting house prices based on size and location.

**Formal definition:**

Given:

$$
Y \in \mathbb{R}
$$

Learn:

$$
h: X \rightarrow \mathbb{R}
$$

---

### 3. Ranking (Optional Advanced Subdivision)

**Definition:**
Predicts relative ordering instead of exact values.

**Analogy:**
Like ranking search results on Google.

**Formal definition:**

Learn a scoring function:

$$
h: X \rightarrow \mathbb{R}
$$

such that ordering is preserved:

$$
h(x_i) > h(x_j)
$$

---

## Subdivisions of Unsupervised Learning

### Definition
Unsupervised learning is subdivided based on the type of structure or pattern extracted from unlabeled data.

---

### 1. Clustering

**Definition:**
Group similar data points together.

**Analogy:**
Grouping customers based on buying behavior.

**Formal definition:**

Partition dataset:

$$
D = \{x_1, ..., x_n\}
$$

into clusters:

$$
C_1, C_2, ..., C_k
$$

such that intra-cluster similarity is maximized.

---

### 2. Dimensionality Reduction

**Definition:**
Reduce the number of features while preserving important information.

**Analogy:**
Summarizing a long book into key points.

**Formal definition:**

$$
f: \mathbb{R}^d \rightarrow \mathbb{R}^k, \quad k < d
$$

---

### 3. Density Estimation

**Definition:**
Estimate the probability distribution of data.

**Analogy:**
Understanding how data is spread (e.g., where most customers lie).

**Formal definition:**

$$
P(X)
$$

---

### 4. Association Rule Learning

**Definition:**
Discover relationships between variables.

**Analogy:**
“People who buy bread also buy butter.”

**Formal definition:**

Find rules:

$$
X \Rightarrow Y
$$

with support and confidence constraints.

---

## Subdivisions of Semi-Supervised Learning

### Definition
Semi-supervised learning combines labeled and unlabeled data and is subdivided based on how unlabeled data is used.

---

### 1. Self-Training

**Definition:**
Model trains on labeled data, then labels unlabeled data itself and retrains.

**Analogy:**
A student solving unsolved problems using patterns learned from solved ones.

**Formal definition:**

$$
L \leftarrow L \cup \hat{U}
$$

---

### 2. Co-Training

**Definition:**
Two models trained on different feature subsets teach each other.

**Analogy:**
Two students with different strengths helping each other learn.

**Formal definition:**

Two hypotheses:

$$
h_1, h_2
$$

label data for each other iteratively.

---

### 3. Graph-Based Methods

**Definition:**
Use graph structure to propagate labels.

**Analogy:**
Friends influencing each other in a network.

**Formal definition:**

$$
\min_{(i,j) \in E} \sum w_{ij}(y_i - y_j)^2
$$

---

## Subdivisions of Reinforcement Learning

### Definition
Reinforcement learning is subdivided based on how the agent learns policies and interacts with the environment.

---

### 1. Model-Free Learning

**Definition:**
Learns policy directly without modeling environment dynamics.

**Analogy:**
Learning to play a game by trial and error without knowing rules explicitly.

**Formal definition:**

Learn:

$$
Q(s,a) \quad \text{or} \quad \pi(a \mid s)
$$

without estimating transition probabilities.

---

### 2. Model-Based Learning

**Definition:**
Learns a model of the environment and uses it for planning.

**Analogy:**
Planning chess moves by simulating future positions.

**Formal definition:**

Learn:

$$
P(s' \mid s,a), \quad R(s,a)
$$

then optimize policy.

---

### 3. Policy-Based Methods

**Definition:**
Directly learn the policy function.

**Analogy:**
Learning strategy directly rather than evaluating each move.

**Formal definition:**

$$
\pi_\theta(a \mid s)
$$

---

### 4. Value-Based Methods

**Definition:**
Learn value functions to derive policy.

**Analogy:**
Choosing actions based on expected future rewards.

**Formal definition:**

$$
Q(s,a) = \mathbb{E}[R]
$$

---

### 5. Actor-Critic Methods

**Definition:**
Combine policy (actor) and value (critic) learning.

**Analogy:**
Actor makes decisions, critic evaluates them.

**Formal definition:**

- Actor: $$\pi_\theta(a \mid s)$$  
- Critic: $$V(s) \text{ or } Q(s,a)$$  

---

## Big Picture

- Supervised → Predict outputs (classification/regression)  
- Unsupervised → Discover structure (clustering, patterns)  
- Semi-supervised → Use limited labels smartly  
- Reinforcement → Learn via interaction and rewards  

---

## Evaluation Metrics Across Learning Paradigms

---

## 1. Supervised Learning

### Definition
Evaluation metrics in supervised learning measure how close predictions $$\hat{y}$$ are to true labels $$y$$.

---

### Classification Metrics

**Analogy:**
Like evaluating an exam not just by total marks, but also by how many correct answers, mistakes, and missed questions.

---

## Confusion Matrix

### Definition

A confusion matrix is a table used to evaluate classification models by comparing actual labels vs predicted labels, showing how many predictions are correct and where errors occur.

---

### Analogy

Think of a teacher checking exam answers:

- Correct answers → True predictions  
- Wrong answers → Errors  

The confusion matrix is like a detailed report card showing what kind of mistakes were made.

---

### Structure (Binary Classification)

|                | Predicted Positive | Predicted Negative |
|----------------|------------------|-------------------|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

---

### Meaning of Each Term

- **True Positive (TP):** Model correctly predicts positive  
- **True Negative (TN):** Model correctly predicts negative  
- **False Positive (FP):** Model predicts positive but actually negative (Type I error)  
- **False Negative (FN):** Model predicts negative but actually positive (Type II error)  

---

### Formal Definition

Let:

- $$y$$ = actual label  
- $$\hat{y}$$ = predicted label  

Then:

$$
TP = \{y = 1, \hat{y} = 1\}
$$

$$
TN = \{y = 0, \hat{y} = 0\}
$$

$$
FP = \{y = 0, \hat{y} = 1\}
$$

$$
FN = \{y = 1, \hat{y} = 0\}
$$

---

### Metrics Derived from Confusion Matrix

---

#### Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

- Measures overall correctness  
- Misleading for imbalanced datasets  

**Finance Example**

- Loan Approval Model  
- Predict: Default / No Default  

- Accuracy = % of correct predictions  

**Problem:**  
- If 95% customers don’t default → model predicting “No Default” always gives 95% accuracy → misleading  

---

#### Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

- Of predicted positives, how many are correct  

**Finance Example**

Fraud Detection  

Precision = Of all flagged frauds, how many are actually fraud?  

- High precision → fewer false fraud alerts  
- Important for reducing customer inconvenience  
---

#### Recall (Sensitivity)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

- Of actual positives, how many are captured  

**Finance Example**

Credit Card Fraud Detection  

Recall = How many actual fraud cases are caught?  

- High recall → fewer frauds missed  
- Critical because missing fraud = direct financial loss  

---

#### F1 Score

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

- Balance between precision and recall  

**Finance Example**

Used when both false positives & false negatives matter  

Example:  
Fraud detection systems balancing:  
- Customer friction (FP)  
- Financial loss (FN) 

---

#### Specificity

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

- Ability to correctly identify negatives  

---

#### ROC-AUC
- Measures model’s ability to separate classes across thresholds  

The ROC curve is a graphical tool used to evaluate the performance of a classification model across different thresholds.

#### Key Idea

It shows the trade-off between:

#### True Positive Rate (TPR / Recall / Sensitivity)

$$
TPR = \frac{TP}{TP + FN}
$$

#### False Positive Rate (FPR)

$$
FPR = \frac{FP}{FP + TN}
$$

---

#### Interpretation

- Each point on the ROC curve = a different classification threshold  
- A good model:
  - High TPR (detects positives well)  
  - Low FPR (few false alarms)  
- The closer the curve is to the **top-left corner**, the better  

---

#### AUC (Area Under Curve)

- AUC = 1 → Perfect model  
- AUC = 0.5 → Random guessing  
- Higher AUC = better separability  

---

#### Intuition

ROC answers:

**“How well can the model distinguish between classes regardless of threshold?”**


---

### B. Regression Metrics

**Analogy:**
Like measuring how far your predicted marks are from actual marks.

---

#### Mean Squared Error (MSE)

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

---

#### Root Mean Squared Error (RMSE)

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

---

#### Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|
$$

---

### R² (Coefficient of Determination)

### Definition

R² measures how well a regression model explains the variance in the target variable $$Y$$ using input features $$X$$.

---

### Analogy

Imagine trying to explain a student’s marks.

- If your model explains almost all variation in marks → high R²  
- If it explains nothing → low R²  

It tells: “How much of the story your model is able to explain.”

---

### Formal Definition

$$
R^2 = 1 - \frac{\text{Unexplained Variance}}{\text{Total Variance}}
$$

---

### Expanded Form

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

Where:

- $$\sum (y_i - \hat{y}_i)^2$$ = Residual Sum of Squares (RSS)  
- $$\sum (y_i - \bar{y})^2$$ = Total Sum of Squares (TSS)  

---

### Key Insight

- $$R^2 = 1$$ → Perfect fit  
- $$R^2 = 0$$ → Model is no better than mean  
- $$R^2 < 0$$ → Model is worse than mean  

---

### Problem with R²

### Definition

R² always increases (or stays same) when you add more features—even if those features are useless.

---

### Analogy

Like adding more variables in an exam explanation—even irrelevant ones—your explanation looks better numerically but not actually meaningful.

---

### Adjusted R²

### Definition

Adjusted R² corrects R² by penalizing unnecessary features, giving a more realistic measure of model performance.

---

### Analogy

Think of a strict teacher who only rewards meaningful explanations and penalizes unnecessary extra points.

---

### Formal Definition

$$
\text{Adjusted } R^2 = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - p - 1} \right)
$$

Where:

- $$n$$ = number of data points  
- $$p$$ = number of features  

---

### Key Insight

- Increases only if new feature improves model genuinely  
- Decreases if feature is irrelevant  
- Helps in feature selection  

---

### R² vs Adjusted R² (Core Difference)

| Aspect | R² | Adjusted R² |
|--------|----|-------------|
| Feature addition | Always increases | Increases only if useful |
| Penalization | No | Yes |
| Reliability | Can be misleading | More reliable |
| Use case | Basic evaluation | Model comparison |

---

## 2. Unsupervised Learning

### Definition
Since there are no true labels, evaluation focuses on structure, similarity, or distribution quality.

---

### A. Clustering Metrics

**Analogy:**
Like checking whether groups formed in a classroom actually make sense.

---

#### Silhouette Score

$$
S = \frac{b - a}{\max(a,b)}
$$

- $$a$$: intra-cluster distance  
- $$b$$: nearest-cluster distance  

---

#### Davies–Bouldin Index

$$
DB = \frac{1}{k} \sum \max \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)
$$

- Lower is better  

---

#### Calinski-Harabasz Index
- Ratio of between-cluster to within-cluster variance  
- Higher is better  

---

### B. Dimensionality Reduction Metrics

#### Reconstruction Error

$$
\|X - \hat{X}\|^2
$$

- Measures information loss  

---

#### Explained Variance (PCA)
- Fraction of variance retained  

---

### C. Density Estimation Metrics

#### Log-Likelihood

$$
\sum \log P(x_i)
$$

- Higher is better  

---

## 3. Semi-Supervised Learning

### Definition
Evaluation is typically done using supervised metrics, since final predictions are on labeled data.

---

### Metrics Used

- Classification → Accuracy, Precision, Recall, F1, ROC-AUC  
- Regression → MSE, RMSE, MAE  

---

### Additional Consideration

#### Label Efficiency
- How well the model performs with limited labeled data  

---

## 4. Reinforcement Learning

### Definition
Evaluation measures how well an agent maximizes cumulative reward over time.

---

### Key Metrics

#### Cumulative Reward

$$
G_t = \sum_{t=0}^{\infty} \gamma^t r_t
$$

- Total reward obtained  

---

#### Average Reward

$$
\frac{1}{T} \sum_{t=1}^{T} r_t
$$

---

#### Policy Performance
- Expected return under policy $$\pi$$  

---

#### Regret

$$
\text{Regret} = \text{Optimal Reward} - \text{Obtained Reward}
$$

- Measures how far agent is from optimal  

---

## Types of Model Learning Approaches

### 1. Instance-Based Learning (Lazy Learning)

**Definition:**
Instance-based learning is a machine learning approach where the algorithm stores training examples and makes predictions by comparing new inputs directly to these stored instances, rather than learning an explicit model.

**Analogy:**
Think of a doctor who doesn’t memorize general medical theory but instead recalls past patient cases. When a new patient arrives, the doctor finds similar past cases and decides treatment based on those.

**Formal definition:**
Given training data $$\{(x_1,y_1), (x_2,y_2), ..., (x_n,y_n)\}$$, prediction for a new input $$x$$ is made as:

$$
\hat{y} = \text{aggregate}\left(\{y_i \mid x_i \in N(x)\}\right)
$$

where $$N(x)$$ represents the set of nearest neighbors of $$x$$ based on a distance metric $$d(x,x_i)$$.

**Characteristics:**
- Lazy learning (no explicit model built upfront).  
- High memory usage (stores many examples).  
- Fast training, slower prediction.  

**Examples:**
- k-Nearest Neighbors (k-NN).  
- Case-based reasoning.  

**Use Cases:**
- Recommendation systems.  
- Pattern recognition when decision boundaries are irregular. 
---

### 2. Model-Based Learning (Eager Learning)

**Definition:**
Model-based learning is a machine learning approach where the algorithm learns a general function (model) from training data and uses this function to make predictions on new data.

**Analogy:**
Think of a student who studies concepts and formulas instead of memorizing examples. Once learned, they can solve new problems quickly using general rules.

**Formal definition:**
Given training data $$\{(x_1,y_1), ..., (x_n,y_n)\}$$, learn a function:

$$
h_\theta : X \rightarrow Y
$$

by optimizing parameters $$\theta$$:

$$
\theta^* = \arg\min_{\theta} \sum_{i=1}^{n} L\left(y_i, h_\theta(x_i)\right)
$$

where $$L$$ is a loss function.

**Characteristics:**
- Eager learning (model built before prediction).  
- Compact representation (doesn’t need to store all data).  
- Training can be computationally expensive, but prediction is fast.  

**Examples:**
- Linear regression, logistic regression.  
- Decision trees, SVMs.  
- Neural networks.  

**Use Cases:**
- Predictive analytics.  
- Classification tasks with large datasets.  

---

### 3. Memory-Based / Hybrid Approaches (Other Learning Styles)

#### a) Memory-Based Learning

**Definition:**
A learning paradigm where the system stores past experiences and reuses them directly for future predictions, often overlapping with instance-based learning.

**Analogy:**
Like solving a coding problem by remembering similar problems you’ve already solved.

**Formal definition:**

$$
\hat{y} = f(x, D)
$$

---

#### b) Rule-Based Learning

**Definition:**
A learning approach where the model learns explicit IF–THEN rules from data.

**Analogy:**
Like following traffic rules: “IF signal is red → STOP”.

**Formal definition:**
Learn a set of rules:

$$
R = \{(condition_i \Rightarrow outcome_i)\}
$$

such that predictions follow:

$$
y = outcome_i \quad \text{if } condition_i(x) \text{ is true}
$$

---

#### c) Probabilistic Learning

**Definition:**
A learning approach where the model captures uncertainty using probability distributions.

**Analogy:**
Like predicting rain with “70% chance” instead of a definite yes/no.

**Formal definition:**
Model the conditional probability:

$$
P(Y \mid X)
$$

and predict:

$$
\hat{y} = \arg\max_{y} P(y \mid x)
$$

---

#### d) Evolutionary / Genetic Learning

**Definition:**
A learning approach where solutions are evolved over generations using selection, mutation, and crossover, inspired by biological evolution.

**Analogy:**
Like natural selection where only the fittest solutions survive and improve over time.

**Formal definition:**
Optimize a population of candidate solutions $$\{s_1, ..., s_k\}$$ using:

$$
s^{(t+1)} = \text{evolve}(s^{(t)})
$$

based on a fitness function $$F(s)$$.

---

#### e) Online Learning

**Definition:**
A learning paradigm where the model updates continuously as new data arrives, instead of training once on a fixed dataset.

**Analogy:**
Like learning from daily experiences instead of studying everything at once.

**Formal definition:**
Given sequential data $$(x_t, y_t)$$, update model parameters iteratively:

$$
\theta_{t+1} = \theta_t - \eta \nabla L\left(y_t, h_{\theta_t}(x_t)\right)
$$

---

### 4. Lazy Learning

**Definition:**
Lazy learning is a strategy where the algorithm delays generalization until a prediction is required, storing training data instead of building a model upfront.

**Analogy:**
Like preparing for an exam only when the question paper is given.

**Formal definition:**
No explicit hypothesis $$h$$ is formed during training; prediction depends directly on dataset $$D$$:

$$
\hat{y} = f(x, D)
$$

### Characteristics
- No explicit training phase (or very minimal).  
- High memory usage (stores many or all training examples).  
- Fast training, but slow prediction (computation happens at query time).  
- Local generalization (decisions depend on nearby examples rather than a global model).

---

### 5. Eager Learning

**Definition:**
Eager learning is a strategy where the algorithm builds a general model during training before any predictions are made.

**Analogy:**
Like studying all concepts before the exam so you can answer quickly during the test.

**Formal definition:**
Learn a hypothesis $$h$$ during training:

$$
h = \arg\min_{h \in H} \sum_{i=1}^{n} L\left(y_i, h(x_i)\right)
$$

and use it for prediction:

$$
\hat{y} = h(x)
$$

### Characteristics
- Explicit training phase (model is built before prediction).  
- Lower memory usage (stores parameters, not all data).  
- Training can be computationally expensive.  
- Predictions are fast (apply the pre-built model).  
- Global generalization (learns overall patterns, not just local neighborhoods).  

---

## Mathematical Representation of Decision Boundaries

| Object | Equation Form | Key Feature |
|-------|--------------|-------------|
| Line (2D) | $$y = mx + c \quad \text{or} \quad ax + by + c = 0$$ | Defined by slope/intercept |
| Plane (3D) | $$ax + by + cz + d = 0$$ | Normal vector defines orientation |
| Line (3D) | $$x = x_0 + \lambda l,\; y = y_0 + \lambda m,\; z = z_0 + \lambda n$$ | Point + direction vector |
| Hyperplane (nD) | $$\sum w_i x_i + b = 0$$ | Generalization of plane in n-D |

---


## Parametric vs Non-Parametric Algorithms

---

### 1. Parametric Algorithms

#### Definition
Parametric algorithms are machine learning models that assume a fixed functional form for the relationship between input $$X$$ and output $$Y$$ and summarize this relationship using a finite set of parameters.

---

#### Analogy
Think of fitting a straight line using a ruler. No matter how complex the data is, you are constrained to use a straight line (fixed shape), and you just adjust its slope and position.

---

#### Formal Definition

Assume a predefined function:

$$
h_\theta(x) = f(x; \theta)
$$

where:
- $$f$$ is a fixed functional form (e.g., linear)  
- $$\theta$$ is a finite set of parameters  

Learn:

$$
\theta^* = \arg\min_{\theta} \sum_{i=1}^{n} L\left(y_i, h_\theta(x_i)\right)
$$

---

#### Key Characteristics
- Strong assumptions (e.g., linearity, Gaussian distribution)  
- Fixed number of parameters (independent of dataset size)  
- Faster training and prediction  
- Works well with smaller datasets  
- Less flexible (risk of underfitting if assumptions are wrong)  

---

#### Examples
- Linear Regression  
- Logistic Regression  
- Naive Bayes  

---

### 2. Non-Parametric Algorithms

#### Definition
Non-parametric algorithms are machine learning models that do not assume a fixed functional form and instead adapt their complexity based on the data, often requiring storage of training examples or flexible structures.

---

#### Analogy
Instead of using a ruler, imagine using a flexible wire that bends to fit the shape of the data points. The more data you have, the more accurately it can adapt.

---

#### Formal Definition

Learn a function:

$$
h(x) = f(x, D)
$$

where:
- $$D$$ is the training dataset  
- Model complexity grows with $$|D|$$  

No fixed parameter size:

$$
\text{Number of parameters} \rightarrow \infty \quad \text{as} \quad n \rightarrow \infty
$$

---

#### Key Characteristics
- Minimal assumptions about data distribution  
- Model complexity grows with data  
- Highly flexible (captures non-linear patterns)  
- Requires more data to generalize well  
- Higher computational and memory cost  

---

#### Examples
- k-Nearest Neighbors (k-NN)  
- Decision Trees  
- Random Forest  
- Support Vector Machines (with non-linear kernels)  
- Kernel Density Estimation (KDE)  

---

### 3. Core Intuition (Very Important)

- Parametric = Learn parameters → compress data into a fixed-size representation  
- Non-parametric = Learn structure/data itself → rely on data to define model  

---

### 4. Comparison Table

| Aspect | Parametric Algorithms | Non-Parametric Algorithms |
|--------|----------------------|--------------------------|
| Assumptions | Strong (fixed form) | Minimal |
| Parameters | Fixed, finite | Grow with data |
| Flexibility | Low | High |
| Data Requirement | Low | High |
| Computation | Fast | Slower |
| Memory Usage | Low | High |
| Risk | Underfitting | Overfitting |

---

### 5. When to Use

#### Parametric Algorithms
- When you know or assume the data distribution  
- When you need fast, interpretable models  
- When data is limited  

#### Non-Parametric Algorithms
- When the relationship is complex or non-linear  
- When you don’t know the data distribution  
- When you have large datasets  

---

### Final Insight

The choice between parametric and non-parametric models is fundamentally a trade-off between:

- Bias (parametric → high bias, low variance)  
- Variance (non-parametric → low bias, high variance)  

In practice, modern machine learning (like deep learning) often blends both ideas—fixed architectures (parametric) with high flexibility (non-linear function approximation).

---

