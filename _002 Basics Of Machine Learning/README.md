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