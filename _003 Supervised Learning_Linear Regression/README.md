## Linear Regression

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

### Summary Insight

Linear regression is essentially about finding the best straight line (or hyperplane) that captures the relationship between inputs and outputs by minimizing prediction error or mean squared error by using optimization techniques like gradient descent. It is simple, interpretable, and forms the foundation for many advanced machine learning models.