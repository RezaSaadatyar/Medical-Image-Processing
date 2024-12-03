Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It is widely known for its ability to handle high-dimensional data and perform well with a clear margin of separation.

* * *

### **Key Concepts of SVM**

1.  **Hyperplane**: In SVM, a hyperplane is the decision boundary that separates different classes in the feature space. For a dataset with nnn features, the hyperplane is an (n−1)(n-1)(n−1)\-dimensional subspace.
    
2.  **Margin**: The margin is the distance between the hyperplane and the nearest data points (support vectors) from either class. SVM aims to maximize this margin to ensure better generalization.
    
3.  **Support Vectors**: These are the data points closest to the hyperplane. They are critical because they determine the position and orientation of the hyperplane.
    
4.  **Linear Separability**: If the data can be separated by a straight line (or a hyperplane in higher dimensions), it is linearly separable.
    
5.  **Kernel Trick**: For non-linearly separable data, SVM uses kernel functions to project the data into a higher-dimensional space where a hyperplane can separate the classes.
    

* * *

### **Mathematical Formulation**

SVM tries to solve the following optimization problem:

#### **Objective: Maximize the Margin**

For a linearly separable dataset, the hyperplane can be represented as:

w⋅x+b\=0w \\cdot x + b = 0w⋅x+b\=0

where:

*   www is the weight vector normal to the hyperplane,
*   xxx is the input feature vector,
*   bbb is the bias term.

The decision function for classification is:

f(x)\=sign(w⋅x+b)f(x) = \\text{sign}(w \\cdot x + b)f(x)\=sign(w⋅x+b)

#### **Constraints**:

For a dataset {(xi,yi)}\\{(x\_i, y\_i)\\}{(xi​,yi​)}, where xix\_ixi​ is the input and yi∈{−1,+1}y\_i \\in \\{-1, +1\\}yi​∈{−1,+1} is the class label, the constraints are:

yi(w⋅xi+b)≥1,∀iy\_i (w \\cdot x\_i + b) \\geq 1, \\quad \\forall iyi​(w⋅xi​+b)≥1,∀i

#### **Optimization Problem**:

Minimize:

12∣∣w∣∣2\\frac{1}{2} ||w||^221​∣∣w∣∣2

subject to the constraints:

yi(w⋅xi+b)≥1y\_i (w \\cdot x\_i + b) \\geq 1yi​(w⋅xi​+b)≥1

This optimization ensures a maximum margin by minimizing the norm of the weight vector (∣∣w∣∣||w||∣∣w∣∣).

* * *

### **Non-Linearly Separable Data**

For non-linear cases, SVM introduces:

1.  **Soft Margin**: Allows some misclassifications by introducing a slack variable ξi\\xi\_iξi​:
    
    yi(w⋅xi+b)≥1−ξi,ξi≥0y\_i (w \\cdot x\_i + b) \\geq 1 - \\xi\_i, \\quad \\xi\_i \\geq 0yi​(w⋅xi​+b)≥1−ξi​,ξi​≥0
    
    The objective becomes:
    
    min⁡12∣∣w∣∣2+C∑i\=1nξi\\min \\frac{1}{2} ||w||^2 + C \\sum\_{i=1}^n \\xi\_imin21​∣∣w∣∣2+Ci\=1∑n​ξi​
    
    where CCC controls the trade-off between maximizing the margin and minimizing misclassification errors.
    
2.  **Kernel Function**: Maps data to a higher-dimensional space to make it linearly separable. Common kernels include:
    
    *   Linear Kernel: K(xi,xj)\=xi⋅xjK(x\_i, x\_j) = x\_i \\cdot x\_jK(xi​,xj​)\=xi​⋅xj​
    *   Polynomial Kernel: K(xi,xj)\=(xi⋅xj+c)dK(x\_i, x\_j) = (x\_i \\cdot x\_j + c)^dK(xi​,xj​)\=(xi​⋅xj​+c)d
    *   Gaussian Radial Basis Function (RBF): K(xi,xj)\=exp⁡(−γ∣∣xi−xj∣∣2)K(x\_i, x\_j) = \\exp(-\\gamma ||x\_i - x\_j||^2)K(xi​,xj​)\=exp(−γ∣∣xi​−xj​∣∣2)
    *   Sigmoid Kernel: K(xi,xj)\=tanh⁡(αxi⋅xj+c)K(x\_i, x\_j) = \\tanh(\\alpha x\_i \\cdot x\_j + c)K(xi​,xj​)\=tanh(αxi​⋅xj​+c)

* * *

### **Dual Formulation**

The SVM problem is often solved in its dual form using Lagrange multipliers:

max⁡∑i\=1nαi−12∑i\=1n∑j\=1nαiαjyiyjK(xi,xj)\\max \\sum\_{i=1}^n \\alpha\_i - \\frac{1}{2} \\sum\_{i=1}^n \\sum\_{j=1}^n \\alpha\_i \\alpha\_j y\_i y\_j K(x\_i, x\_j)maxi\=1∑n​αi​−21​i\=1∑n​j\=1∑n​αi​αj​yi​yj​K(xi​,xj​)

subject to:

∑i\=1nαiyi\=0,0≤αi≤C\\sum\_{i=1}^n \\alpha\_i y\_i = 0, \\quad 0 \\leq \\alpha\_i \\leq Ci\=1∑n​αi​yi​\=0,0≤αi​≤C

Here:

*   αi\\alpha\_iαi​ are Lagrange multipliers,
*   K(xi,xj)K(x\_i, x\_j)K(xi​,xj​) is the kernel function.

* * *

### **Decision Rule**

After solving for www and bbb, the decision rule for classifying a new input xxx is:

f(x)\=sign(∑i\=1nαiyiK(xi,x)+b)f(x) = \\text{sign} \\left( \\sum\_{i=1}^n \\alpha\_i y\_i K(x\_i, x) + b \\right)f(x)\=sign(i\=1∑n​αi​yi​K(xi​,x)+b)

* * *

### **Advantages**

*   Effective in high-dimensional spaces.
*   Works well with both linear and non-linear boundaries.
*   Robust to overfitting with proper parameter tuning.

### **Limitations**

*   Computationally expensive for large datasets.
*   Sensitive to the choice of the kernel and its parameters.
*   Performance can degrade with noisy data or overlapping classes.

* * *

SVM is a powerful tool that balances complexity and predictive accuracy, making it a go-to algorithm for many classification and regression problems.