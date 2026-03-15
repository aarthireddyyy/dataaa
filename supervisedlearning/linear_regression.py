#Scratch 
import numpy as np

class LinearRegression:
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        
        # Initialize weights and bias to zero
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for i in range(self.n_iterations):
            
            # Step 1 - predict
            y_predicted = self.predict(X)
            
            # Step 2 - compute loss
            loss = self.compute_loss(y, y_predicted)
            self.loss_history.append(loss)
            
            # Step 3 - compute gradients
            dw = (-2 / n_samples) * np.dot(X.T, (y - y_predicted))
            db = (-2 / n_samples) * np.sum(y - y_predicted)
            
            # Step 4 - update weights
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db
            
            # Step 5 - log progress
            if i % 100 == 0:
                print(f"Iteration {i} | Loss: {loss:.4f}")
        
        return self

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.squeeze() + np.random.randn(100)

# Train
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# Results
print(f"\nFinal weight : {model.weights[0]:.4f}  — should be close to 3")
print(f"Final bias   : {model.bias:.4f}        — should be close to 4")