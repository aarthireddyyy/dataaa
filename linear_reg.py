import numpy as np

class LinearRegression:
    """
    Production-minded Linear Regression from scratch.
    
    Every line here has a reason. Read the comments.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []  # You ALWAYS track this in production
    
    def fit(self, X, y):
        """
        X: shape (n_samples, n_features)
        y: shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights to zero
        # Question: Could you initialize randomly? Does it matter here?
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for iteration in range(self.n_iterations):
            
            # --- FORWARD PASS ---
            # Compute predictions: shape (n_samples,)
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # --- COMPUTE LOSS ---
            # MSE: scalar value
            loss = np.mean((y - y_predicted) ** 2)
            self.loss_history.append(loss)
            
            # --- COMPUTE GRADIENTS ---
            # How much does each weight need to change?
            # dL/dw — shape (n_features,)
            dw = (-2 / n_samples) * np.dot(X.T, (y - y_predicted))
            
            # dL/db — scalar
            db = (-2 / n_samples) * np.sum(y - y_predicted)
            
            # --- UPDATE WEIGHTS ---
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Log every 100 iterations — this habit saves you in production
            if iteration % 100 == 0:
                print(f"Iteration {iteration} | Loss: {loss:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Inference — this is what runs in production.
        Pure matrix multiply. Fast. Stateless.
        """
        return np.dot(X, self.weights) + self.bias
    
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def rmse(self, y_true, y_pred):
        return np.sqrt(self.mse(y_true, y_pred))
    
    def r_squared(self, y_true, y_pred):
        """
        R² tells you: how much of the variance in y does your model explain?
        R² = 1.0 → perfect
        R² = 0.0 → no better than predicting the mean
        R² < 0.0 → worse than predicting the mean (your model is broken)
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
# Generate synthetic house price data
np.random.seed(42)
n_samples = 500

# Features: [sqft, num_rooms, age_of_house]
sqft      = np.random.uniform(500, 5000, n_samples)
rooms     = np.random.randint(1, 8, n_samples).astype(float)
age       = np.random.uniform(0, 50, n_samples)

X = np.column_stack([sqft, rooms, age])

# True relationship (what we're trying to recover)
# Price = 150*sqft + 10000*rooms - 500*age + 50000 + noise
true_weights = np.array([150, 10000, -500])
true_bias = 50000
noise = np.random.normal(0, 15000, n_samples)

y = X @ true_weights + true_bias + noise

# --- Feature Scaling (CRITICAL — don't skip this) ---
# sqft ranges 500–5000, rooms ranges 1–8, age ranges 0–50
# Without scaling, the sqft gradient dominates everything
# Your model will converge badly or not at all

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

y_mean = y.mean()
y_std  = y.std()
y_scaled = (y - y_mean) / y_std

# Train
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_scaled, y_scaled)

# Evaluate
y_pred_scaled = model.predict(X_scaled)
y_pred = y_pred_scaled * y_std + y_mean  # Inverse transform

print(f"\nR²:   {model.r_squared(y, y_pred):.4f}")
print(f"RMSE: ${model.rmse(y, y_pred):,.0f}")
print(f"\nLearned weights: {model.weights}")
print(f"Learned bias:    {model.bias:.4f}")