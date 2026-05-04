import numpy as np

# ============================================================
#  Simple Neural Network: Forward + Backward Propagation
#  Architecture: Input(2) -> Hidden(2) -> Output(1)
# ============================================================

# ---------- Activation Function ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)           # d(sigmoid)/dz

# ---------- Loss Function ----------
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# ============================================================
#  INITIALIZE WEIGHTS & BIASES
# ============================================================
np.random.seed(42)

# Layer 1: Input(2) -> Hidden(2)
W1 = np.random.randn(2, 2) * 0.5   # shape (2, 2)
b1 = np.zeros((1, 2))               # shape (1, 2)

# Layer 2: Hidden(2) -> Output(1)
W2 = np.random.randn(2, 1) * 0.5   # shape (2, 1)
b2 = np.zeros((1, 1))               # shape (1, 1)

print("=" * 50)
print("  Initial Weights")
print("=" * 50)
print(f"W1:\n{W1}\nb1: {b1}")
print(f"W2:\n{W2}\nb2: {b2}")

# ============================================================
#  TRAINING DATA  (XOR problem)
# ============================================================
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])   # shape (4, 2)

y = np.array([[0],
              [1],
              [1],
              [0]])       # shape (4, 1)

# ============================================================
#  FORWARD PROPAGATION
# ============================================================
def forward(X, W1, b1, W2, b2):
    """
    Step 1: Input -> Hidden
      Z1 = X @ W1 + b1       (linear)
      A1 = sigmoid(Z1)       (activation)

    Step 2: Hidden -> Output
      Z2 = A1 @ W2 + b2      (linear)
      A2 = sigmoid(Z2)       (activation / prediction)
    """
    # ----- Layer 1 -----
    Z1 = X @ W1 + b1         # (4,2) @ (2,2) = (4,2)
    A1 = sigmoid(Z1)          # (4,2)

    # ----- Layer 2 -----
    Z2 = A1 @ W2 + b2        # (4,2) @ (2,1) = (4,1)
    A2 = sigmoid(Z2)          # (4,1) — final prediction

    cache = (Z1, A1, Z2, A2)
    return A2, cache

# ============================================================
#  BACKWARD PROPAGATION  (Chain Rule)
# ============================================================
def backward(X, y, cache, W1, W2):
    """
    Chain Rule — peeche se aage:

    dL/dW2 = A1.T @ (dL/dA2 * dA2/dZ2)
    dL/dW1 = X.T  @ (dL/dA1 * dA1/dZ1)

    dL/dA2 = 2 * (A2 - y) / n          (MSE derivative)
    dA2/dZ2 = sigmoid'(Z2)              (activation derivative)
    dZ2/dA1 = W2.T                       (chain to previous layer)
    """
    m = X.shape[0]           # number of samples
    Z1, A1, Z2, A2 = cache

    # ------ Output Layer Gradients ------
    dL_dA2 = 2 * (A2 - y) / m           # dLoss/dA2
    dA2_dZ2 = sigmoid_derivative(Z2)     # dA2/dZ2
    dZ2 = dL_dA2 * dA2_dZ2              # combine (element-wise)

    dW2 = A1.T @ dZ2                     # dLoss/dW2  shape (2,1)
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # dLoss/db2

    # ------ Hidden Layer Gradients (chain rule goes deeper) ------
    dA1 = dZ2 @ W2.T                     # dLoss/dA1  shape (4,2)
    dA1_dZ1 = sigmoid_derivative(Z1)     # dA1/dZ1
    dZ1 = dA1 * dA1_dZ1                  # combine

    dW1 = X.T @ dZ1                      # dLoss/dW1  shape (2,2)
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # dLoss/db1

    grads = {"dW1": dW1, "db1": db1,
             "dW2": dW2, "db2": db2}
    return grads

# ============================================================
#  TRAINING LOOP
# ============================================================
learning_rate = 0.5
epochs = 5000

print("\n" + "=" * 50)
print("  Training Started")
print("=" * 50)

for epoch in range(epochs):

    # 1. Forward pass
    A2, cache = forward(X, W1, b1, W2, b2)

    # 2. Compute loss
    loss = mse_loss(A2, y)

    # 3. Backward pass
    grads = backward(X, y, cache, W1, W2)

    # 4. Update weights (Gradient Descent)
    W1 -= learning_rate * grads["dW1"]
    b1 -= learning_rate * grads["db1"]
    W2 -= learning_rate * grads["dW2"]
    b2 -= learning_rate * grads["db2"]

    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

# ============================================================
#  FINAL RESULTS
# ============================================================
print("\n" + "=" * 50)
print("  Final Predictions")
print("=" * 50)
A2_final, _ = forward(X, W1, b1, W2, b2)
for i in range(len(X)):
    print(f"Input: {X[i]}  |  Predicted: {A2_final[i][0]:.4f}  |  Actual: {y[i][0]}")

print("\n" + "=" * 50)
print("  Learned Weights")
print("=" * 50)
print(f"W1:\n{W1}\nb1: {b1}")
print(f"W2:\n{W2}\nb2: {b2}")

# ============================================================
#  STEP-BY-STEP SINGLE EXAMPLE (for understanding)
# ============================================================
print("\n" + "=" * 50)
print("  Step-by-step: Input [0, 1]")
print("=" * 50)

x_sample = np.array([[0, 1]])

# Forward
Z1_s = x_sample @ W1 + b1
A1_s = sigmoid(Z1_s)
Z2_s = A1_s @ W2 + b2
A2_s = sigmoid(Z2_s)

print(f"Z1 (linear hidden)  : {Z1_s}")
print(f"A1 (hidden output)  : {A1_s}")
print(f"Z2 (linear output)  : {Z2_s}")
print(f"A2 (final pred)     : {A2_s[0][0]:.4f}")
print(f"Actual answer       : 1")
