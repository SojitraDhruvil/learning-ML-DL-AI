import numpy as np

# ================================================================
#  Neural Network — Total Derivative har jagah dikhaya gaya hai
#  Architecture: Input(2) -> Hidden(3) -> Output(1)
#
#  Total Derivative formula:
#  dL/dW = (∂L/∂A2)·(∂A2/∂Z2)·(∂Z2/∂W2)   <- chain rule = total deriv
# ================================================================

np.random.seed(42)

# ----------------------------------------------------------------
#  ACTIVATION FUNCTIONS
# ----------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    # ∂sigmoid/∂z = sigmoid(z) × (1 - sigmoid(z))
    s = sigmoid(z)
    return s * (1 - s)

def mse_loss(y_pred, y_true):
    # L = (1/m) × Σ(ŷ - y)²
    m = y_true.shape[0]
    return np.sum((y_pred - y_true) ** 2) / m

# ----------------------------------------------------------------
#  WEIGHTS INITIALIZE  (Input=2, Hidden=3, Output=1)
# ----------------------------------------------------------------
W1 = np.random.randn(2, 3) * 0.5   # shape (2, 3)
b1 = np.zeros((1, 3))               # shape (1, 3)
W2 = np.random.randn(3, 1) * 0.5   # shape (3, 1)
b2 = np.zeros((1, 1))               # shape (1, 1)

# ----------------------------------------------------------------
#  TRAINING DATA  (XOR)
# ----------------------------------------------------------------
X = np.array([[0,0],[0,1],[1,0],[1,1]])  # (4,2)
y = np.array([[0],[1],[1],[0]])           # (4,1)
m = X.shape[0]                           # 4 samples

# ================================================================
#  FORWARD PROPAGATION
#  Sirf values calculate — koi gradient nahi
# ================================================================
def forward(X, W1, b1, W2, b2):
    # Layer 1
    Z1 = X @ W1 + b1        # (4,2)@(2,3) = (4,3)  | Z1 = xW1 + b1
    A1 = sigmoid(Z1)         # (4,3)                 | A1 = σ(Z1)

    # Layer 2
    Z2 = A1 @ W2 + b2       # (4,3)@(3,1) = (4,1)  | Z2 = A1·W2 + b2
    A2 = sigmoid(Z2)         # (4,1)                 | A2 = σ(Z2) = ŷ

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# ================================================================
#  BACKWARD PROPAGATION  — TOTAL DERIVATIVE yahan hai!
#
#  Total Derivative of Loss w.r.t. each weight:
#
#  dL/dW2 = dL/dA2 · dA2/dZ2 · dZ2/dW2     (3 parts multiply)
#  dL/dW1 = dL/dA2 · dA2/dZ2 · dZ2/dA1 · dA1/dZ1 · dZ1/dW1
#           (chain se peeche jaate hain — yahi total derivative hai)
# ================================================================
def backward(X, y, cache, W1, W2):
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    A2 = cache["A2"]

    # ── OUTPUT LAYER ──────────────────────────────────────────
    # Part 1: ∂L/∂A2  (Loss ka A2 ke saath partial derivative)
    #         L = (1/m)Σ(A2-y)²  →  ∂L/∂A2 = 2(A2-y)/m
    dL_dA2 = 2 * (A2 - y) / m              # shape (4,1)

    # Part 2: ∂A2/∂Z2  (sigmoid ka derivative)
    #         A2 = sigmoid(Z2)  →  ∂A2/∂Z2 = A2(1-A2)
    dA2_dZ2 = sigmoid_grad(Z2)             # shape (4,1)

    # Part 3: ∂Z2/∂W2  (linear ka derivative)
    #         Z2 = A1·W2 + b2  →  ∂Z2/∂W2 = A1
    # dZ2_dW2 = A1  (use karte hain neeche)

    # TOTAL DERIVATIVE for W2:
    # dL/dW2 = (∂L/∂A2) · (∂A2/∂Z2) · (∂Z2/∂W2)
    #        = dL_dA2 * dA2_dZ2   [element-wise]  → dZ2
    #        then A1.T @ dZ2      [matrix multiply for W2]
    dZ2    = dL_dA2 * dA2_dZ2              # (4,1) — combined signal
    dL_dW2 = A1.T @ dZ2                    # (3,1) ← TOTAL DERIVATIVE w.r.t W2
    dL_db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1,1)

    # ── HIDDEN LAYER ──────────────────────────────────────────
    # Chain continues peeche...

    # Part 4: ∂Z2/∂A1  (Z2 ka A1 ke saath partial derivative)
    #         Z2 = A1·W2 + b2  →  ∂Z2/∂A1 = W2
    dZ2_dA1 = W2                            # shape (3,1)

    # Part 5: ∂A1/∂Z1  (sigmoid ka derivative, hidden layer)
    dA1_dZ1 = sigmoid_grad(Z1)             # shape (4,3)

    # Part 6: ∂Z1/∂W1
    #         Z1 = X·W1 + b1  →  ∂Z1/∂W1 = X
    # dZ1_dW1 = X  (use karte hain neeche)

    # TOTAL DERIVATIVE for W1:
    # dL/dW1 = dL/dA2 · dA2/dZ2 · dZ2/dA1 · dA1/dZ1 · dZ1/dW1
    #        = dZ2 @ W2.T      → propagate gradient to A1
    #        * sigmoid_grad    → through activation
    #        then X.T @ dZ1   → w.r.t W1
    dA1    = dZ2 @ dZ2_dA1.T               # (4,3) — gradient A1 tak aaya
    dZ1    = dA1 * dA1_dZ1                 # (4,3) — through sigmoid
    dL_dW1 = X.T @ dZ1                     # (2,3) ← TOTAL DERIVATIVE w.r.t W1
    dL_db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1,3)

    grads = {
        "dL_dW1": dL_dW1, "dL_db1": dL_db1,
        "dL_dW2": dL_dW2, "dL_db2": dL_db2
    }
    return grads

# ================================================================
#  TRAINING LOOP
# ================================================================
lr     = 0.5
epochs = 8000

print("=" * 60)
print("  Total Derivative in Action — Neural Network Training")
print("=" * 60)
print(f"{'Epoch':>8}  {'Loss':>10}  {'dL/dW1 norm':>14}  {'dL/dW2 norm':>14}")
print("-" * 60)

for epoch in range(epochs):
    # 1. Forward — values calculate
    A2, cache = forward(X, W1, b1, W2, b2)

    # 2. Loss
    loss = mse_loss(A2, y)

    # 3. Backward — TOTAL DERIVATIVES calculate
    grads = backward(X, y, cache, W1, W2)

    # 4. Gradient Descent — weights update
    W1 -= lr * grads["dL_dW1"]
    b1 -= lr * grads["dL_db1"]
    W2 -= lr * grads["dL_dW2"]
    b2 -= lr * grads["dL_db2"]

    if epoch % 2000 == 0:
        g1_norm = np.linalg.norm(grads["dL_dW1"])
        g2_norm = np.linalg.norm(grads["dL_dW2"])
        print(f"{epoch:>8}  {loss:>10.6f}  {g1_norm:>14.6f}  {g2_norm:>14.6f}")

print("-" * 60)

# ================================================================
#  TOTAL DERIVATIVE — STEP BY STEP SINGLE SAMPLE
#  Ek example (0,1) ke liye poora calculation dikhata hai
# ================================================================
print("\n" + "=" * 60)
print("  Step-by-step Total Derivative: input [0, 1]")
print("=" * 60)

x_s = np.array([[0, 1]])
y_s = np.array([[1]])

# Forward
Z1_s = x_s @ W1 + b1
A1_s = sigmoid(Z1_s)
Z2_s = A1_s @ W2 + b2
A2_s = sigmoid(Z2_s)
L_s  = mse_loss(A2_s, y_s)

print(f"\nForward pass:")
print(f"  Z1         = {Z1_s.round(4)}")
print(f"  A1         = {A1_s.round(4)}")
print(f"  Z2         = {Z2_s.round(4)}")
print(f"  A2 (pred)  = {A2_s[0,0]:.4f}")
print(f"  Loss L     = {L_s:.6f}")

# Backward — total derivative har step pe
dL_dA2_s  = 2 * (A2_s - y_s) / 1
dA2_dZ2_s = sigmoid_grad(Z2_s)
dZ2_s     = dL_dA2_s * dA2_dZ2_s
dL_dW2_s  = A1_s.T @ dZ2_s

dA1_s     = dZ2_s @ W2.T
dA1_dZ1_s = sigmoid_grad(Z1_s)
dZ1_s     = dA1_s * dA1_dZ1_s
dL_dW1_s  = x_s.T @ dZ1_s

print(f"\nBackward pass (Total Derivatives):")
print(f"  ∂L/∂A2    = {dL_dA2_s.round(4)}   ← 2(A2-y)/m")
print(f"  ∂A2/∂Z2   = {dA2_dZ2_s.round(4)}  ← sigmoid'(Z2)")
print(f"  dZ2       = {dZ2_s.round(4)}   ← element-wise product")
print(f"  dL/dW2    =\n{dL_dW2_s.round(4)}    ← TOTAL DERIV w.r.t W2")
print(f"  dL/dW1    =\n{dL_dW1_s.round(4)}  ← TOTAL DERIV w.r.t W1")

# ================================================================
#  FINAL RESULTS
# ================================================================
print("\n" + "=" * 60)
print("  Final Predictions after Training")
print("=" * 60)
A2_final, _ = forward(X, W1, b1, W2, b2)
for i in range(len(X)):
    bar = "✓" if abs(A2_final[i,0] - y[i,0]) < 0.1 else "✗"
    print(f"  {bar} Input {X[i]} | Pred: {A2_final[i,0]:.4f} | Actual: {y[i,0]}")

final_loss = mse_loss(A2_final, y)
print(f"\n  Final Loss: {final_loss:.6f}")
print("\n  Total Derivative = Chain Rule se saare partial derivatives")
print("  ka product — yahi backpropagation ka core hai!")
