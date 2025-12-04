import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print("Shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)


class GaussianGenerativeClassifier:
    def __init__(self, lambda_reg=1e-3):
        self.lambda_reg = lambda_reg  # regularization strength
        self.means = None
        self.priors = None
        self.cov = None
        self.inv_cov = None
        self.num_classes = None
        self.num_features = None

    def fit(self, X, y):
        """
        X: (N, D) training data
        y: (N,) training labels
        """
        N, D = X.shape
        self.num_classes = len(np.unique(y))
        self.num_features = D

        # 1. Estimate Class Priors π_k
        self.priors = np.zeros(self.num_classes)
        for k in range(self.num_classes):
            self.priors[k] = np.mean(y == k)

        # 2. Estimate Class Means μ_k
        self.means = np.zeros((self.num_classes, D))
        for k in range(self.num_classes):
            self.means[k] = X[y == k].mean(axis=0)

        # 3. Estimate Shared Covariance Σ
        cov_matrix = np.zeros((D, D))

        for i in range(N):
            k = y[i]
            diff = (X[i] - self.means[k]).reshape(-1, 1)
            cov_matrix += diff @ diff.T

        cov_matrix /= N  # average over all N samples

        # 4. Regularization Σ_λ = Σ + λI
        cov_matrix += self.lambda_reg * np.eye(D)

        self.cov = cov_matrix
        self.inv_cov = np.linalg.inv(cov_matrix)  # precompute inverse

    # Multivariate Gaussian log-density
    def _log_gaussian(self, x, mean):
        D = self.num_features

        diff = (x - mean)
        term1 = -0.5 * diff.T @ self.inv_cov @ diff
        term2 = -0.5 * np.log(np.linalg.det(self.cov))
        term3 = -0.5 * D * np.log(2 * np.pi)

        return term1 + term2 + term3

    # Predict class labels
    def predict(self, X):
        N = X.shape[0]
        scores = np.zeros((N, self.num_classes))

        for i in range(N):
            for k in range(self.num_classes):
                log_prior = np.log(self.priors[k])
                log_likelihood = self._log_gaussian(X[i], self.means[k])
                scores[i, k] = log_prior + log_likelihood

        return np.argmax(scores, axis=1)



lambdas = [1e-4, 1e-3, 1e-2, 1e-1]
val_accuracies = {}

for lam in lambdas:
    model = GaussianGenerativeClassifier(lambda_reg=lam)
    model.fit(X_train_scaled, y_train)

    y_val_pred = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_val_pred)
    val_accuracies[lam] = acc
    print(f"λ={lam}: Validation Accuracy = {acc:.4f}")

best_lambda = max(val_accuracies, key=val_accuracies.get)
print("\nBest λ:", best_lambda)

# Combine training + validation sets
X_train_full = np.vstack([X_train_scaled, X_val_scaled])
y_train_full = np.hstack([y_train, y_val])

# Train final model
final_model = GaussianGenerativeClassifier(lambda_reg=best_lambda)
final_model.fit(X_train_full, y_train_full)


from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)

y_test_pred = final_model.predict(X_test_scaled)


def evaluate_model(y_true, y_pred, title="Confusion Matrix"):
    """
    Prints classification metrics and plots confusion matrix.
    """

    #Metrics
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("=== Evaluation Results ===")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro):    {rec:.4f}")
    print(f"F1-score (macro):  {f1:.4f}")

    #Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    #Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y_true),
        yticklabels=np.unique(y_true)
    )

    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

evaluate_model(y_test, y_test_pred)
