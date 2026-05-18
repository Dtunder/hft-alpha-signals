import os
import numpy as np
import zipfile
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate Synthetic Data
# We want 20 steps of OBI as input (X) and 1 step of mid-price direction as output (y).
# OBI ranges between -1 and 1.
np.random.seed(42)

NUM_SAMPLES = 10000
SEQ_LENGTH = 20

# Simulate OBI values with some auto-correlation
obi_series = np.zeros(NUM_SAMPLES + SEQ_LENGTH + 1)
obi_series[0] = np.random.uniform(-1, 1)
for i in range(1, len(obi_series)):
    obi_series[i] = obi_series[i-1] * 0.8 + np.random.normal(0, 0.2)
    # clip to [-1, 1]
    obi_series[i] = np.clip(obi_series[i], -1, 1)

# Simulate mid-price changes.
# Let's say mid-price change is positively correlated with recent OBI
price_changes = np.zeros(len(obi_series))
for i in range(SEQ_LENGTH, len(obi_series)):
    # if average OBI in recent steps is positive, price is more likely to go up
    recent_obi = np.mean(obi_series[i-SEQ_LENGTH:i])
    prob_up = 1 / (1 + np.exp(-5 * recent_obi)) # sigmoid
    price_changes[i] = 1 if np.random.rand() < prob_up else 0

X = []
y = []
for i in range(NUM_SAMPLES):
    X.append(obi_series[i:i+SEQ_LENGTH])
    y.append(price_changes[i+SEQ_LENGTH])

X = np.array(X)
y = np.array(y)

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train MLP Model
print("Training MLPClassifier...")
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# 5. Save Model and Metrics
os.makedirs('brain', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Save model
model_path = 'brain/alpha_model.joblib'
joblib.dump(model, model_path)

# Zip model
zip_path = 'brain/alpha_model.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(model_path, os.path.basename(model_path))

# Clean up unzipped model if desired
os.remove(model_path)

# Save report
report_path = 'logs/alpha_report.txt'
with open(report_path, 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"Model saved to {zip_path}")
print(f"Report saved to {report_path}")
