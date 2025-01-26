import pickle
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Load model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Evaluate model
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
