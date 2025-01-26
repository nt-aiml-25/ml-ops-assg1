import pickle
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import warnings

# Suppress sklearn warnings about version mismatches
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def test_model_accuracy():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Load the model
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    # Make predictions
    y_pred = model.predict(X)
    
    # Assert accuracy is above a threshold
    accuracy = accuracy_score(y, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")
    assert accuracy > 0.9, f"Model accuracy too low: {accuracy}"
