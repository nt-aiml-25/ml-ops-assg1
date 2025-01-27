from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import optuna
import joblib

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X, y = newsgroups.data, newsgroups.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("Data preprocessing complete. TF-IDF transformation applied.")
                                   
# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    #c = trial.suggest_loguniform('C', 1e-4, 10.0)  # Regularization strength
    c = trial.suggest_float('C', 1e-4, 10.0,log=True)  # Regularization strength
    penalty = trial.suggest_categorical('penalty', ['l2'])  # Logistic regression only supports l2 for sparse inputs
    solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])  # Solvers for Logistic Regression

    # Initialize the model with the trial's hyperparameters
    model = LogisticRegression(C=c, penalty=penalty, solver=solver, random_state=42, max_iter=500)

    # Fit the model on the training data
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model on the validation set
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    # Print progress during the tuning process
    print(f"Trial {trial.number}: C={c}, penalty={penalty}, solver={solver}, accuracy={accuracy}")

    return 1 - accuracy  # Optuna minimizes the objective, so we use 1 - accuracy

# Create a study and optimize the hyperparameters
print("Starting hyperparameter tuning with Optuna...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Get the best hyperparameters and their performance
best_params = study.best_params
best_score = 1 - study.best_value

print(f"\nBest Hyperparameters: {best_params}")
print(f"Best Accuracy: {best_score}")

# Train the final model using the best hyperparameters
final_model = LogisticRegression(
    C=best_params['C'],
    penalty=best_params['penalty'],
    solver=best_params['solver'],
    random_state=42,
    max_iter=500
)
final_model.fit(X_train_tfidf, y_train)

# Save the trained model
joblib.dump(final_model, 'newsgroups_text_classifier.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
print("Model and vectorizer saved successfully.")