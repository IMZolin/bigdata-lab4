import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_naive_bayes():
    X_train = pd.read_csv("X_train.csv")
    X_val = pd.read_csv("X_val.csv")
    y_train = pd.read_csv("y_train.csv")["Sentiment"]
    y_val = pd.read_csv("y_val.csv")["Sentiment"]

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_val, y_pred))

    # Save the trained model
    joblib.dump(model, "naive_bayes_model.pkl")
    print("Model saved as naive_bayes_model.pkl")

if __name__ == "__main__":
    train_naive_bayes()
