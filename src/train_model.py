import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the diabetes dataset.
    Handle zero values by replacing with median.
    """
    try:
        column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(filepath, names=column_names)
        print("Dataset loaded successfully.")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Features to handle zero values
        zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

        for feature in zero_features:
            median_value = df[df[feature] != 0][feature].median()
            df[feature] = df[feature].replace(0, median_value)
            print(f"Replaced zero values in {feature} with median: {median_value}")

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_models(X_train, X_test, y_train, y_test):
    """
    Train three different models and return their performance metrics.
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    results = {}

    for name, model in models.items():
        try:
            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

        except Exception as e:
            print(f"Error training {name}: {e}")

    return results

def main():
    """
    Main function to train the models.
    """
    # File paths
    data_path = 'data/diabetes.csv'
    models_dir = 'models'

    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Load and preprocess data
    df = load_and_preprocess_data(data_path)
    if df is None:
        return

    # Split features and target, excluding Pregnancies for gender neutrality
    X = df.drop(['Outcome', 'Pregnancies'], axis=1)
    y = df['Outcome']

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Find best model based on F1-score
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']

    print(f"\nBest model: {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")

    # Save best model and scaler
    joblib.dump(best_model, os.path.join(models_dir, 'diabetes_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))

    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    main()
