import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
import os

def load_models_and_data():
    """
    Load trained models, scaler, and test data.
    """
    try:
        # Load models and scaler
        model = joblib.load('models/diabetes_model.pkl')
        scaler = joblib.load('models/scaler.pkl')

        # Load original data for evaluation
        column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv('data/diabetes.csv', names=column_names)

        # Preprocess data (same as in training)
        zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for feature in zero_features:
            median_value = df[df[feature] != 0][feature].median()
            df[feature] = df[feature].replace(0, median_value)

        X = df.drop('Outcome', axis=1)
        y = df['Outcome']

        # Scale features
        X_scaled = scaler.transform(X)

        return model, scaler, X_scaled, y
    except Exception as e:
        print(f"Error loading models/data: {e}")
        return None, None, None, None

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot confusion matrix for the model.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'visualizations/confusion_matrix_{model_name.lower().replace(" ", "_")}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(models_dict, X_test, y_test):
    """
    Plot ROC curves for all models.
    """
    plt.figure(figsize=(10, 8))

    for name, model in models_dict.items():
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        except Exception as e:
            print(f"Error plotting ROC for {name}: {e}")

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc="lower right")
    plt.savefig('visualizations/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for Random Forest model.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances (Random Forest)')
        plt.bar(range(len(importances)), importances[indices],
                align='center', color='skyblue')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Model does not support feature importance.")

def compare_models(models_dict, X_test, y_test):
    """
    Compare all models side by side.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    results = []

    for name, model in models_dict.items():
        try:
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })

            # Plot confusion matrix for each model
            plot_confusion_matrix(y_test, y_pred, name)

        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    # Create comparison table
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    # Save results to CSV
    results_df.to_csv('visualizations/model_comparison.csv', index=False)

    return results_df

def load_all_models():
    """
    Load all trained models for comparison.
    """
    models = {}
    model_files = ['logistic_regression.pkl', 'random_forest.pkl', 'svm.pkl']

    for model_file in model_files:
        try:
            model_path = os.path.join('models', model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
                models[model_name] = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading {model_file}: {e}")

    return models

def main():
    """
    Main function for model evaluation.
    """
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)

    # Load best model and data
    model, scaler, X, y = load_models_and_data()
    if model is None:
        print("Could not load model and data. Please ensure training is completed first.")
        return

    # Load all models for comparison
    models_dict = load_all_models()

    # If no individual models, use the best model
    if not models_dict:
        models_dict = {'Best Model': model}

    # Split data for evaluation (using same split as training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compare models
    results_df = compare_models(models_dict, X_test, y_test)

    # Plot ROC curves
    plot_roc_curves(models_dict, X_test, y_test)

    # Plot feature importance (if Random Forest is available)
    if 'Random Forest' in models_dict:
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        plot_feature_importance(models_dict['Random Forest'], feature_names)

    print("\nModel evaluation completed!")
    print("Results saved to 'visualizations/' folder.")

if __name__ == "__main__":
    main()
