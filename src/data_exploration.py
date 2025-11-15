import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(filepath):
    """
    Load the diabetes dataset.
    """
    try:
        column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(filepath, names=column_names)
        print("Dataset loaded successfully.")
        print(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def basic_statistics(df):
    """
    Display basic statistics of the dataset.
    """
    print("\nBasic Statistics:")
    print(df.describe())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nOutcome Distribution:")
    print(df['Outcome'].value_counts())
    print(df['Outcome'].value_counts(normalize=True))

def create_visualizations(df):
    """
    Create and save visualizations.
    """
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # 1. Outcome distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Outcome', data=df, palette='Set2')
    plt.title('Distribution of Diabetes Outcome')
    plt.xlabel('Outcome (0=Non-Diabetic, 1=Diabetic)')
    plt.ylabel('Count')
    plt.savefig('visualizations/outcome_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Features')
    plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Feature distributions
    features = df.columns[:-1]  # Exclude Outcome
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.ravel()

    for i, feature in enumerate(features):
        sns.histplot(df[feature], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('visualizations/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Box plots for features by outcome
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.ravel()

    for i, feature in enumerate(features):
        sns.boxplot(x='Outcome', y=feature, data=df, ax=axes[i], palette='Set2')
        axes[i].set_title(f'{feature} by Outcome')
        axes[i].set_xlabel('Outcome')
        axes[i].set_ylabel(feature)

    plt.tight_layout()
    plt.savefig('visualizations/boxplots_by_outcome.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Visualizations saved to 'visualizations/' folder.")

def handle_zero_values(df):
    """
    Analyze and handle zero values in the dataset.
    """
    features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    print("\nZero Values Analysis:")
    for feature in features:
        zero_count = (df[feature] == 0).sum()
        total_count = len(df)
        percentage = (zero_count / total_count) * 100
        print(f"{feature}: {zero_count} zeros ({percentage:.2f}%)")

    # Replace zeros with median (for features where zero is invalid)
    df_clean = df.copy()
    for feature in features:
        median_value = df[df[feature] != 0][feature].median()
        df_clean[feature] = df[feature].replace(0, median_value)
        print(f"Replaced zeros in {feature} with median: {median_value:.2f}")

    return df_clean

def main():
    """
    Main function for data exploration.
    """
    data_path = 'data/diabetes.csv'

    # Load data
    df = load_data(data_path)
    if df is None:
        return

    # Basic statistics
    basic_statistics(df)

    # Handle zero values
    df_clean = handle_zero_values(df)

    # Create visualizations
    create_visualizations(df_clean)

    print("\nData exploration completed!")

if __name__ == "__main__":
    main()
