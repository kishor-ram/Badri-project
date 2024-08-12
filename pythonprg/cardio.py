# Step 1: Setting Up the Environment
# Install Necessary Libraries (Uncomment if needed)
# !pip install pandas numpy matplotlib seaborn scikit-learn

# Import the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Step 2: Load and Explore the Dataset
# Load your dataset into a pandas DataFrame with the correct delimiter
df = pd.read_csv('D:/pythonprg/cardio_train.csv', delimiter=';')  # Use your actual file path here

# Explore the Dataset
print(df.head())  # Shows the first few rows of the dataset
print(df.info())  # Gives information about the dataset like column names, non-null counts, and data types
print(df.describe())  # Provides summary statistics of numerical columns

# Step 3: Data Preprocessing
# Handling Missing Values
print(df.isnull().sum())  # Check for missing values
df = df.fillna(df.mean())  # Example: Filling missing values with mean

# Encoding Categorical Variables (if necessary)
df = pd.get_dummies(df, drop_first=True)  # One-hot encoding

# Feature Scaling
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
df = pd.DataFrame(scaled_df, columns=df.columns)

# Step 4: Exploratory Data Analysis (EDA)
# Visualizing Data Distribution
df.hist(figsize=(12, 12))
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

# Pair Plot
sns.pairplot(df)
plt.show()

# Step 5: Splitting the Data
# Separate Features and Target
X = df.drop('cardio', axis=1)  # Features (replace 'target' with the actual target column name)
y = df['cardio']  # Target variable (replace 'target' with the actual
