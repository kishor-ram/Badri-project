import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset and handle bad lines
df = pd.read_csv('D:/pythonprg/spotifydataset.csv', delimiter=';', on_bad_lines='skip')

# Display basic info and check for non-numeric columns
print(df.info())

# Convert non-numeric columns to numeric using Label Encoding or drop them if not needed
non_numeric_cols = df.select_dtypes(include=['object']).columns
print(f"Non-numeric columns: {non_numeric_cols}")

# Option 1: Encode non-numeric columns
label_encoders = {}
for col in non_numeric_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col].astype(str))

# Option 2: Drop non-numeric columns if they are not relevant
# df = df.drop(columns=non_numeric_cols)

# Ensure the dataframe is now numeric
print(df.head())
print(df.info())

# Data preprocessing (if needed, e.g., handling missing values)
df = df.fillna(df.mean())

# Data Analysis & Visualization
# Heatmap for correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Spotify Dataset')
plt.show()

# Clustering using a simple KMeans model as an example (optional step)
from sklearn.cluster import KMeans

# Assuming the number of clusters you want is 5 (can be adjusted)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df)

# Visualizing clusters (e.g., using a pairplot for selected features)
sns.pairplot(df, hue='Cluster', palette='tab10')
plt.show()

# Final Model Building (based on clusters or other analysis)
# Example: Simple model to predict a target variable (optional step)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Assuming 'target' is the name of the column you want to predict (adjust as needed)
# X = df.drop(columns=['target'])  # Features
# y = df['target']                 # Target variable
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print(classification_report(y_test, y_pred))

print("Project completed successfully.")
