import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("accidents_india.csv")

# Fill missing values appropriately
df["Number_of_Pasengers"].fillna(df["Number_of_Pasengers"].median(), inplace=True)
df["Speed_limit"].fillna(df["Speed_limit"].median(), inplace=True)
df.dropna(inplace=True)  # Drop rows with remaining missing values

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split data into features and target
X = df.drop("Severity", axis=1)  # Assuming "Severity" is the target column
y = df["Severity"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Model
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Train Random Forest Model
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

# Train SVM Model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Train Logistic Regression Model
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)

# Save the best-performing model (Decision Tree for now)
with open("test1.pkl", "wb") as model_file:
    pickle.dump(decision_tree, model_file)

# Print model accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, decision_tree.predict(X_test)))
print("Random Forest Accuracy:", accuracy_score(y_test, random_forest.predict(X_test)))
print("SVM Accuracy:", accuracy_score(y_test, svm_model.predict(X_test)))
print("Logistic Regression Accuracy:", accuracy_score(y_test, logistic_reg.predict(X_test)))
