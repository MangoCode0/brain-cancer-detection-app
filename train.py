import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/Brain_GSE50161.csv")

# Drop unnecessary column
df = df.drop(columns=["samples"])

# Encode target
encoder = LabelEncoder()
df["type"] = encoder.fit_transform(df["type"])

# Split features and target
X = df.drop("type", axis=1)
y = df["type"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_pca, y_train)

# Save model and PCA
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(pca, open("model/pca.pkl", "wb"))
pickle.dump(encoder, open("model/encoder.pkl", "wb"))

print("✅ Model trained and saved successfully!")