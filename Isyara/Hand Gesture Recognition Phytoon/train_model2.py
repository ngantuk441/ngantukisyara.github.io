import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "bisindo_model.pkl")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

df = df.dropna()
df = df.drop_duplicates()

X = df.drop("label", axis=1).values
y = df["label"].values

print(f"Total data: {len(df)}")
print(f"Label tersedia: {sorted(df['label'].unique())}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining model...")
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
svm = SVC(probability=True, kernel='rbf', C=10, random_state=42)
model = VotingClassifier(estimators=[('rf', rf), ('svm', svm)], voting='soft', n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAkurasi: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred))

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("\nModel tersimpan:", MODEL_PATH)