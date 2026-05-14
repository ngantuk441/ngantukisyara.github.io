"""
train_model.py
Jalankan setelah collect data dengan collect_bisindo.py (format baru)

Dataset format baru (130 kolom):
  label, hand_mode, hand1_side,
  x0_h1..z20_h1 (63 kolom),
  hand2_side,
  x0_h2..z20_h2 (63 kolom)

Model disimpan per kategori:
  models/model_alfabet.pkl
  models/model_angka.pkl
  models/model_kata.pkl
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Definisi kategori label ───────────────────────────────────
# Sesuaikan jika kamu punya label kata yang berbeda
KATEGORI = {
    "alfabet": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "angka":   [str(i) for i in range(1, 11)],   # "1" sampai "10"
    "kata":    ["HALO", "HI", "ILY", "PEACE", "SIP", "OK"],  # tambah sesuai kebutuhan
}

# ── Load dataset ──────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.dropna().drop_duplicates()
df['label'] = df['label'].astype(str)
print(f"Total data: {len(df)} baris")
print(f"Label tersedia: {sorted(df['label'].unique(), key=str)}")
print(f"Kolom: {list(df.columns[:5])} ... ({len(df.columns)} total)\n")

# ── Kolom fitur (semua kecuali label) ────────────────────────
# hand_mode, hand1_side, x0_h1..z20_h1, hand2_side, x0_h2..z20_h2
FEATURE_COLS = [c for c in df.columns if c != "label"]

# ── Train per kategori ────────────────────────────────────────
for kategori, label_list in KATEGORI.items():
    # Filter hanya baris dengan label yang masuk kategori ini
    df_kat = df[df["label"].isin(label_list)].copy()

    if len(df_kat) == 0:
        print(f"[SKIP] Kategori '{kategori}' — belum ada data\n")
        continue

    label_ada = sorted(df_kat["label"].unique())
    print(f"{'='*50}")
    print(f"Kategori : {kategori.upper()}")
    print(f"Label    : {label_ada}")
    print(f"Jumlah   : {len(df_kat)} baris")

    X = df_kat[FEATURE_COLS].values
    y = df_kat["label"].values

    # Minimal 2 label untuk split stratify
    if len(label_ada) < 2:
        print(f"[SKIP] Butuh minimal 2 label berbeda untuk training\n")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training {len(X_train)} | Testing {len(X_test)}")

    # Model: VotingClassifier (RF + SVM) sama seperti sebelumnya
    rf  = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    svm = SVC(probability=True, kernel="rbf", C=10, random_state=42)
    model = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm)],
        voting="soft",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi  : {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Simpan model
    model_path = os.path.join(MODEL_DIR, f"model_{kategori}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Tersimpan: {model_path}\n")

print("Selesai! Semua model tersimpan di folder 'models/'")