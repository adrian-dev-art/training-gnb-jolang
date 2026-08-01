"""
==========================================================================
TRAINING GAUSSIAN NAIVE BAYES - BAB III SKRIPSI
==========================================================================
Klasifikasi Tingkat Dampak Lingkungan Akibat Perubahan Tutupan Hutan
di Kabupaten Musi Rawas Utara

Dataset : dataset_dampak_lingkungan.csv (351 sampel)
Metode  : Gaussian Naive Bayes (scikit-learn)
Split   : 70% Training, 30% Testing (Stratified)
Target  : Label (Rendah, Sedang, Tinggi)
Features: Suhu (LST), Humid, Hujan, NDVI, Luas Konversi
==========================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
import warnings
import os
import json
import sys

# Fix Windows encoding issues (cp1252 -> utf-8)
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

# ========================================================================
# 1. PENGUMPULAN DATA
# ========================================================================
print("=" * 70)
print("  TRAINING GAUSSIAN NAIVE BAYES - BAB III SKRIPSI")
print("  Klasifikasi Dampak Lingkungan Kabupaten Musi Rawas Utara")
print("=" * 70)

# Tentukan path relatif ke script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "dataset_gabungan.csv")

df = pd.read_csv(csv_path)
print(f"\n[1] PENGUMPULAN DATA")
print(f"    File       : dataset_gabungan.csv")
print(f"    Total data : {len(df)} sampel")
print(f"    Kolom      : {list(df.columns)}")
print(f"    Kecamatan  : {sorted(df['Kecamatan'].unique())}")

# ========================================================================
# 2. EKSPLORASI DATA AWAL
# ========================================================================
print(f"\n{'=' * 70}")
print(f"[2] EKSPLORASI DATA AWAL")
print(f"{'=' * 70}")

# Distribusi label
print("\n    Distribusi Label Dampak Lingkungan:")
label_counts = df['Label'].value_counts()
for label, count in label_counts.items():
    pct = count / len(df) * 100
    print(f"    - {label:8s} : {count:4d} sampel ({pct:.1f}%)")

# Distribusi per kecamatan
print("\n    Distribusi per Kecamatan:")
kec_counts = df['Kecamatan'].value_counts().sort_index()
for kec, count in kec_counts.items():
    print(f"    - {kec:15s} : {count:4d} sampel")

# Statistik deskriptif
features = ['Suhu', 'Humid', 'Hujan', 'NDVI', 'Luas Konversi']
print("\n    Statistik Deskriptif Variabel:")
print(f"    {'Variabel':<18s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'Std':>8s}")
print(f"    {'-' * 50}")
for feat in features:
    print(f"    {feat:<18s} {df[feat].min():>8.2f} {df[feat].max():>8.2f} "
          f"{df[feat].mean():>8.2f} {df[feat].std():>8.2f}")

# Cek missing values
print(f"\n    Missing values : {df.isnull().sum().sum()} (total)")

# ========================================================================
# 3. PREPROCESSING DATA
# ========================================================================
print(f"\n{'=' * 70}")
print(f"[3] PREPROCESSING DATA")
print(f"{'=' * 70}")

# 3a. Data Cleaning
print("\n    [3a] Data Cleaning:")
print(f"         Missing values per kolom:")
for col in df.columns:
    mv = df[col].isnull().sum()
    print(f"         - {col:<18s} : {mv}")
print(f"         Total missing : {df.isnull().sum().sum()}")
print(f"         Status        : {'Data bersih, tidak ada missing values' if df.isnull().sum().sum() == 0 else 'Perlu penanganan'}")

# 3b. Feature Selection
print("\n    [3b] Seleksi Fitur (Feature Selection):")
X = df[features].copy()
y = df['Label'].copy()
print(f"         Variabel Input  (X) : {features}")
print(f"         Variabel Target (y) : Label")
print(f"         Dimensi X           : {X.shape}")

# 3c. Label Encoding
print("\n    [3c] Label Encoding:")
le = LabelEncoder()
# Atur urutan: Rendah=0, Sedang=1, Tinggi=2
le.classes_ = np.array(['Rendah', 'Sedang', 'Tinggi'])
y_encoded = le.transform(y)
for cls_name, cls_id in zip(le.classes_, range(len(le.classes_))):
    print(f"         {cls_name:8s} → {cls_id}")

# 3d. Normalisasi Min-Max
print("\n    [3d] Normalisasi Min-Max (skala [0, 1]):")
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(
    scaler.fit_transform(X),
    columns=features
)
print(f"         Contoh data sebelum normalisasi (5 baris pertama):")
print(X.head().to_string(index=False).replace('\n', '\n         '))
print(f"\n         Contoh data sesudah normalisasi (5 baris pertama):")
print(X_normalized.head().to_string(index=False).replace('\n', '\n         '))

# Simpan parameter normalisasi
print(f"\n         Parameter Normalisasi:")
print(f"         {'Variabel':<18s} {'Min':>10s} {'Max':>10s}")
print(f"         {'-' * 38}")
for i, feat in enumerate(features):
    print(f"         {feat:<18s} {scaler.data_min_[i]:>10.2f} {scaler.data_max_[i]:>10.2f}")

# 3e. Pembagian Dataset
print("\n    [3e] Pembagian Dataset (Stratified Random Sampling):")
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded
)
print(f"         Total data     : {len(X_normalized)}")
print(f"         Data Training  : {len(X_train)} ({len(X_train)/len(X_normalized)*100:.1f}%)")
print(f"         Data Testing   : {len(X_test)} ({len(X_test)/len(X_normalized)*100:.1f}%)")
print(f"         Random State   : 42")
print(f"         Stratified     : Ya (distribusi kelas proporsional)")

# Distribusi kelas di train dan test
print(f"\n         Distribusi kelas di Training set:")
for cls_id, cls_name in enumerate(le.classes_):
    count = np.sum(y_train == cls_id)
    print(f"         - {cls_name:8s} : {count:4d} ({count/len(y_train)*100:.1f}%)")

print(f"\n         Distribusi kelas di Testing set:")
for cls_id, cls_name in enumerate(le.classes_):
    count = np.sum(y_test == cls_id)
    print(f"         - {cls_name:8s} : {count:4d} ({count/len(y_test)*100:.1f}%)")

# ========================================================================
# 4. IMPLEMENTASI GAUSSIAN NAIVE BAYES
# ========================================================================
print(f"\n{'=' * 70}")
print(f"[4] IMPLEMENTASI GAUSSIAN NAIVE BAYES")
print(f"{'=' * 70}")

# Training model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Prior Probability
print("\n    [4a] Prior Probability (Peluang Awal Kelas):")
for cls_id, cls_name in enumerate(le.classes_):
    prior = gnb.class_prior_[cls_id]
    print(f"         P({cls_name:8s}) = {prior:.4f} ({prior*100:.2f}%)")

# Mean dan Std per kelas per fitur
print("\n    [4b] Parameter Model - Mean (μ) per Kelas per Fitur:")
print(f"         {'Fitur':<18s}", end="")
for cls_name in le.classes_:
    print(f" {cls_name:>10s}", end="")
print()
print(f"         {'-' * 48}")
for i, feat in enumerate(features):
    print(f"         {feat:<18s}", end="")
    for cls_id in range(len(le.classes_)):
        print(f" {gnb.theta_[cls_id][i]:>10.6f}", end="")
    print()

print(f"\n    [4c] Parameter Model - Variance (σ²) per Kelas per Fitur:")
print(f"         {'Fitur':<18s}", end="")
for cls_name in le.classes_:
    print(f" {cls_name:>10s}", end="")
print()
print(f"         {'-' * 48}")
for i, feat in enumerate(features):
    print(f"         {feat:<18s}", end="")
    for cls_id in range(len(le.classes_)):
        print(f" {gnb.var_[cls_id][i]:>10.6f}", end="")
    print()

print(f"\n    [4d] Parameter Model - Std Deviasi (σ) per Kelas per Fitur:")
print(f"         {'Fitur':<18s}", end="")
for cls_name in le.classes_:
    print(f" {cls_name:>10s}", end="")
print()
print(f"         {'-' * 48}")
for i, feat in enumerate(features):
    print(f"         {feat:<18s}", end="")
    for cls_id in range(len(le.classes_)):
        print(f" {np.sqrt(gnb.var_[cls_id][i]):>10.6f}", end="")
    print()

# ========================================================================
# 5. PREDIKSI & EVALUASI MODEL
# ========================================================================
print(f"\n{'=' * 70}")
print(f"[5] PREDIKSI & EVALUASI MODEL")
print(f"{'=' * 70}")

y_pred = gnb.predict(X_test)

# Confusion Matrix
print("\n    [5a] Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"\n         {'':18s}", end="")
for cls_name in le.classes_:
    print(f" Pred-{cls_name:>6s}", end="")
print()
print(f"         {'-' * 54}")
for i, cls_name in enumerate(le.classes_):
    print(f"         Actual-{cls_name:<10s}", end="")
    for j in range(len(le.classes_)):
        print(f" {cm[i][j]:>12d}", end="")
    print()

# Metrik evaluasi
accuracy = accuracy_score(y_test, y_pred) * 100
precision_macro = precision_score(y_test, y_pred, average='macro') * 100
recall_macro = recall_score(y_test, y_pred, average='macro') * 100
f1_macro = f1_score(y_test, y_pred, average='macro') * 100

print(f"\n    [5b] Metrik Evaluasi:")
print(f"         Accuracy                    : {accuracy:.2f}%")
print(f"         Macro Average Precision     : {precision_macro:.2f}%")
print(f"         Macro Average Recall        : {recall_macro:.2f}%")
print(f"         Macro Average F1-Score      : {f1_macro:.2f}%")

# Classification Report detail
print(f"\n    [5c] Classification Report Detail:")
report = classification_report(y_test, y_pred, target_names=le.classes_, digits=4)
for line in report.split('\n'):
    print(f"         {line}")

# Per-class metrics
print(f"\n    [5d] Metrik Per Kelas:")
precision_per = precision_score(y_test, y_pred, average=None)
recall_per = recall_score(y_test, y_pred, average=None)
f1_per = f1_score(y_test, y_pred, average=None)
print(f"         {'Kelas':<10s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>8s}")
print(f"         {'-' * 48}")
for i, cls_name in enumerate(le.classes_):
    support = np.sum(y_test == i)
    print(f"         {cls_name:<10s} {precision_per[i]*100:>9.2f}% {recall_per[i]*100:>9.2f}% "
          f"{f1_per[i]*100:>9.2f}% {support:>8d}")

# ========================================================================
# 6. CONTOH PERHITUNGAN MANUAL (untuk BAB III)
# ========================================================================
print(f"\n{'=' * 70}")
print(f"[6] CONTOH PERHITUNGAN MANUAL (untuk narasi BAB III)")
print(f"{'=' * 70}")

# Ambil satu sampel dari data test
sample_idx = 0
sample = X_test.iloc[sample_idx]
actual_label = le.classes_[y_test[sample_idx]]

print(f"\n    Sampel yang digunakan untuk contoh perhitungan:")
print(f"    (Data Testing index ke-{sample_idx})")
for feat in features:
    print(f"    - {feat:<18s} : {sample[feat]:.4f} (ternormalisasi)")

# Hitung manual probability
print(f"\n    Actual Label : {actual_label}")
print(f"\n    Perhitungan Likelihood P(x|C) untuk setiap kelas:")

for cls_id, cls_name in enumerate(le.classes_):
    print(f"\n    --- Kelas: {cls_name} ---")
    log_likelihood = 0
    for i, feat in enumerate(features):
        mu = gnb.theta_[cls_id][i]
        var = gnb.var_[cls_id][i]
        sigma = np.sqrt(var)
        x_val = sample[feat]
        
        # Gaussian PDF
        likelihood = (1 / (sigma * np.sqrt(2 * np.pi))) * \
                     np.exp(-((x_val - mu) ** 2) / (2 * var))
        log_likelihood += np.log(likelihood)
        
        print(f"    P({feat}={x_val:.4f} | {cls_name})")
        print(f"      μ = {mu:.6f}, σ = {sigma:.6f}")
        print(f"      = (1 / ({sigma:.6f} × √(2π))) × exp(-({x_val:.4f} - {mu:.6f})² / (2 × {var:.6f}))")
        print(f"      = {likelihood:.6f}")
    
    prior = gnb.class_prior_[cls_id]
    print(f"\n    Prior P({cls_name}) = {prior:.4f}")
    print(f"    Log-Likelihood Total = {log_likelihood:.6f}")

# Prediksi
pred_proba = gnb.predict_proba(X_test.iloc[[sample_idx]])
print(f"\n    Probabilitas Posterior (dari model):")
for cls_id, cls_name in enumerate(le.classes_):
    print(f"    P({cls_name:8s} | data) = {pred_proba[0][cls_id]:.6f} ({pred_proba[0][cls_id]*100:.2f}%)")

predicted = le.classes_[gnb.predict(X_test.iloc[[sample_idx]])[0]]
print(f"\n    ⇒ Prediksi: {predicted} (kelas dengan probabilitas tertinggi)")
print(f"    ⇒ Aktual  : {actual_label}")
print(f"    ⇒ Status  : {'✓ BENAR' if predicted == actual_label else '✗ SALAH'}")

# ========================================================================
# 7. KLASIFIKASI PER KECAMATAN (untuk TOPSIS)
# ========================================================================
print(f"\n{'=' * 70}")
print(f"[7] KLASIFIKASI DAMPAK PER KECAMATAN (Input untuk TOPSIS)")
print(f"{'=' * 70}")

# Prediksi seluruh dataset
X_all_normalized = X_normalized.copy()
y_pred_all = gnb.predict(X_all_normalized)
y_pred_labels = le.inverse_transform(y_pred_all)

df['Prediksi'] = y_pred_labels
df['Label_Encoded'] = le.transform(df['Label'])
df['Prediksi_Encoded'] = y_pred_all

# Mapping dampak ke skala ordinal
dampak_map = {'Rendah': 1, 'Sedang': 2, 'Tinggi': 3}

print(f"\n    Rata-rata Skor Dampak per Kecamatan (Skala 1-3):")
print(f"    {'Kecamatan':<15s} {'Skor Dampak':>12s} {'Kelas':>8s} {'N':>5s}")
print(f"    {'-' * 44}")

kec_scores = {}
for kec in sorted(df['Kecamatan'].unique()):
    kec_data = df[df['Kecamatan'] == kec]
    avg_score = kec_data['Prediksi'].map(dampak_map).mean()
    n = len(kec_data)
    if avg_score <= 1.5:
        kelas = "Rendah"
    elif avg_score <= 2.5:
        kelas = "Sedang"
    else:
        kelas = "Tinggi"
    print(f"    {kec:<15s} {avg_score:>12.3f} {kelas:>8s} {n:>5d}")
    kec_scores[kec] = avg_score

# Detail distribusi prediksi per kecamatan
print(f"\n    Detail Distribusi Prediksi per Kecamatan:")
for kec in sorted(df['Kecamatan'].unique()):
    kec_data = df[df['Kecamatan'] == kec]
    print(f"\n    {kec} (n={len(kec_data)}):")
    for label in ['Rendah', 'Sedang', 'Tinggi']:
        count = len(kec_data[kec_data['Prediksi'] == label])
        pct = count / len(kec_data) * 100
        bar = '█' * int(pct / 2)
        print(f"      {label:8s}: {count:3d} ({pct:5.1f}%) {bar}")

# ========================================================================
# 8. AKURASI KESELURUHAN PADA SELURUH DATASET
# ========================================================================
print(f"\n{'=' * 70}")
print(f"[8] AKURASI PREDIKSI PADA SELURUH DATASET")
print(f"{'=' * 70}")

acc_all = accuracy_score(le.transform(df['Label']), y_pred_all) * 100
print(f"\n    Akurasi keseluruhan (350 sampel) : {acc_all:.2f}%")

cm_all = confusion_matrix(le.transform(df['Label']), y_pred_all)
print(f"\n    Confusion Matrix (seluruh data):")
print(f"    {'':18s}", end="")
for cls_name in le.classes_:
    print(f" Pred-{cls_name:>6s}", end="")
print()
print(f"    {'-' * 54}")
for i, cls_name in enumerate(le.classes_):
    print(f"    Actual-{cls_name:<10s}", end="")
    for j in range(len(le.classes_)):
        print(f" {cm_all[i][j]:>12d}", end="")
    print()

# ========================================================================
# 9. SIMPAN HASIL
# ========================================================================
print(f"\n{'=' * 70}")
print(f"[9] MENYIMPAN HASIL")
print(f"{'=' * 70}")

# Simpan hasil prediksi ke CSV
output_path = os.path.join(script_dir, "hasil_klasifikasi_gnb.csv")
df_output = df[['ID', 'Suhu', 'Humid', 'Hujan', 'NDVI', 'Luas Konversi', 
                 'Kecamatan', 'Label', 'Prediksi']].copy()
df_output.to_csv(output_path, index=False)
print(f"    Hasil prediksi disimpan ke : hasil_klasifikasi_gnb.csv")

# Simpan skor dampak per kecamatan
kec_output_path = os.path.join(script_dir, "skor_dampak_kecamatan.csv")
kec_df = pd.DataFrame([
    {'Kecamatan': kec, 'Skor_Dampak': score, 'N_Sampel': len(df[df['Kecamatan'] == kec])}
    for kec, score in kec_scores.items()
])
kec_df = kec_df.sort_values('Skor_Dampak', ascending=False)
kec_df.to_csv(kec_output_path, index=False)
print(f"    Skor dampak per kecamatan  : skor_dampak_kecamatan.csv")

# Simpan model parameters
model_params = {
    'class_names': le.classes_.tolist(),
    'class_prior': gnb.class_prior_.tolist(),
    'theta_mean': gnb.theta_.tolist(),
    'variance': gnb.var_.tolist(),
    'features': features,
    'scaler_min': scaler.data_min_.tolist(),
    'scaler_max': scaler.data_max_.tolist(),
    'metrics': {
        'accuracy': round(accuracy, 2),
        'precision_macro': round(precision_macro, 2),
        'recall_macro': round(recall_macro, 2),
        'f1_macro': round(f1_macro, 2),
        'confusion_matrix': cm.tolist()
    },
    'split': {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'random_state': 42,
        'stratified': True
    }
}

params_path = os.path.join(script_dir, "model_gnb_params.json")
with open(params_path, 'w') as f:
    json.dump(model_params, f, indent=2)
print(f"    Parameter model GNB        : model_gnb_params.json")

# ========================================================================
# 10. RINGKASAN UNTUK BAB III
# ========================================================================
print(f"\n{'=' * 70}")
print(f"[10] RINGKASAN UNTUK BAB III SKRIPSI")
print(f"{'=' * 70}")

print(f"""
    ┌─────────────────────────────────────────────────────────┐
    │              HASIL TRAINING GAUSSIAN NAIVE BAYES        │
    ├─────────────────────────────────────────────────────────┤
    │  Dataset       : 351 sampel, 7 kecamatan               │
    │  Features      : LST, Humid, Hujan, NDVI, Luas Konversi│
    │  Normalisasi   : Min-Max [0, 1]                         │
    │  Split         : 70% Train / 30% Test (Stratified)      │
    │  Training      : {len(X_train)} sampel                             │
    │  Testing       : {len(X_test)} sampel                             │
    ├─────────────────────────────────────────────────────────┤
    │  HASIL EVALUASI (Data Testing)                          │
    │  Accuracy      : {accuracy:.2f}%                              │
    │  Precision     : {precision_macro:.2f}%  (Macro Average)            │
    │  Recall        : {recall_macro:.2f}%  (Macro Average)            │
    │  F1-Score      : {f1_macro:.2f}%  (Macro Average)            │
    └─────────────────────────────────────────────────────────┘
""")

print("=" * 70)
print("  SELESAI! Semua hasil telah disimpan.")
print("  File output:")
print("  1. hasil_klasifikasi_gnb.csv   - Prediksi per sampel")
print("  2. skor_dampak_kecamatan.csv   - Skor C1 per kecamatan (untuk TOPSIS)")
print("  3. model_gnb_params.json       - Parameter model untuk dokumentasi")
print("=" * 70)
