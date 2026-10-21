# Studi Kasus: Prediksi Harga Rumah Menggunakan Regresi Linier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Pengumpulan Data (Dataset Dummy)
data = {
    'Luas_Bangunan': [36, 45, 60, 80, 100, 120, 150, 200],
    'Harga_Rumah': [350, 450, 600, 800, 1050, 1250, 1600, 2100]
}

df = pd.DataFrame(data)
print("--- Data Mentah ---")
print(df)
print("\n")

# 2. Membagi Data (Features & Target)
X = df[['Luas_Bangunan']]
y = df['Harga_Rumah']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Melatih Model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model berhasil dilatih!\n")

# 4. Evaluasi Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Hasil Evaluasi Model ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared Score (R2): {r2:.2f}\n")

# 5. Simulasi Prediksi Data Baru (Contoh: Rumah 90 m2)
luas_baru = np.array([[90]])
prediksi_harga = model.predict(luas_baru)

print("--- Simulasi Prediksi ---")
print(f"Perkiraan harga untuk rumah seluas 90 m2 adalah: Rp {prediksi_harga[0]:.2f} Juta")
