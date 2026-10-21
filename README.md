# Studi Kasus Machine Learning: Prediksi Harga Rumah

Repositori ini berisi implementasi sederhana studi kasus *Machine Learning* menggunakan pendekatan **Supervised Learning** (*Regression*) berdasarkan materi kuliah umum Machine Learning[cite: 3].

## 📋 Deskripsi Studi Kasus
Studi kasus ini mensimulasikan bagaimana sebuah model *Machine Learning* dapat mempelajari pola dari data historis untuk memprediksi nilai kontinu (harga rumah) berdasarkan fitur bangunan (luas dalam meter persegi)[cite: 3].

Alur kerja (*workflow*) yang diterapkan mengikuti standar pengembangan ML:
1. **Pengumpulan Data (*Data Collection*):** Menggunakan dataset dummy berisi ukuran luas bangunan dan harga rumah[cite: 2].
2. **Pemisahan Data (*Split Data*):** Membagi data menjadi data latih (*training*) 75% dan data uji (*testing*) 25%[cite: 2].
3. **Pelatihan Model (*Train Model*):** Melatih model menggunakan algoritma *Linear Regression*[cite: 2, 3].
4. **Evaluasi Model (*Evaluate Model*):** Mengukur performa model menggunakan metrik *Mean Absolute Error* (MAE) dan *R-squared Score* (R2)[cite: 2, 3].
5. **Inferensi (*Deploy for Inference*):** Melakukan prediksi harga untuk data baru di dunia nyata[cite: 2, 3].

---

## 🛠️ Prasyarat & Library
Pastikan Anda telah menginstal pustaka (*library*) Python berikut sebelum menjalankan program:
```bash
pip install pandas numpy scikit-learn
