# Feature Specifications

## Tahap 1: Data Pipeline (Branch: feature/data-pipeline)
* Input: `data/raw/train.csv`, `data/raw/oil.csv`, `data/raw/holidays_events.csv`, dan `data/raw/stores.csv`.
* Proses: 
  1. Filter data agar relevan (bisa di-downsample agar komputasi ringan jika diperlukan).
  2. Gabungkan data minyak (oil) berdasarkan tanggal transaksi.
  3. Gabungkan data libur nasional (holidays_events) berdasarkan tanggal.
  4. Gabungkan data metadata toko (stores) berdasarkan store_nbr.
  5. Lakukan encoding pada variabel kategorikal (seperti tipe toko, kota, kategori produk).
* Output: `data/processed/training_data.csv`.

## Tahap 2: Model Training (Branch: feature/model-training)
* Input: `data/processed/training_data.csv`.
* Proses: Lakukan train-test split secara berurutan berdasarkan waktu (time-series split, jangan diacak/random). Latih model Linear Regression sebagai baseline, lalu Random Forest sebagai perbandingan. Hitung MAE dan RMSE untuk evaluasi.
* Output: Simpan model terbaik dalam format `app/best_model.pkl`.

## Tahap 3: Web Dashboard (Branch: feature/ui-dashboard)
* Router 1 (`/`): Halaman Dashboard. Menampilkan rangkuman prediksi total bulan depan.
* Router 2 (`/predict`): Halaman Form. Input berupa form kategori produk (family) dan tanggal. Output berupa tabel rekomendasi angka restock (berdasarkan angka prediksi model).
* Router 3 (`/evaluation`): Menampilkan metrik MAE/RMSE statis dari model yang sedang dipakai untuk validasi.