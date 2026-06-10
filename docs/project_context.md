# Project Context: Retail Demand Forecasting

## Tujuan Bisnis
Mengembangkan aplikasi prediksi permintaan produk retail untuk membantu manajer mengoptimalkan persediaan barang, dengan fokus menekan risiko overstock dan stockout.

## Sumber Data
Dataset berasal dari Kaggle (Store Sales - Time Series Forecasting).

Link: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

Pemodelan akan menggunakan variabel independen utama (tanggal, id toko, kategori produk, status promosi) dan diperkuat dengan variabel eksogen (harga minyak makroekonomi dan libur nasional). Target prediksi (dependent variable) adalah jumlah penjualan (sales).

## Arsitektur Sistem
1. Data Preprocessing (Pembersihan, penanganan missing values, penggabungan variabel eksogen).
2. Exploratory Data Analysis (Statistik deskriptif dasar).
3. Machine Learning Regression (Mengevaluasi Linear Regression, Decision Tree, Random Forest, dan Gradient Boosting). Metrik utama: MAE dan RMSE.
4. Web Application menggunakan arsitektur MVC (FastAPI + Jinja2 Templates).