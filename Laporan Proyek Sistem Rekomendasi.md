# Laporan Proyek Machine Learning - Sistem Rekomendasi Buku

## Project Overview

Sistem rekomendasi berperan penting dalam meningkatkan pengalaman pengguna, terutama di platform berbasis konten seperti toko buku online atau aplikasi pembaca digital. Dalam proyek ini, kami membangun sistem rekomendasi buku yang dipersonalisasi berdasarkan riwayat interaksi pengguna dan informasi konten buku.

Masalah ini penting karena sistem rekomendasi yang baik dapat meningkatkan keterlibatan pengguna, memperluas jangkauan bacaan mereka, serta mendukung penjualan dan retensi pengguna.

**Referensi**:
- Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.
- Resnick, P., & Varian, H. R. (1997). *Recommender systems*. Communications of the ACM, 40(3), 56–58.

## Business Understanding

### Problem Statements

- Bagaimana merekomendasikan buku yang relevan dan menarik bagi setiap pengguna?
- Bagaimana membangun sistem yang memanfaatkan informasi buku dan riwayat pembacaan pengguna?

### Goals

- Menghasilkan daftar Top-N rekomendasi buku yang dipersonalisasi.
- Meningkatkan relevansi rekomendasi melalui pendekatan gabungan: konten dan kolaboratif.

### Solution Approach

Untuk mencapai tujuan di atas, kami mengusulkan dua pendekatan sistem rekomendasi:

1. **Content-Based Filtering**  
   Rekomendasi berdasarkan konten dari fitur buku seperti judul.

2. **Collaborative Filtering**  
   Menggunakan data interaksi antar pengguna untuk menemukan pola kesamaan dan memberikan rekomendasi.

## Data Understanding

Dataset yang digunakan diunduh dari Kaggle dan terdiri dari tiga file utama:
- Books.csv: berisi metadata buku
- Ratings.csv: berisi rating yang diberikan pengguna terhadap buku
- Users.csv: berisi informasi pengguna

Sumber dataset:  
[Kaggle - Book Recomendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

### Fitur-Fitur dalam Dataset:
1. Books.csv (271,369 entries):
   - ISBN (Object): Kode unik buku
   - Book-Title (Object): Judul buku
   - Book-Author (Object): Penulis
   - Year-Of-Publication (Object): Tahun terbit
   - Publisher (Object): Penerbit
   - Image-URL (3 kolom): Link gambar cover
2. Ratings.csv (1,149,780 entries):
   - User-ID (int64): ID unik pengguna
   - ISBN (Object): Kode buku
   - Book-Rating (int64): Rating 0-10
3. Users.csv (278,858 entries):
   - User-ID (int64): ID pengguna
   - Location (Object): Lokasi pengguna
   - Age (float64): Usia pengguna

#### 1. Analisis Distribusi Rating
```python
# Visualisasi distribusi rating eksplisit (1-10)
plt.figure(figsize=(10,5))
sns.countplot(x='Book-Rating', data=ratings[ratings['Book-Rating'] > 0])
plt.title('Distribusi Rating Eksplisit (halaman 4 PDF)')
```
Insight:
- Rating dominan pada nilai 8 (23% dari total) kecuali rating 0 karena itu merupakan faktor implisit
- Hanya 5% rating dibawah 5
- Membuat plot distribusi rating buku yang diberikan oleh pengguna untuk mengetahui pola atau bias dalam rating. Menampilkan statistik deskriptif umur pengguna serta visualisasi distribusi umur pengguna dalam dataset.

Beberapa temuan awal:
- Mayoritas pengguna hanya memberi sedikit rating (banyak data sparsity)
- Banyak buku yang hanya dirating sekali atau dua kali
- Penulis populer (seperti J.K. Rowling) cenderung memiliki rating lebih banyak

## Data Preparation

Langkah-langkah yang kami lakukan:

1. **Data Cleaning**  
   - Membersihkan data dari duplikat dan nilai-nilai yang tidak valid seperti umur ekstrem atau rating nol
   - Membuang kolom yang tidak relevan seperti URL gambar.

2. **Data Filtering**  
   - Mengambil subset dari data sebanyak 30.000 baris untuk mempercepat proses training dan menghindari beban komputasi berlebih.

3. **Merge Data**  
   - Menggabungkan dataset ratings, books, dan users menjadi satu dataframe utama (full_data) dan membuang kolom yang tidak diperlukan untuk fokus pada fitur penting

## Modeling

Kami membangun dua jenis sistem rekomendasi:

### 1. Content-Based Filtering

- Menggunakan TF-IDF vectorizer untuk fitur konten dari judul
- Menghitung similarity antar buku menggunakan cosine similarity
- Memberikan rekomendasi berdasarkan judul buku

Kelebihan:
- Tidak memerlukan data pengguna lain
- Cocok untuk pengguna baru

Kekurangan:
- Hanya bisa merekomendasikan buku yang mirip dengan buku yang sudah diberi rating

Menggunakan TF-IDF untuk mengubah judul buku menjadi representasi numerik dan menghitung cosine similarity antar judul untuk keperluan content-based recommendation. Fungsi content_based_recommendation menerima input judul buku dan mengembalikan daftar buku yang mirip berdasarkan kemiripan kontennya (judul), menggunakan cosine similarity.

**Output**:  
Top-5 rekomendasi buku berdasarkan konten yang mirip dengan judul buku

### 2. Collaborative Filtering (Matrix Factorization)

- Menggunakan teknik Singular Value Decomposition (SVD) dari `Surprise` library
- Mengestimasi rating yang belum diberikan pengguna
- Memberikan rekomendasi berdasarkan prediksi rating tertinggi

Kelebihan:
- Bisa menangkap pola kompleks dalam interaksi user-item
- Rekomendasi personal

Kekurangan:
- Membutuhkan cukup banyak data pengguna
- Tidak bekerja untuk user/item baru (cold start problem)

Melakukan encoding terhadap kolom user dan judul buku menggunakan LabelEncoder agar bisa digunakan dalam embedding layer pada model deep learning. Membagi data menjadi data pelatihan dan validasi untuk mengevaluasi performa model collaborative filtering. Membangun model rekomendasi berbasis neural network sederhana dengan dua input embedding: satu untuk user dan satu untuk buku. Output berupa prediksi rating. Melatih model menggunakan data pelatihan dan validasi selama 5 epoch, serta mencatat history loss-nya untuk evaluasi lebih lanjut.

**Output**:  
Top-5 buku dengan estimasi rating tertinggi untuk setiap pengguna.

### Perbandingan Kedua Pendekatan

| Pendekatan             | Kelebihan                                  | Kekurangan                                  |
|------------------------|--------------------------------------------|---------------------------------------------|
| Content-Based          | Tidak memerlukan banyak data pengguna lain | Cenderung memberi rekomendasi sempit        |
| Collaborative Filtering| Bisa menangkap preferensi yang lebih luas  | Cold-start problem jika pengguna/item baru  |

## Evaluation
### Evaluation Model Content Based Filtering
Model Content Based Filtering dievaluasi seberapa presisi hasil terhadap input yang diberikan. Evaluasi ini mengukur seberapa banyak hasil rekomendasi yang mengandung kata kunci dari input judul (misalnya: "Baby").

**Rumus:**

$$
\text{Precision} = \frac{\text{Jumlah judul yang mengandung kata kunci}}{\text{Jumlah total rekomendasi}}
$$

**Contoh:**
- Input Judul: **"Baby"**
- Hasil Rekomendasi:
  1. He's My Baby Now ✅
  2. Baby ✅
  3. One for My Baby : A Novel ✅
  4. Baby Elephant (Baby) ✅
  5. Baby, Oh Baby! (Time of Your Life) ✅

**Hasil:**

| Metrik           | Nilai    |
|------------------|----------|
| Precision        | 1.00     |
| Total Rekomendasi| 5        |
| Match Judul      | 5        |

### Evaluation Model Collaborative Filtering
Model Collaborative Filtering dievaluasi menggunakan dua metrik regresi umum: **Mean Squared Error (MSE)** dan **Root Mean Squared Error (RMSE)**.  
Keduanya digunakan untuk mengukur sejauh mana prediksi model menyimpang dari nilai rating sebenarnya. Model Collaborative Filtering dievaluasi menggunakan MSE dan RMSE, dua metrik regresi umum. Nilai RMSE menunjukkan seberapa besar deviasi prediksi terhadap rating sebenarnya. Distribusi rating asli dan prediksi juga diplot untuk melihat apakah model mampu meniru pola rating dari data asli.

### Rumus

- **MSE (Mean Squared Error)**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

MSE mengukur rata-rata selisih kuadrat antara rating aktual dan prediksi.

- **RMSE (Root Mean Squared Error)**

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

RMSE lebih mudah diinterpretasikan karena satuannya sama dengan rating sebenarnya.

### Hasil Evaluasi Model Collaborative Filtering
- MSE: 3.6381
- RMSE: 1.9074

## Kesimpulan
Proyek ini mengembangkan dua pendekatan utama dalam sistem rekomendasi yaitu Content-Based Filtering dan Collaborative Filtering. Kesimpulan dari proyek ini yaitu:
- Content-Based Filtering menggunakan TF-IDF pada judul buku untuk menghitung kemiripan antar buku berdasarkan kontennya. Model content-based dapat digunakan saat hanya informasi buku yang tersedia.
- Collaborative Filtering menggunakan Neural Matrix Factorization (model deep learning dengan embedding) untuk mempelajari pola interaksi antara pengguna dan buku. Model collaborative filtering memberikan rekomendasi yang lebih personal dan fleksibel, tapi membutuhkan interaksi pengguna.
- Model collaborative filtering dievaluasi menggunakan metrik Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE). Hasil evaluasi menunjukkan performa yang cukup baik untuk memberikan prediksi rating, dengan tren training dan validation loss yang stabil. Model juga berhasil memberikan rekomendasi buku untuk pengguna tertentu secara personal.
- Sistem rekomendasi seperti ini dapat diterapkan pada platform toko buku online atau aplikasi pembaca digital untuk meningkatkan pengalaman pengguna.

Saran pengembangan:
- Menambahkan metadata buku seperti genre atau sinopsis
- Menggabungkan content dan collaborative filtering (hybrid)
- Mengatasi masalah cold-start dengan pendekatan berbasis metadata
