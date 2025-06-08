# Laporan Proyek Machine Learning - Prediksi Diabetes Menggunakan Data Kesehatan Pasien

## 1. Domain Proyek

### Latar Belakang

Diabetes adalah penyakit kronis yang sangat mempengaruhi kualitas hidup serta memiliki risiko komplikasi serius seperti penyakit jantung, stroke, gagal ginjal, dan kebutaan. Menurut data dari World Health Organization (WHO), jumlah penderita diabetes meningkat tajam dari tahun ke tahun, terutama di negara berkembang. Oleh karena itu, deteksi dini dan penanganan proaktif sangat penting untuk mencegah konsekuensi jangka panjang dari penyakit ini.

Model prediksi berbasis machine learning (ML) memberikan peluang besar untuk mendeteksi diabetes secara dini menggunakan data medis pasien. ML mampu mempelajari pola kompleks dari data dan menghasilkan prediksi yang akurat serta cepat, yang dapat membantu tenaga medis dalam pengambilan keputusan klinis.

### Referensi/Rubrik Tambahan

1.  World Health Organization. (2024). *Diabetes Fact Sheet*. [WHO Diabetes Report](https://www.who.int/news-room/fact-sheets/detail/diabetes)
2. Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). [Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus](https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/). Proceedings of the Annual Symposium on Computer Application in Medical Care, 261–265.

Dataset: [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data)

---

## 2. Business Understanding

### Problem Statements

* Bagaimana memprediksi apakah seorang pasien menderita diabetes berdasarkan data medis?
* Model klasifikasi apa yang paling efektif dan akurat untuk digunakan dalam prediksi risiko diabetes?

### Goals

* Mengembangkan model machine learning yang mampu memprediksi kemungkinan seorang pasien terkena diabetes.
* Membandingkan performa berbagai model untuk menentukan algoritma terbaik.

### Solution Statement/Matrik Tambahan

Untuk mencapai tujuan tersebut, solusi yang dilakukan adalah:

1. **Menggunakan tiga algoritma ML:** Logistic Regression, Random Forest, dan XGBoost.
2. **Melakukan preprocessing data**, termasuk imputasi nilai kosong dan normalisasi.
3. **Mengatasi class imbalance menggunakan SMOTE.**
4. **Mengevaluasi model** dengan metrik: Accuracy, Precision, Recall, F1 Score, dan ROC AUC.
5. **Menyimpan model terbaik** untuk kebutuhan deployment.

---

## 3. Data Understanding

Dataset yang digunakan adalah *Pima Indians Diabetes Dataset* yang terdiri dari 768 data pasien perempuan keturunan Pima Indian dengan 8 fitur prediktor dan 1 label (Outcome).
[Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### Variabel Fitur:

| Fitur                      | Deskripsi                       |
| -------------------------- | ------------------------------- |
| `Pregnancies`              | Jumlah kehamilan                |
| `Glucose`                  | Kadar glukosa plasma            |
| `BloodPressure`            | Tekanan darah diastolik         |
| `SkinThickness`            | Ketebalan lipatan kulit triceps |
| `Insulin`                  | Kadar insulin dalam darah       |
| `BMI`                      | Indeks massa tubuh              |
| `DiabetesPedigreeFunction` | Faktor keturunan diabetes       |
| `Age`                      | Usia pasien                     |
| `Outcome`                  | 1 = diabetes, 0 = tidak         |

### Eksplorasi Data/Rubrik Tambahan:

* Cek null dan tipe data dengan `info()`
* Statistik deskriptif dengan `describe()`

---

## 4. Data Preparation

### Tahapan:

1. **Nilai Tidak Valid:** Mengganti nilai nol pada fitur medis (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) dengan NaN.
2. **Imputasi Median:** Mengisi NaN dengan nilai median kolom terkait.
3. **Normalisasi:** Skala fitur dengan `StandardScaler`.
4. **Train-Test Split:** Proporsi 70:30.
5. **SMOTE:** Menyeimbangkan distribusi target `Outcome`.

### Alasan/Rubrik Tambahan:

* Nilai nol secara medis tidak valid untuk parameter seperti tekanan darah atau glukosa.
* Normalisasi diperlukan untuk algoritma yang sensitif terhadap skala seperti Logistic Regression.
* SMOTE meningkatkan kemampuan model mendeteksi kelas minoritas.

---

## 5. Modeling

### Model yang Digunakan:

#### 1. Logistic Regression

* `max_iter=1000`
* Cocok sebagai baseline.
* Kelemahan: Tidak menangkap hubungan non-linear.

#### 2. Random Forest

* `n_estimators=100`
* Model ensemble berbasis decision tree.
* Kelebihan: Tahan terhadap overfitting, menangani fitur non-linear.

#### 3. XGBoost

* `learning_rate=0.1`, `max_depth=3`
* Model boosting canggih.
* Kelebihan: Akurasi tinggi, efisiensi tinggi, mendukung regularisasi.

### Proses:

* Data hasil SMOTE digunakan untuk training.
* Data testing menggunakan data asli (tanpa SMOTE) untuk memastikan evaluasi objektif.

---

## 6. Evaluation

### Metrik Evaluasi:

* **Accuracy**: Proporsi prediksi yang benar.
* **Precision**: TP / (TP + FP)
* **Recall**: TP / (TP + FN)
* **F1 Score**: Harmonik Precision dan Recall.
* **ROC AUC**: Mengukur kemampuan model dalam membedakan kelas.

### Hasil:

| Model               | Accuracy   | AUC-ROC    | Precision (1) | Recall (1) |
| ------------------- | ---------- | ---------- | ------------- | ---------- |
| Random Forest       | **0.7532** | **0.8053** | 0.3493        | 1.0000     |
| XGBoost             | 0.7186     | 0.8003     | 0.3478        | 1.0000     |
| Logistic Regression | 0.7013     | 0.7999     | 0.3478        | 1.0000     |

### Analisis:

* Semua model memiliki Recall sempurna, cocok untuk kasus kritis.
* Precision masih rendah, menunjukkan banyak false positive.
* **Random Forest unggul dalam Accuracy dan AUC-ROC**, dipilih sebagai model terbaik.

### Visualisasi:

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Feature Importance

---

## 7. Deployment

Model terbaik disimpan menggunakan `joblib` untuk implementasi lebih lanjut.

```python
import joblib
joblib.dump(rf_model, 'random_forest_diabetes.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

Model dapat digunakan dalam API atau aplikasi prediksi berbasis web.

---

## 8. Kesimpulan

* Machine learning dapat digunakan untuk prediksi diabetes dengan akurasi yang baik.
* Random Forest terbukti memberikan hasil terbaik dibanding Logistic Regression dan XGBoost.
* Recall tinggi penting untuk aplikasi medis karena dapat mendeteksi semua pasien berisiko.
* Precision yang rendah masih menjadi tantangan dan dapat diperbaiki di pengembangan selanjutnya.

---

## 9. Referensi

* Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus. Proceedings of the Annual Symposium on Computer Application in Medical Care, 261–265.
* Dataset: [Kaggle - Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data)
