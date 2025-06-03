# Laporan Proyek Machine Learning - Mohamad Farrel Aryansyah

## Project Overview

Ponsel telah menjadi bagian esensial dalam kehidupan modern, tidak hanya sebagai alat komunikasi, tetapi juga sebagai sarana hiburan, produktivitas, dan akses informasi. Karena variasi spesifikasi, merek, dan harga yang sangat luas, memilih ponsel yang sesuai dengan kebutuhan individu menjadi semakin kompleks. Di tengah banyaknya pilihan di pasar, konsumen sering kali mengalami kesulitan dalam menemukan produk yang benar-benar relevan dengan preferensi mereka. Oleh karena itu, sistem yang mampu memberikan rekomendasi produk secara personal menjadi sangat dibutuhkan, baik oleh konsumen maupun oleh platform e-commerce.

Kebutuhan akan sistem rekomendasi yang efektif juga dirasakan oleh pelaku industri. Tanpa sistem rekomendasi yang tepat, pengguna bisa merasa kewalahan atau bahkan meninggalkan platform karena tidak menemukan produk yang sesuai. Di sisi lain, penjual berpotensi kehilangan peluang karena produknya tidak ditampilkan ke target pengguna yang relevan. Untuk mengatasi masalah ini, pendekatan berbasis machine learning dapat digunakan untuk membangun sistem rekomendasi yang mampu menganalisis pola interaksi pengguna dan karakteristik produk, lalu menghasilkan rekomendasi yang akurat dan dipersonalisasi.

Beberapa studi menunjukkan efektivitas sistem rekomendasi dalam meningkatkan kepuasan pengguna dan konversi penjualan. Misalnya, Adomavicius dan Tuzhilin \[1] menekankan bahwa sistem rekomendasi modern mampu memberikan nilai bisnis yang signifikan melalui peningkatan loyalitas dan pengalaman pengguna. Selain itu, studi oleh Jannach et al. \[2] mengungkap bahwa pendekatan seperti content-based filtering dan collaborative filtering dapat saling melengkapi dalam menyajikan rekomendasi yang lebih relevan, tergantung pada ketersediaan data dan konteks pengguna. Oleh karena itu, pengembangan sistem rekomendasi berbasis machine learning merupakan solusi strategis yang layak untuk diimplementasikan dalam domain produk teknologi seperti ponsel.

## Business Understanding

Di era digital saat ini, pengguna dihadapkan pada banyak pilihan produk ponsel yang beragam, baik dari segi merek, spesifikasi, harga, maupun fitur. Banyaknya variasi ini justru dapat membuat pengguna merasa bingung dan kesulitan dalam menentukan pilihan yang paling sesuai dengan kebutuhan dan preferensinya. Di sisi lain, penyedia platform e-commerce dan toko online menghadapi tantangan besar dalam menyajikan produk yang relevan secara personal kepada setiap pengguna agar dapat meningkatkan kepuasan dan loyalitas pelanggan.

Masalah utama yang muncul adalah bagaimana membantu pengguna menemukan ponsel yang tepat di tengah lautan produk yang sangat banyak dan beragam, serta bagaimana meningkatkan keterlibatan pengguna sehingga potensi konversi penjualan dapat dimaksimalkan. Tanpa sistem rekomendasi yang efektif, pengguna mungkin merasa overwhelmed dan cepat meninggalkan platform, sementara peluang bisnis untuk meningkatkan penjualan menjadi terlewatkan.

Berikut adalah versi **revisi** dari bagian **Problem Statements, Goals, dan Solution Approach** tanpa menggunakan emotikon, dengan gaya penulisan formal dan sesuai dengan kriteria proyek akhir Machine Learning Dicoding:

### Problem Statements

Dalam era digital, pengguna dihadapkan pada banyak pilihan produk smartphone dengan berbagai variasi spesifikasi, harga, dan merek. Hal ini dapat menyulitkan pengguna dalam menentukan produk yang paling sesuai dengan kebutuhan dan preferensi mereka. Tanpa adanya sistem yang mampu memahami preferensi pengguna, proses pencarian produk yang relevan menjadi kurang efisien dan tidak personal.

Proyek ini bertujuan untuk menjawab dua pertanyaan utama berikut:

* Bagaimana membangun sistem rekomendasi smartphone yang mampu memberikan saran secara personal berdasarkan perilaku dan pola interaksi pengguna?
* Bagaimana memprediksi produk yang berpotensi menarik bagi pengguna dengan memanfaatkan data interaksi historis mereka?

### Goals

Tujuan dari proyek ini adalah:

* Mengembangkan sistem rekomendasi smartphone dengan menerapkan dua pendekatan utama, yaitu Content-Based Filtering dan Collaborative Filtering.
* Menyediakan rekomendasi produk dalam bentuk Top-N recommendation yang relevan dengan preferensi masing-masing pengguna.
* Mengevaluasi dan membandingkan performa kedua pendekatan untuk mengetahui metode yang paling efektif berdasarkan data yang tersedia.

### Solution Approach

Untuk mencapai tujuan tersebut, solusi dikembangkan melalui dua pendekatan berikut:

* **Content-Based Filtering**
  Pendekatan ini merekomendasikan produk berdasarkan kemiripan atribut produk dengan produk yang sebelumnya disukai atau dilihat oleh pengguna. Informasi produk seperti merek, kategori, dan spesifikasi digunakan untuk menghitung tingkat kesamaan antar produk.

* **Collaborative Filtering**
  Pendekatan ini memanfaatkan data interaksi historis antara pengguna dan produk untuk menemukan pola dan preferensi. Rekomendasi diberikan berdasarkan preferensi pengguna lain yang memiliki pola interaksi yang serupa.

## Data Understanding

Dataset yang digunakan berisi informasi lengkap mengenai berbagai model ponsel, meliputi merek, tipe, sistem operasi, serta berbagai fitur teknis lainnya. Dataset ini tersedia secara publik dan dapat diunduh dari [Kaggle](https://www.kaggle.com/datasets/meirnizri/cellphones-recommendations).

Dataset ini terbagi menjadi tiga bagian utama:

* data ponsel (cellphones data)
* data rating ponsel (cellphones rating)
* data pengguna (cellphones users).

**Data ponsel** terdiri dari 33 baris dan 14 kolom, tanpa adanya nilai yang hilang (missing value). Dataset ini menyajikan spesifikasi rinci dari setiap ponsel yang tercakup, memberikan gambaran lengkap tentang fitur teknis yang dimiliki oleh masing-masing model. Berikut adalah informasi umum dan contoh data dari dataset ponsel tersebut:

```python
# Load dataset
df_phones = pd.read_csv('cellphones data.csv')
df_ratings = pd.read_csv('cellphones ratings.csv')
df_users = pd.read_csv('cellphones users.csv')
```

### 1. Data Ponsel (`df_phones`)

**Informasi Kolom**

| Kolom            | Tipe Data | Keterangan                      |
|------------------|-----------|--------------------------------|
| cellphone_id     | int64     | ID unik ponsel                 |
| brand            | object    | Merek ponsel                  |
| model            | object    | Model ponsel                  |
| operating system | object    | Sistem operasi ponsel         |
| internal memory  | int64     | Kapasitas memori internal (GB)|
| RAM              | int64     | Kapasitas RAM (GB)            |
| performance      | float64   | Skor performa ponsel          |
| main camera      | int64     | Resolusi kamera utama (MP)    |
| selfie camera    | int64     | Resolusi kamera depan (MP)    |
| battery size     | int64     | Kapasitas baterai (mAh)       |
| screen size      | float64   | Ukuran layar (inci)           |
| weight           | int64     | Berat ponsel (gram)           |
| price            | int64     | Harga ponsel (USD)            |
| release date     | object    | Tanggal rilis ponsel          |

**Contoh Data**

| cellphone_id | brand | model          | operating system | internal memory | RAM | performance | main camera | selfie camera | battery size | screen size | weight | price | release date |
|--------------|-------|----------------|------------------|-----------------|-----|-------------|-------------|---------------|--------------|-------------|--------|-------|--------------|
| 0            | Apple | iPhone SE (2022)| iOS              | 128             | 4   | 7.23        | 12          | 7             | 2018         | 4.7         | 144    | 429   | 18/03/2022   |

- **Jumlah data**: 33 baris, 14 kolom  
- **Data lengkap**: Tidak ada nilai yang hilang (semua kolom terisi penuh)  
- **Tipe data**:  
  - Numerik: 10 kolom (8 kolom bertipe `int64`, 2 kolom bertipe `float64`)  
  - Kategorikal: 4 kolom (`object`)  

### 2. Data Rating (`df_ratings`)

**Informasi Kolom**

| Kolom        | Tipe Data | Keterangan                      |
|--------------|-----------|--------------------------------|
| user_id      | int64     | ID pengguna                    |
| cellphone_id | int64     | ID ponsel                     |
| rating       | int64     | Rating pengguna terhadap ponsel (skala tidak standar, nilai max 18) |

- **Jumlah data**: 990 baris, 3 kolom  
- **Data lengkap**: Tidak ada nilai yang hilang  
- **Tipe data**: Semua kolom bertipe numerik `int64`  

**Statistik Rating**

| Statistik | Nilai   |
|-----------|---------|
| Count     | 990     |
| Mean      | 6.7     |
| Median    | 7       |
| Min       | 1       |
| Max       | 18      |
| 25%       | 5       |
| 75%       | 9       |

**Informasi Tambahan**

- `user_id` tersebar dari 0 hingga 258  
- `cellphone_id` mencakup 33 ponsel dengan ID 0 sampai 32  

### 3. Data Pengguna (`df_users`)

**Informasi Kolom**

| Kolom      | Tipe Data | Keterangan                    |
|------------|-----------|------------------------------|
| user_id    | int64     | ID pengguna                  |
| age        | int64     | Usia pengguna                |
| gender     | object    | Jenis kelamin pengguna       |
| occupation | object    | Pekerjaan pengguna (ada 1 nilai hilang) |

**Contoh Data**

| user_id | age | gender | occupation         |
|---------|-----|--------|--------------------|
| 0       | 38  | Female | Data analyst       |
| 1       | 40  | Female | team worker in it  |
| 6       | 55  | Female | IT                 |

- **Jumlah data**: 99 baris, 4 kolom  
- **Nilai hilang**: 1 nilai hilang pada kolom `occupation`  
- **Tipe data**:  
  - Numerik: 2 kolom (`int64`)  
  - Kategorikal: 2 kolom (`object`)

### Exploratory Data Analysis (EDA)

**EDA - `df_phones`**

![EDA_df_phones](https://github.com/user-attachments/assets/c50de444-d259-413f-bf0d-7a86c4c5b424)

* **Jumlah Ponsel per Merek**

  * Samsung mendominasi dengan jumlah ponsel terbanyak.
  * Diikuti oleh Apple dan Xiaomi.
  * Merek dengan jumlah ponsel paling sedikit adalah Asus, Sony, dan Vivo.

* **Sistem Operasi**

  * Mayoritas ponsel menggunakan **Android**.
  * Hanya sebagian kecil yang menggunakan **iOS** (khusus Apple).

* **RAM**

  * Ponsel dengan **RAM 6GB** paling umum ditemukan.
  * Diikuti oleh 4GB dan 8GB.
  * RAM 3GB dan 12GB hanya dimiliki sedikit ponsel.

* **Memori Internal**

  * Kapasitas **128GB** merupakan yang paling banyak digunakan.
  * Diikuti oleh 256GB, sedangkan kapasitas 512GB dan 64GB sangat jarang.

* **Kamera Utama**

  * Kamera 48MP paling banyak ditemukan.
  * Kamera 12MP juga cukup umum.
  * Kamera dengan resolusi sangat tinggi seperti 64MP dan 108MP hanya ada pada beberapa ponsel.

* **Kamera Selfie**

  * Resolusi kamera selfie sangat bervariasi.
  * Resolusi 20MP paling banyak ditemukan, disusul oleh 13MP dan 11MP.
  * Beberapa ponsel memiliki kamera selfie di bawah 10MP, namun jumlahnya lebih sedikit.

**EDA - `df_ratings`**

![EDA_df_ratings](https://github.com/user-attachments/assets/ec0b1294-d6e6-4fd3-a566-bc61e1f356c3)

* **Distribusi nilai rating**:

  * Rating paling umum: 8, diikuti oleh 7, 9, dan 10.
  * Sebagian besar pengguna memberi rating antara 6–10.
  * Terdapat outlier pada rating 18 (kemungkinan data salah input).

* **Jumlah rating per ponsel**:

  * Setiap ponsel mendapat jumlah rating yang cukup merata (sekitar 20–42 rating).
  * Ponsel dengan ID 30 memiliki jumlah rating terbanyak.

**EDA - `df_users`**

![EDA_df_users](https://github.com/user-attachments/assets/22b9e3fd-c33f-4153-9199-9d13ada63a4a)

* **Distribusi Usia**

  * Rentang usia pengguna paling banyak berada di **21–31 tahun**, dengan puncaknya pada usia **25 tahun**.
  * Setelah usia 35 tahun, jumlah pengguna menurun secara signifikan.
  * Terdapat sebaran hingga usia 61 tahun, namun jumlahnya lebih kecil.

* **Distribusi Gender**

  * Komposisi gender cukup seimbang:

    * **Laki-laki** sedikit lebih banyak dari perempuan.
    * Ada sebagian kecil data dengan gender tidak terisi (`-Select Gender-`).

* **Distribusi Pekerjaan**

  * Pengguna terbanyak berasal dari bidang `IT` dan `teknologi informasi`:
  * Pekerjaan seperti `Information Technology`, `IT`, `software developer`, `Data analyst`, dan `System Administrator` mendominasi.
  * Beberapa pekerjaan di bidang manajerial juga muncul cukup sering: `Manager`, `SALES MANAGER`, `Administrative officer`.
  * Selain itu, ada keberagaman pekerjaan dari sektor **kesehatan, pendidikan, keuangan, retail, konstruksi**, hingga **homemaker**.
  * Pengguna berasal dari latar belakang pekerjaan yang beragam. Namun, terdapat beberapa **inkonsistensi** penulisan yang perlu dibersihkan, seperti:
    * Kesalahan penulisan seperti `healthare` yang seharusnya adalah `healthcare`.
    * Pekerjaan `information technology` dan `IT` dapat digabungkan karena merujuk pada bidang yang sama.

## Data Preparation

Pada tahap ini, dilakukan sejumlah proses persiapan data yang mencakup penggabungan ketiga dataset, pembersihan data, transformasi, serta pembuatan fitur baru yang akan digunakan dalam pemodelan sistem rekomendasi.

**1. Penggabungan Dataset**

Terdapat tiga file yang dimuat, yaitu:

- `cellphones data.csv`: berisi spesifikasi teknis dari masing-masing ponsel.
- `cellphones ratings.csv`: berisi data rating yang diberikan pengguna terhadap ponsel tertentu.
- `cellphones users.csv`: berisi informasi demografis pengguna seperti usia, jenis kelamin, dan pekerjaan.

Ketiga dataset ini kemudian digabungkan menggunakan kolom `cellphone_id` dan `user_id` untuk membentuk satu data utama yang komprehensif.

**2. Pembersihan dan Standarisasi Data**

Beberapa langkah pembersihan data dilakukan sebagai berikut:

- Kolom `release date` dikonversi menjadi format `datetime` untuk mendapatkan informasi tahun rilis.
- Menghapus nilai kosong yang tersisa setelah penggabungan data.
- Menghapus nilai rating yang melebihi batas normal (dalam hal ini `rating = 18`).
- Menyaring hanya nilai rating yang valid (≤ 5).
- Mengoreksi kesalahan pengetikan pada kolom `occupation`, seperti `healthare` menjadi `healthcare`.
- Menyaring hanya data dengan gender yang valid (`Male`, `Female`).
- Menambahkan kolom `timestamp` sebagai penanda waktu preprocessing.
- Menghapus duplikat berdasarkan `cellphone_id`.

**3. Pembuatan Fitur untuk Content-Based Filtering**

Untuk sistem rekomendasi berbasis konten, dibuat sebuah fitur baru bernama `combined_features` yang menggabungkan informasi penting seperti `brand`, `model`, dan `operating system`. Fitur ini digunakan sebagai dasar dalam menghitung kemiripan antar ponsel menggunakan metode **TF-IDF vectorization** dan **cosine similarity**.

**4. Encoding dan Normalisasi untuk Collaborative Filtering**

Untuk sistem rekomendasi berbasis kolaboratif:

- Setiap `user_id` dan `cellphone_id` di-*encode* menjadi angka agar dapat diproses oleh model neural network.
- Nilai rating dinormalisasi ke skala 0–1 agar cocok digunakan dalam fungsi aktivasi sigmoid.
- Dataset diacak dan dibagi menjadi data latih (80%) dan validasi (20%).

Berikut adalah penulisan ulang tahap **Modeling** yang telah disesuaikan dengan **kriteria proyek akhir Dicoding**, termasuk penjelasan dua algoritma rekomendasi yang digunakan, kelebihan dan kekurangannya, serta penyajian *top-N recommendation* sebagai output:

## Modeling

Pada tahap ini dibangun dua pendekatan sistem rekomendasi untuk menyelesaikan permasalahan pemberian rekomendasi ponsel terbaik kepada pengguna, yaitu:

1. **Content-Based Filtering**
2. **Collaborative Filtering** (berbasis Deep Learning)

Kedua pendekatan dirancang untuk menghasilkan **top-N recommendation**, yaitu daftar ponsel yang paling relevan berdasarkan preferensi pengguna.

### 1. Content-Based Filtering

Pendekatan ini menyarankan ponsel berdasarkan **kemiripan konten** antar produk, seperti `brand`, `model`, dan `operating system`. Pendekatan ini cocok digunakan meskipun tidak tersedia data interaksi pengguna.

#### a. Proses

* **Ekstraksi Fitur**: Dibentuk kolom `combined_features` yang menggabungkan atribut konten utama.
* **Representasi Teks**: Diterapkan **TF-IDF Vectorizer** untuk mengubah data teks menjadi representasi numerik.
* **Pengukuran Kemiripan**: Digunakan **cosine similarity** untuk menghitung tingkat kemiripan antar produk.
* **Fungsi Rekomendasi**: Dibuat fungsi `recommend_similar_models()` untuk mengembalikan produk yang mirip terhadap input.

#### b. Implementasi Fungsi

```python
def recommend_similar_models(target_model, similarity_data=similarity_df, items=df_content[['model', 'brand', 'operating system']], top_n=4):
    ...
```

#### c. Contoh Output

```python
recommend_similar_models('iPhone 13')
```

**Rekomendasi yang Dihasilkan:**

* iPhone XR
* iPhone 13 Mini
* iPhone 13 Pro Max
* iPhone 13 Pro

Semua model berasal dari brand **Apple** dengan OS **iOS**, yang menunjukkan kemiripan konten yang tinggi.

#### d. Kelebihan & Kekurangan

| Kelebihan                                                 | Kekurangan                                                       |
| --------------------------------------------------------- | ---------------------------------------------------------------- |
| Tidak memerlukan data interaksi pengguna                  | Tidak bisa melakukan personalisasi preferensi pengguna           |
| Dapat menangani item baru (cold start) jika ada deskripsi | Terbatas pada atribut konten dan tidak adaptif terhadap feedback |

### 2. Collaborative Filtering

Pendekatan ini menggunakan data interaksi pengguna (rating) untuk mempelajari preferensi unik tiap individu. Model dibangun menggunakan pendekatan **Neural Collaborative Filtering (NCF)**.

#### a. Pra-pemrosesan

* **Encoding**: `user_id` dan `cellphone_id` diubah ke integer.
* **Pembersihan**: Menghapus rating outlier (`rating = 18`), normalisasi rating ke rentang \[0, 1].
* **Split Data**: Dibagi menjadi 80% data latih dan 20% validasi.

#### b. Arsitektur Model

Model `RecommenderNet` menggunakan:

* **Embedding Layer** untuk merepresentasikan pengguna dan item.
* **Bias Layer** untuk mengakomodasi bias dari user dan item.
* **Dot Product** sebagai fungsi kecocokan.
* **Sigmoid Output** untuk menghasilkan prediksi antara 0 dan 1.

```python
class RecommenderNet(tf.keras.Model):
    ...
```

#### c. Proses Pelatihan

* **Loss Function**: Binary Crossentropy
* **Optimizer**: Adam (lr=0.001)
* **Metric**: RMSE (Root Mean Squared Error)
* **EarlyStopping**: Untuk menghentikan pelatihan saat tidak ada perbaikan validasi

```python
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=8,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[early_stop],
    verbose=1
)
```

#### d. Fungsi Rekomendasi Pengguna

```python
def recommend_for_user(user_id, top_n=10):
    ...
```

Fungsi ini akan merekomendasikan ponsel yang belum pernah dinilai oleh pengguna dengan skor prediksi tertinggi.

#### e. Kelebihan & Kekurangan

| Kelebihan                                | Kekurangan                                                              |
| ---------------------------------------- | ----------------------------------------------------------------------- |
| Mampu menangkap preferensi unik pengguna | Tidak bisa memberikan rekomendasi jika user/item belum ada (cold start) |
| Akurat jika data interaksi mencukupi     | Memerlukan jumlah data pelatihan yang lebih besar                       |

## Evaluation

Evaluasi dilakukan untuk mengukur efektivitas kedua pendekatan sistem rekomendasi, baik secara kualitatif maupun kuantitatif.

### 1. Evaluasi Content-Based Filtering

Karena tidak melibatkan proses pembelajaran (unsupervised), evaluasi dilakukan secara **kualitatif**, menggunakan dua aspek:

* **Relevansi Rekomendasi**: Apakah ponsel yang disarankan memiliki fitur yang mirip.
* **Konsistensi**: Apakah sistem menghasilkan rekomendasi stabil untuk input yang mirip.

**Contoh:**

```python
recommend_similar_models('iPhone 13')
```

**Output:**

| Model             | Brand | OS  |
| ----------------- | ----- | --- |
| iPhone XR         | Apple | iOS |
| iPhone 13 Mini    | Apple | iOS |
| iPhone 13 Pro Max | Apple | iOS |
| iPhone 13 Pro     | Apple | iOS |

Rekomendasi di atas dinilai **relevan**, karena semua berasal dari brand dan OS yang sama, serta merupakan seri sejenis.

### 2. Evaluasi Collaborative Filtering

Evaluasi dilakukan secara **kuantitatif** menggunakan metrik **Root Mean Squared Error (RMSE)**, yang mengukur seberapa dekat prediksi model terhadap data aktual.

#### a. Formula RMSE

$$
RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 }
$$

#### b. Implementasi Evaluasi

```python
# Evaluasi data training
y_train_pred = model.predict(x_train).flatten()
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Evaluasi data validasi
y_val_pred = model.predict(x_val).flatten()
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

print(f"RMSE Train     : {rmse_train:.4f}")
print(f"RMSE Validation: {rmse_val:.4f}")
```

#### c. Hasil Evaluasi

| Dataset    | RMSE    |
| ---------- | ------- |
| Training   | 0.0262 |
| Validation | 0.2331 |

RMSE yang rendah menunjukkan bahwa model memiliki kemampuan generalisasi yang baik dan mempelajari pola interaksi pengguna secara efektif.

#### Visualisasi RMSE

Gambar berikut memperlihatkan perubahan nilai RMSE selama proses pelatihan:

![evaluation_rmse](https://github.com/user-attachments/assets/100a2e12-cc70-4cbd-97f9-70917a711ce6)

* **Train RMSE** menunjukkan penurunan yang sangat signifikan dari awal pelatihan dan terus menurun secara stabil hingga mendekati **0.025**, menandakan bahwa model mampu mempelajari pola data pelatihan dengan sangat baik.
* **Validation RMSE** juga mengalami penurunan secara bertahap, kemudian stabil di sekitar nilai **0.235**, menunjukkan bahwa performa model pada data validasi cukup baik.
* Tidak terlihat adanya indikasi *overfitting* yang mencolok, karena tidak terdapat kenaikan RMSE pada data validasi di akhir pelatihan.

Kurva ini mendukung kesimpulan bahwa model berhasil melakukan generalisasi terhadap data baru dan dapat diandalkan dalam melakukan prediksi.

### Contoh Rekomendasi Berdasarkan Collaborative Filtering

```python
recommend_for_user(user_id=1, top_n=10)
```

Model akan merekomendasikan 10 ponsel dengan prediksi rating tertinggi yang belum pernah dinilai oleh pengguna tersebut. Hasil yang ditampilkan mencerminkan preferensi unik berdasarkan riwayat rating pengguna.

## Kesimpulan

| Pendekatan         | Kelebihan                                             | Kekurangan                                               |
| ------------------ | ----------------------------------------------------- | -------------------------------------------------------- |
| Content-Based      | Tidak perlu data interaksi, bisa menangani item baru  | Tidak bisa personalisasi, terbatas pada atribut konten   |
| Collaborative (NN) | Mampu personalisasi, akurat bila data interaksi cukup | Sulit menangani cold start untuk pengguna atau item baru |

**Rekomendasi**: Kombinasi kedua pendekatan (hybrid) dapat memperkuat sistem rekomendasi dan mengatasi kekurangan masing-masing metode.

---

**Referensi**

\[1] G. Adomavicius and A. Tuzhilin, "Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions," *IEEE Transactions on Knowledge and Data Engineering*, vol. 17, no. 6, pp. 734–749, 2005.

\[2] D. Jannach, L. Lerche, F. Gedikli, and G. Bonnin, "What recommenders recommend – An analysis of accuracy, popularity, and sales diversity effects," *User Modeling and User-Adapted Interaction*, vol. 25, no. 5, pp. 353–388, 2015.
