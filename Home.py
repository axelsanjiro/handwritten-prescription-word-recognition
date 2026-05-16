import streamlit as st

# Configure page
st.set_page_config(page_title="Medical OCR - Home", layout="wide")

# home page content
st.title("Handwritten Prescription Word Recognition")
st.markdown("### **Super Feature Fusion using Classical Machine Learning**")
st.divider()

col_main, col_side = st.columns([2.5, 1])

with col_side:
    st.markdown("### Anggota Kelompok: ")
    st.write("""
    **Group 3:**
    * **Abraham Gregorius Anderson Thio - 2802473504**
    * **Alwan Athallah Mumtaz - 2802473896**
    * **Axel Sanjiro Yang - 2802472400**
    * **Stanislaus Alva Jufinto - 2802473214**
    ---
    *Final Project - Computer Vision*
    """)

with col_main:
    # 2. Overview
    st.markdown("### Overview: ")
    st.write("""
    Pengenalan kata pada resep medis tulisan tangan dokter merupakan salah satu tugas *Optical Character Recognition* (OCR) yang paling menantang. Hal ini disebabkan oleh gaya tulisan guratan miring (*cursive*) yang tidak beraturan serta tingkat kemiripan visual yang sangat tinggi antar nama obat farmasi. Proyek ini mengajukan sebuah sistem pengenalan berbasis pendekatan **Classical Machine Learning** yang dirancang agar bersifat **lightweight (ringan)** dan hemat resource komputasi.
    
    Berbeda dengan model *Deep Learning* konvensional yang membutuhkan dataset raksasa serta akselerasi hardware (GPU) yang masif, sistem ini membuktikan bahwa kombinasi ekstraksi fitur konvensional (*handcrafted features*) yang dioptimalkan dapat menjadi solusi alternatif yang andal, cepat, dan efisien untuk diterapkan pada perangkat keras standar.
    """)

    # 3. Objectives
    st.markdown("### Objectives: ")
    st.write("""
    * **Feature Fusion:** Menggabungkan karakteristik struktural, tekstur mikro, frekuensi spasial, dan bentuk global secara bersamaan guna menutupi kelemahan keterbatasan fitur tunggal (*Super Feature Fusion*).
    * **Efficiency:** Menerapkan reduksi dimensi matematis untuk menyaring *noise* latar belakang tanpa menghilangkan informasi diskriminatif utama dari bentuk guratan huruf.
    * **Visual Interpretability (XAI):** Menyediakan transparansi visual berbasis peta panas (*Simulated Grad-CAM*) untuk memvalidasi secara empiris fokus area pengenalan dari setiap algoritma ekstraksi fitur yang digunakan.
    """)

    # 4. Dataset Specification
    st.markdown("### Dataset Specification: ")
    st.write("""
    * **Dataset Name:** Doctors' Handwritten Prescription BD Dataset (Kaggle).
    * **Total Samples:** 4.680 sampel citra potongan tulisan tangan tingkat kata (*word-level samples*).
    * **Number of Classes:** 78 nama obat-obatan farmasi (misalnya: *Aceta, Napa, dll.*).
    * **Image Resolution:** Di-standardisasi melalui proses resize menjadi ukuran **$64 \\times 64$ piksel** dengan skala normalisasi nilai piksel [0, 1].
    * **Evaluation Method:** Pembagian data dengan rasio **80% Training dan 20% Testing**.
    """)

st.divider()
col_method, col_nav = st.columns(2)

with col_method:
    # 5. Methodology Pipeline
    st.markdown("### Pipeline Metodologi: ")
    st.write("""
    1. **Preprocessing Layer:** Proses konversi ke citra *grayscale*, penerapan binarisasi (*thresholding inverse*), serta penyesuaian ukuran dimensi ke standar matriks $64 \\times 64$.
    2. **Feature Extraction Layer (Super Fusion):**
       * *Histogram of Oriented Gradients (HOG):* Menangkap orientasi tepi makro dan kerangka stroke huruf.
       * *Local Binary Pattern (LBP):* Ekstraksi variasi tekstur mikro dan karakteristik ketebalan goresan tinta.
       * *Gabor Filters:* Mengisolasi arah frekuensi spasial gelombang guratan pada sudut kemiringan $45^\circ$.
       * *Zernike Moments:* Deskriptor bentuk global lingkaran unit yang bersifat invariant terhadap rotasi citra.
    3. **Dimensionality Reduction Layer:** Penyelarasan skala menggunakan *StandardScaler*, dilanjutkan reduksi dimensi via *Principal Component Analysis* (PCA) dengan mempertahankan **95% varians data informasi**.
    4. **Classification Layer:** Tahap klasifikasi komparatif menggunakan tiga model utama: *Support Vector Machine* (SVM) dengan RBF Kernel, *K-Nearest Neighbors* (KNN), dan *Random Forest* (RF).
    """)

with col_nav:
    # 6. Navigation Guide
    st.markdown("### Panduan Navigasi Halaman: ")
    st.write("""
    Silakan gunakan menu navigasi pada komponen *sidebar* di sebelah kiri untuk berpindah halaman:
    * **1. EDA:** Menampilkan analisis eksplorasi data, statistik sebaran kelas, dan contoh sampel gambar tulisan asli dari dataset.
    * **2. Preprocessing:** Modul pengujian interaktif untuk mengatur parameter *threshold* binarisasi dan resolusi ukuran gambar secara langsung (*real-time*).
    * **3. Feature Extraction (XAI):** Fitur utama Computer Vision untuk mengekstrak komponen fitur dan memetakan visualisasi *heatmap overlay* transparan bergaya Grad-CAM untuk setiap algoritma.
    * **4. Classification:** Halaman laporan hasil metrik performa komparatif akurasi, nilai *F1-Macro*, serta efisiensi durasi waktu proses latih dari ketiga model klasifikasi.
    * **5. Prediction Demo:** Antarmuka simulasi *end-to-end* untuk mengunggah gambar potongan resep baru dari komputer dan memprediksi nama obat farmasinya secara instan.
    """)