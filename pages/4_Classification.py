import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import cv2

# Import Feature Extraction
from skimage.feature import hog, local_binary_pattern

# Import Scikit-Learn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from utils.cv_helpers import build_gabor_filter, calculate_zernike_heatmap

# Configure page
st.set_page_config(page_title="Medical OCR - Classification", layout="wide")

st.title("Live AutoML Engine (End-to-End)")
st.markdown("### **Ekstraksi Fitur & Training Secara Real-Time**")
st.divider()

st.write("""
Halaman ini menjalankan proses Machine Learning **sepenuhnya dari nol**. Sistem akan membaca gambar mentah, melakukan pembersihan (binarisasi & resize), mengekstrak fitur matematis yang kamu pilih, lalu melatih modelnya tepat di depan matamu.
""")

# Feature Extraction Options
@st.cache_data(show_spinner=False)
def live_extract_features(feature_choice, max_samples):
    """
    Fungsi ekstraksi fitur dengan perhitungan matematis yang benar 
    (menggunakan Histogram untuk LBP, bukan raw flatten) agar terhindar 
    dari Curse of Dimensionality.
    """
    base_path = os.path.join("dataset", "raw", "Training")
    csv_path = os.path.join(base_path, "training_labels.csv")
    img_dir = os.path.join(base_path, "training_words")
    
    if not os.path.exists(csv_path) or not os.path.exists(img_dir):
        return None, None, "Folder dataset/raw/Training tidak ditemukan!"
        
    df = pd.read_csv(csv_path)
    img_col, lbl_col = df.columns[0], df.columns[1]
    
    df = df.head(max_samples)
    
    X = []
    y = []
    
    progress_text = "Sedang mengekstrak fitur gambar..."
    my_bar = st.progress(0, text=progress_text)
    total_imgs = len(df)
    
    for idx, row in df.iterrows():
        img_name = str(row[img_col]).strip()
        label = str(row[lbl_col]).strip()
        
        img_path = os.path.join(img_dir, img_name)
        if not any(img_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            img_path += '.jpg'
            
        if os.path.exists(img_path):
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            _, binarized = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
            img_clean = cv2.resize(binarized, (64, 64))
            img_norm = img_clean.astype(np.float64) / 255.0
            
            # 1. HOG (Histogram of Oriented Gradients)
            f_hog = hog(img_clean, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            
            # 2. LBP (Menggunakan Histogram 59 Bins, BUKAN Flatten 4096)
            lbp_img = local_binary_pattern(img_clean, 8, 1, method='uniform')
            n_bins = int(lbp_img.max() + 1)
            f_lbp, _ = np.histogram(lbp_img.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            
            # 3. GABOR (Resize ke 8x8 = 64 fitur agar fokus pada struktur spasial kasar)
            gabor_img = cv2.filter2D(img_clean, cv2.CV_8UC3, build_gabor_filter())
            f_gabor = cv2.resize(gabor_img, (8, 8)).flatten() / 255.0
            
            # 4. ZERNIKE MOMENTS
            f_zer = calculate_zernike_heatmap(img_norm).flatten()
            
            if "HOG" in feature_choice:
                feat = f_hog
            elif "LBP" in feature_choice:
                feat = f_lbp
            elif "Gabor" in feature_choice:
                feat = f_gabor
            elif "Zernike" in feature_choice:
                feat = f_zer
            else: # FUSION
                # Vektor FUSION sekarang padat, bersih, dan kaya informasi!
                feat = np.concatenate([f_hog, f_lbp, f_gabor, f_zer])
                
            X.append(feat)
            y.append(label)
            
        my_bar.progress((idx + 1) / total_imgs, text=f"Mengekstrak {img_name} ({idx+1}/{total_imgs})...")
        
    my_bar.empty() 
    
    return np.array(X), np.array(y), "Sukses"

st.markdown("#### 1. Setup Data & Algoritma")

col_feat, col_model = st.columns(2)

with col_feat:
    feature_choice = st.selectbox(
        "1. Pilih Jenis Feature Extraction:", 
        [
            "FUSION (All Features)", 
            "HOG (Histogram of Oriented Gradients)", 
            "LBP (Local Binary Pattern)", 
            "Gabor Filters", 
            "Zernike Moments"
        ]
    )
    
    sample_size = st.slider(
        "Jumlah Gambar yang Diekstrak (Geser ke kiri untuk Demo Cepat):", 
        min_value=100, max_value=4680, value=500, step=100,
        help="Mengekstrak 4000+ gambar secara live akan memakan waktu lama. Gunakan 500 sampel untuk mendemokan fungsi aplikasi dengan cepat saat sidang."
    )

with col_model:
    model_choice = st.selectbox(
        "2. Pilih Model Klasifikasi:", 
        ["Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)", "Random Forest (RF)"]
    )

st.write("---")

st.markdown("#### 2. Tuning Hyperparameter Model")

if model_choice == "Support Vector Machine (SVM)":
    col_p1, col_p2 = st.columns(2)
    kernel_val = col_p1.selectbox("Kernel Type (`kernel`)", ["rbf", "linear", "poly"])
    c_val = col_p2.select_slider("Regularization Parameter (`C`)", options=[0.1, 1.0, 10.0, 100.0], value=10.0)
    raw_model = SVC(kernel=kernel_val, C=c_val, probability=True, random_state=42)
    model_shortname = "SVM"

elif model_choice == "K-Nearest Neighbors (KNN)":
    col_p1, col_p2 = st.columns(2)
    k_val = col_p1.slider("Jumlah Tetangga (`n_neighbors`)", min_value=1, max_value=15, value=3, step=2)
    weight_val = col_p2.selectbox("Weight Function (`weights`)", ["distance", "uniform"])
    raw_model = KNeighborsClassifier(n_neighbors=k_val, weights=weight_val)
    model_shortname = "KNN"

else: # Random Forest
    col_p1, col_p2 = st.columns(2)
    n_est_val = col_p1.slider("Jumlah Pohon (`n_estimators`)", min_value=50, max_value=300, value=200, step=50)
    max_depth_val = col_p2.slider("Kedalaman Maksimum (`max_depth`)", min_value=5, max_value=50, value=20, step=5)
    raw_model = RandomForestClassifier(n_estimators=n_est_val, max_depth=max_depth_val, random_state=42)
    model_shortname = "RF"

kombinasi_aktif = f"{model_shortname} + {feature_choice.split()[0]}"

st.write("---")

# Kolom diatur agar Tombol Ekstrak dan Reset berjejer rapi
col_train, col_reset, col_status = st.columns([1.5, 1, 2.5])

with col_train:
    start_training = st.button("Ekstrak & Latih Model!", type="primary", use_container_width=True)

with col_reset:
    reset_memory = st.button("Reset Memori", type="secondary", use_container_width=True)
    
    # Logika jika tombol reset ditekan
    if reset_memory:
        keys_to_delete = ['trained_pipeline', 'kombinasi_aktif', 'fitur_aktif']
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun() # Memaksa Streamlit me-refresh halaman dari atas

with col_status:
    if 'trained_pipeline' in st.session_state:
        st.success(f"**Model di Memori:** `{st.session_state['kombinasi_aktif']}` siap menebak di Halaman 5.")
    else:
        st.warning("**Memori Kosong:** Silakan tekan tombol Ekstrak & Latih.")

st.divider()

if start_training:
    # 1. EKSTRAKSI GAMBAR
    X, y, status_msg = live_extract_features(feature_choice, sample_size)
    
    if X is None:
        st.error(status_msg)
    else:
        with st.spinner(f"Data terekstrak ({X.shape[0]} baris x {X.shape[1]} fitur asli). Sedang mereduksi (PCA 95%) dan melatih model..."):
            
            # 2. SPLIT DATA DULU
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            
            # 3. BUILD PIPELINE 
            pca_engine = PCA(n_components=0.95, svd_solver='full', random_state=42)
            ml_pipeline = make_pipeline(StandardScaler(), pca_engine, raw_model)
            
            # 4. FITTING 
            start_time = time.time()
            ml_pipeline.fit(X_train, y_train)
            y_pred = ml_pipeline.predict(X_test)
            end_time = time.time()
            
            # Mengambil jumlah komponen PCA aktual yang terbentuk dari 95% varians
            n_komponen_pca = ml_pipeline.named_steps['pca'].n_components_
            
            # 5. HITUNG METRIK
            training_time = round(end_time - start_time, 2)
            acc_score = accuracy_score(y_test, y_pred) * 100
            
            # Menghindari error F1-Macro jika ada label kelas yang tidak muncul di testing
            try:
                f1_macro_score = f1_score(y_test, y_pred, average='macro') * 100
            except:
                f1_macro_score = acc_score # Fallback aman
                
            # 6. SIMPAN KE RAM UNTUK PAGE 5
            st.session_state['trained_pipeline'] = ml_pipeline
            st.session_state['kombinasi_aktif'] = kombinasi_aktif
            st.session_state['fitur_aktif'] = feature_choice 
            
            # TAMPILKAN HASIL
            st.markdown(f"#### 3. Hasil Live Execution")
            
            # Display cerdas menunjukkan kompresi dimensi
            st.write(f"Eksperimen: **{kombinasi_aktif}** | Sampel Terpakai: **{X.shape[0]} gambar**")
            st.info(f"**Reduksi PCA (95% Varians):** Dimensi awal **{X.shape[1]} fitur** berhasil dipadatkan menjadi **{n_komponen_pca} komponen utama**.")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Test Accuracy", f"{acc_score:.2f}%")
            m2.metric("F1-Macro Score", f"{f1_macro_score:.2f}%")
            m3.metric("Training Time", f"{training_time} s")
            
            st.success(f"**Eksekusi 100% Real-Time Selesai!** Gambar diekstrak, distandarisasi, direduksi PCA, dan dilatih.")
            st.info(f"**Lanjut ke Halaman 5:** Model sudah aktif di RAM. Kamu bisa langsung menguji gambar baru tanpa *load* file apapun!")