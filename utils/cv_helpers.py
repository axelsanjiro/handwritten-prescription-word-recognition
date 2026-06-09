import cv2
import numpy as np
import math

def build_gabor_filter():
    # Membuat kernel Gabor (45 Derajat)
    return cv2.getGaborKernel((31, 31), 4.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

def radial_polynomial(n, m, r):
    # Matematika dasar untuk Zernike
    R = np.zeros_like(r, dtype=np.float64)
    for s in range(int((n - abs(m)) / 2) + 1):
        c = ((-1)**s * math.factorial(n - s)) / \
            (math.factorial(s) * math.factorial(int((n + abs(m)) / 2) - s) * math.factorial(int((n - abs(m)) / 2) - s))
        R += c * (r ** (n - 2 * s))
    return R

def calculate_zernike_heatmap(img_norm, radius=32, degree=8):
    # Membuat Activation Map untuk Zernike Moments
    x, y = np.arange(-radius, radius), np.arange(-radius, radius)
    X, Y = np.meshgrid(x, y)
    R, Theta = np.sqrt(X**2 + Y**2) / radius, np.arctan2(Y, X)
    
    mask = R <= 1.0
    R_mask, Theta_mask, img_mask = R[mask], Theta[mask], img_norm[mask]
    
    heatmap_canvas = np.zeros_like(R, dtype=np.float64)
    heatmap_mask = np.zeros_like(R_mask, dtype=np.float64)
    
    for n in range(degree + 1):
        for m in range(n + 1):
            if (n - m) % 2 == 0:
                R_nm = radial_polynomial(n, m, R_mask)
                V_nm_real = R_nm * np.cos(m * Theta_mask)
                V_nm_imag = R_nm * np.sin(m * Theta_mask)
                real_part = np.sum(img_mask * V_nm_real)
                imag_part = np.sum(img_mask * V_nm_imag)
                magnitude = ((n + 1) / np.pi) * np.sqrt(real_part**2 + imag_part**2)
                heatmap_mask += magnitude * np.abs(V_nm_real)
                
    heatmap_canvas[mask] = heatmap_mask
    heatmap_norm = (heatmap_canvas - np.min(heatmap_canvas)) / (np.max(heatmap_canvas) - np.min(heatmap_canvas) + 1e-8)
    return np.power(heatmap_norm, 0.8) 

def make_gradcam_style(feature_map, ksize=15):
    # Mengubah fitur kasar menjadi gumpalan halus (Grad-CAM Style)
    norm = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-8)
    blurred = cv2.GaussianBlur(norm, (ksize, ksize), 0)
    return (blurred - np.min(blurred)) / (np.max(blurred) - np.min(blurred) + 1e-8)