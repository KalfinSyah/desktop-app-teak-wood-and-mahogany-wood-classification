import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from skimage.feature import graycomatrix, graycoprops


def klasifikasikan(img_path):
    model = tf.keras.models.load_model('wood_texture_classifier_model.keras')
    
    img = cv2.imread(img_path)
    if img is None:
        return "Error: Image tidak bisa di load"
    

    # resize image
    img_resized = cv2.resize(img, (224, 224))
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray = (gray / 32).astype(np.uint8)
    

    
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, levels=8, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    ASM = graycoprops(glcm, 'ASM').mean()
    
    glcm_features = np.array([contrast, dissimilarity, homogeneity, energy, correlation, ASM])
    glcm_features = np.expand_dims(glcm_features, axis=0)
    
    # normalisasi data image
    img_normalized = img_resized / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=0)
    
    # prediksi
    prediction = model.predict([img_normalized, glcm_features])
    class_idx = np.argmax(prediction, axis=1)[0]
    
    wood_types = ['Jati (Teak)', 'Mahoni (Mahogany)']
    probability = prediction[0][class_idx] * 100
    
    return f"Hasil klasifikasi: {wood_types[class_idx]}\nTingkat kepercayaan: {probability:.2f}%"


def select_file():
    selected_image = filedialog.askopenfilename(
        filetypes=[("JPEG files", "*.jpg"), ("JPEG files", "*.jpeg")]
    )

    if selected_image:
        if selected_image.lower().endswith(('.jpg', '.jpeg')):
            messagebox.showinfo("Classification Result", klasifikasikan(selected_image))
        else:
            messagebox.showerror("ERROR", "Only .jpg/.jpeg files are allowed!")
    else:
        messagebox.showerror("ERROR", "No file selected")

root = tk.Tk()
root.title("Teak Wood (Jati) And Mahogany Wood (Mahoni) Classifier")

label = tk.Label(root, text="Select .jpg image to classify")
label.pack(padx=10, pady=10)

select_button = tk.Button(root, text="Select", command=select_file)
select_button.pack(padx=10, pady=10)

root.mainloop()
