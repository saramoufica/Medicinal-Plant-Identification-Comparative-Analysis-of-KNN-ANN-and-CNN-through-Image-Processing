import os
import numpy as np
import pandas as pd
import threading
import tkinter as tk
from tkinter import Label, filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16  
from tensorflow.keras.preprocessing.image import img_to_array, load_img

knn_model = None
image_path = None
plant_name_to_index = {}
folder_path = "D:/Edatasets"  
excel_file = "Book1.xlsx"
sheet_name_plant_usage = 'sheet2'
plant_usage_df = pd.read_excel(os.path.join(folder_path, excel_file), sheet_name=sheet_name_plant_usage)
plant_name_usage_dict = dict(zip(plant_usage_df['plant_name'], plant_usage_df['usage']))
feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

def extract_features(image_path):
    image = load_img(image_path, target_size=(64, 64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize image
    features = feature_extractor.predict(image)
    return features.flatten()

def load_and_prepare_data():
    global plant_name_to_index    
    features = []
    labels = []    
    for plant_name in os.listdir(folder_path):
        plant_folder = os.path.join(folder_path, plant_name)
        if os.path.isdir(plant_folder):
            for image_name in os.listdir(plant_folder):
                image_path = os.path.join(plant_folder, image_name)
                if image_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    try:
                        feature_vector = extract_features(image_path)
                        features.append(feature_vector)
                        labels.append(plant_name)
                    except Exception as e:
                        print(f"Error processing {image_name}: {e}")    
    features = np.array(features)
    labels = np.array(labels)    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)    
    plant_name_to_index = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    return features, labels_encoded

def train_knn_model():
    global knn_model    
    features, labels_encoded = load_and_prepare_data()    
    X_train, X_val, y_train, y_val = train_test_split(features, labels_encoded, test_size=0.3, random_state=42)   
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)    
    y_pred = knn_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Trained KNN Model Accuracy: {accuracy * 100:.2f}%')
    messagebox.showinfo("Training Complete", f"KNN model trained with accuracy: {accuracy * 100:.2f}%")

def predict_new_image_knn(image_path):
    global knn_model
    if knn_model is None:
        raise ValueError("KNN model is not trained.")    
    features = extract_features(image_path).reshape(1, -1)    
    predicted_label_index = knn_model.predict(features)[0]
    predicted_label = [key for key, value in plant_name_to_index.items() if value == predicted_label_index][0]
    return predicted_label

def upload_image():
    global image_path, uploaded_image_label
    image_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
    )
    if image_path:
        img = Image.open(image_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        uploaded_image_label.config(image=img_tk)
        uploaded_image_label.image = img_tk
        uploaded_image_label.pack(pady=10)

def start_detection_knn():
    if not image_path:
        messagebox.showerror("Error", "Please upload an image first.")
        return

    try:
        predicted_label = predict_new_image_knn(image_path)
        search_column = 'plant_name'
        target_column = 'usage'
        result = plant_usage_df[plant_usage_df[search_column] == predicted_label]

        if not result.empty:
            plant_use = result[target_column].values[0]
            result_label_name.config(text=f"Plant Name: {predicted_label}")
            result_label_usage.config(text=f"Usage: {plant_use}")
        else:
            result_label_name.config(text="Plant Name: Not Found")
            result_label_usage.config(text="Usage: Not Found")

    except Exception as e:
        messagebox.showerror("Detection Error", str(e))

def start_training_thread_knn():
    training_thread = threading.Thread(target=train_knn_model)
    training_thread.start()
root = tk.Tk()
root.title("Herbal Insight Technology")
root.geometry("600x600")
header = tk.Label(root, text="Herbal Insight Technology", font=("Times New Roman", 25, "bold"))
header.pack(pady=20)
frame = tk.Frame(root)
frame.pack(pady=5)
upload_button = tk.Button(frame, text="Upload Image for Detection", command=upload_image, font=("Times New Roman", 14), bg='light green', fg='black')
upload_button.pack(pady=5)
uploaded_image_label = tk.Label(frame)
uploaded_image_label.pack(pady=5)
result_frame = tk.Frame(root)
result_frame.pack(pady=10)
detect_button = tk.Button(result_frame, text="Predict", command=start_detection_knn, font=("Times New Roman", 14), bg='light green', fg='black')
detect_button.pack(pady=5)
result_label_name = tk.Label(result_frame, text="Plant Name: ", font=("Times New Roman", 14, "bold"))
result_label_name.pack(pady=5)
result_label_usage = tk.Label(result_frame, text="Usage: ", font=("Times New Roman", 14, "bold"))
result_label_usage.pack(pady=5)
train_button = tk.Button(root, text="Train KNN Model", command=start_training_thread_knn, font=("Times New Roman", 14), bg='light green', fg='black')
train_button.pack(pady=20)
root.mainloop()
