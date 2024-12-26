import os
import numpy as np
import pandas as pd
import threading
import tkinter as tk
from tkinter import Label, filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
import tensorflow as tf

tf.get_logger().setLevel('ERROR')



# Initialize global variables
cnn_model = None
image_path = None
plant_name_to_index = {}
folder_path = "D:/Edatasets"  # Dataset directory

# Load the Excel file for plant use details
excel_file = "Book1.xlsx"
sheet_name_plant_usage = 'sheet2'
plant_usage_df = pd.read_excel(os.path.join(folder_path, excel_file), sheet_name=sheet_name_plant_usage)

# Create a dictionary to map plant names to their usage
plant_name_usage_dict = dict(zip(plant_usage_df['plant_name'], plant_usage_df['usage']))

class AccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Print accuracy at the end of every epoch
        accuracy = logs.get('accuracy', 0)
        print(f"Epoch {epoch + 1}: Accuracy: {accuracy * 100:.2f}%")

def preprocess_image(image_path):
    """Preprocess image for VGG16 model."""
    image = load_img(image_path, target_size=(224, 224))  # Resize to 224x224 for VGG16
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize image
    return image

def load_and_prepare_data():
    global plant_name_to_index

    # Create empty lists to hold the images and labels
    features = []
    labels = []

    # Load the VGG16 model without the top layers (for feature extraction)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Iterate over the dataset and load each image
    for plant_name in os.listdir(folder_path):
        plant_folder = os.path.join(folder_path, plant_name)
        if os.path.isdir(plant_folder):
            for image_name in os.listdir(plant_folder):
                image_path = os.path.join(plant_folder, image_name)
                if image_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    try:
                        image = preprocess_image(image_path)
                        feature = base_model.predict(image)  # Extract features using VGG16
                        features.append(feature)
                        labels.append(plant_name)
                    except Exception as e:
                        print(f"Error processing {image_name}: {e}")

    # Convert lists to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Reshape features to match the input for the custom layers
    features = features.reshape(features.shape[0], -1)

    # Encode plant names to numerical labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Create a dictionary to map plant index to name
    plant_name_to_index = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # One-hot encode the labels
    labels_encoded = to_categorical(labels_encoded)

    return features, labels_encoded

def build_model(input_shape, num_classes):
    """Build and compile a model using extracted features."""
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))  # Fully connected layer
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(num_classes, activation='softmax'))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_cnn_model():
    global cnn_model

    # Load and prepare data
    features, labels_encoded = load_and_prepare_data()

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, labels_encoded, test_size=0.3, random_state=42)

    # Build and train the model
    cnn_model = build_model(X_train.shape[1:], y_train.shape[1])
    
    # Custom accuracy callback to print accuracy after every epoch
    accuracy_callback = AccuracyCallback()

    # Train the model with verbose=0 to suppress step output
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    # Evaluate the model without step output
    loss, accuracy = cnn_model.evaluate(X_val, y_val, verbose=0)
    print(f'Trained CNN Model Accuracy: {accuracy * 100:.2f}%')
    messagebox.showinfo("Training Complete", f"CNN model trained with accuracy: {accuracy * 100:.2f}%")

def predict_new_image_cnn(image_path):
    global cnn_model
    if cnn_model is None:
        raise ValueError("CNN model is not trained.")

    # Preprocess the uploaded image
    image = preprocess_image(image_path)

    # Extract features using VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    feature = base_model.predict(image).reshape(1, -1)  # Match input shape for the custom layers

    # Predict the plant name using the CNN model
    predictions = cnn_model.predict(feature)
    predicted_label_index = np.argmax(predictions[0])
    predicted_label = [key for key, value in plant_name_to_index.items() if value == predicted_label_index][0]
    
    return predicted_label

def upload_image():
    global image_path, uploaded_image_label
    image_path = filedialog.askopenfilename(
        filetypes=[("Image files", ".jpg;.jpeg;.png;.bmp;*.tiff")]
    )
    if image_path:
        img = Image.open(image_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        uploaded_image_label.config(image=img_tk)
        uploaded_image_label.image = img_tk
        uploaded_image_label.pack(pady=10)

def start_detection_cnn():
    if not image_path:
        messagebox.showerror("Error", "Please upload an image first.")
        return

    try:
        predicted_label = predict_new_image_cnn(image_path)
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

def start_training_thread_cnn():
    training_thread = threading.Thread(target=train_cnn_model)
    training_thread.start()

# Initialize the main window
root = tk.Tk()
root.title("Herbal Insight Technology")
root.geometry("600x600")

# Create a header label
header = tk.Label(root, text="Herbal Insight Technology", font=("Times New Roman", 25, "bold"))
header.pack(pady=20)

# Frame for uploading image and showing it
frame = tk.Frame(root)
frame.pack(pady=5)

upload_button = tk.Button(frame, text="Upload Image for Detection", command=upload_image, font=("Times New Roman", 14), bg='light green', fg='black')
upload_button.pack(pady=5)

uploaded_image_label = tk.Label(frame)
uploaded_image_label.pack(pady=5)

result_frame = tk.Frame(root)
result_frame.pack(pady=10)

detect_button = tk.Button(result_frame, text="Predict", command=start_detection_cnn, font=("Times New Roman", 14), bg='light green', fg='black')
detect_button.pack(pady=5)

result_label_name = tk.Label(result_frame, text="Plant Name: ", font=("Times New Roman", 14, "bold"))
result_label_name.pack(pady=5)

result_label_usage = tk.Label(result_frame, text="Usage: ", font=("Times New Roman", 14, "bold"))
result_label_usage.pack(pady=5)

train_button = tk.Button(root, text="Train CNN Model", command=start_training_thread_cnn, font=("Times New Roman", 14), bg='light green', fg='black')
train_button.pack(pady=20)

root.mainloop()
