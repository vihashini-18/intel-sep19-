import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Define class labels
labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
num_classes = len(labels)

# Build CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create model instance
model = create_model()

# ------------------------------
# TRAINING ON RANDOM DATA FOR DEMO
# (Replace this with real data for real predictions)
# ------------------------------
X_train = np.random.rand(100, 150, 150, 3)  # 100 random images
y_train = np.random.randint(0, num_classes, 100)
y_train = to_categorical(y_train, num_classes)

model.fit(X_train, y_train, epochs=3, batch_size=10, verbose=1)

# Prediction function
def predict_image(img):
    img = img.resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)
    pred_class = labels[np.argmax(preds)]
    confidence = float(np.max(preds) * 100)
    
    return f"{pred_class} ({confidence:.2f}%)"

# Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Intel Image Classification (Demo)",
    description="Upload an image and get a predicted class with confidence."
)

if __name__ == "__main__":
    iface.launch()
