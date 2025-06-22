# train_wastewatch_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
# Set the input image size for the model. MobileNetV2 expects 224x224.
IMAGE_SIZE = (224, 224)
# Batch size for training. Adjust based on your computer's RAM. Smaller for less RAM.
BATCH_SIZE = 32
# Number of training iterations. For a demo, start low (e.g., 5-10).
# More epochs can improve accuracy but take longer and risk overfitting.
EPOCHS = 10
# Path to your main data directory (where cardboard, glass, etc., folders are).
DATA_DIR = 'data'
# Path to save your trained model.
MODEL_SAVE_PATH = 'wastewatch_model.h5'

# Define your categories. These MUST EXACTLY MATCH your folder names in the 'data' directory.
# The order here doesn't strictly matter for ImageDataGenerator, but consistency is good.
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

print(f"--- Starting WasteWatch PH Model Training ---")
print(f"Data Directory: {DATA_DIR}")
print(f"Categories: {CLASS_NAMES}")
print(f"Image Size: {IMAGE_SIZE}")
print(f"Epochs: {EPOCHS}")

# --- 1. Prepare Data Generators ---
# ImageDataGenerator helps load images from directories and perform data augmentation.
# Data augmentation creates variations of your images (rotations, flips, zooms)
# to make the model more robust and prevent overfitting, especially with limited data.
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to 0-1 (important for neural networks)
    rotation_range=20, # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2, # Randomly shift images horizontally
    height_shift_range=0.2, # Randomly shift images vertically
    shear_range=0.2, # Apply shearing transformations
    zoom_range=0.2, # Apply random zooms
    horizontal_flip=True, # Randomly flip images horizontally
    fill_mode='nearest', # Strategy for filling in new pixels after transformations
    validation_split=0.2 # Use 20% of your data for validation during training
)

# Generator for training data
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Use 'categorical' because you have multiple classes (one-hot encoding)
    subset='training', # Specify this is for the training set
    classes=CLASS_NAMES, # Explicitly pass your class names to ensure mapping consistency
    shuffle=True # Shuffle data for better training
)

# Generator for validation data (used to evaluate model performance on unseen data during training)
validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation', # Specify this is for the validation set
    classes=CLASS_NAMES,
    shuffle=False # No need to shuffle validation data
)

# Verify the number of images found and their mapping
print(f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes.")
print(f"Found {validation_generator.samples} validation images belonging to {validation_generator.num_classes} classes.")
print(f"Class indices: {train_generator.class_indices}") # Shows how categories are mapped to numbers

# --- 2. Load Pre-trained Base Model (MobileNetV2) ---
# MobileNetV2 is a good choice because it's relatively light and fast, but still powerful.
# 'weights='imagenet'' loads weights pre-trained on a massive dataset (ImageNet).
# 'include_top=False' means we remove the original classification head of MobileNetV2,
# as we'll add our own head tailored to our specific waste categories.
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Freeze the layers of the base model.
# This prevents their weights from being updated during training. We only want to train
# our new classification layers, leveraging MobileNetV2's learned features.
base_model.trainable = False

# --- 3. Build the Custom Classification Head ---
# Add new layers on top of the base model to classify our specific waste types.
x = base_model.output
x = GlobalAveragePooling2D()(x) # Averages the features across spatial dimensions, reducing dimensionality
x = Dense(128, activation='relu')(x) # A fully connected layer with ReLU activation
# The final output layer with 'softmax' activation for multi-class classification.
# The number of units must equal the number of your categories.
predictions = Dense(len(CLASS_NAMES), activation='softmax')(x)

# Create the final model by combining the base model and our new head
model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. Compile the Model ---
# Configure the model for training.
model.compile(
    optimizer='adam', # Adam is a popular and effective optimizer
    loss='categorical_crossentropy', # Appropriate loss function for multi-class, one-hot encoded labels
    metrics=['accuracy'] # We want to track accuracy during training
)

model.summary() # Print a summary of the model's architecture

# --- 5. Train the Model ---
print(f"\n--- Training Model for {EPOCHS} epochs ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        # Early stopping helps prevent overfitting by stopping training if validation loss
        # doesn't improve for 'patience' number of epochs. It restores the best weights found.
    ]
)

# --- 6. Save the Model ---
model.save(MODEL_SAVE_PATH)
print(f"\n--- Model saved successfully to {MODEL_SAVE_PATH} ---")

# --- Optional: Plot Training History ---
# Visualize training progress (accuracy and loss over epochs)
print("\n--- Plotting Training History ---")
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history_plot.png')
plt.close()

print("\n--- Training complete. ---")