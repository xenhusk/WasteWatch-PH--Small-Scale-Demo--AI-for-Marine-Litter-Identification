# predict_waste_type.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sys # To exit gracefully if model not found

# --- Configuration (MUST match training script) ---
IMAGE_SIZE = (224, 224) # Image size the model expects
MODEL_PATH = 'wastewatch_model.h5' # Path to your saved model file

# IMPORTANT: These must be in the exact same order as they were during training.
# The train_wastewatch_model.py script will print the 'Class indices:'
# Make sure this list matches that order (alphabetical by folder name is default).
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

print(f"--- WasteWatch PH Prediction Tool ---")
print(f"Loading model from: {MODEL_PATH}")

# --- Load the trained model ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'train_wastewatch_model.py' has been run successfully and 'wastewatch_model.h5' exists in the current directory.")
    sys.exit(1) # Exit the script if model cannot be loaded

# --- Prediction Function ---
def predict_litter_type(img_path):
    """
    Loads an image, preprocesses it, and uses the model to predict its waste type.
    """
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at '{img_path}'")
        return

    try:
        # Load image and resize it to the target size expected by the model
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        # Convert the PIL image to a NumPy array
        img_array = image.img_to_array(img)
        # Expand dimensions to create a batch of 1 image (models expect batches)
        img_array = np.expand_dims(img_array, axis=0)
        # Normalize pixel values to 0-1, just like during training
        img_array /= 255.0

        # Make predictions
        predictions = model.predict(img_array)

        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(predictions[0])
        # Get the name of the predicted class
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        # Get the confidence score for the predicted class
        confidence = predictions[0][predicted_class_index] * 100

        print(f"\n--- Prediction for '{os.path.basename(img_path)}' ---")
        print(f"Predicted Type: {predicted_class_name}")
        print(f"Confidence: {confidence:.2f}%")

        # Optional: Print all class probabilities for more detailed output
        print("\nAll Class Probabilities:")
        for i, prob in enumerate(predictions[0]):
            print(f"  {CLASS_NAMES[i]}: {prob * 100:.2f}%")

    except Exception as e:
        print(f"An error occurred during prediction for '{img_path}': {e}")

# --- Example Usage ---
if __name__ == '__main__':
    # --- Instructions for testing ---
    print("\nTo test, place new images (that were NOT used for training) into a folder like 'test_images'.")
    print("Then, run this script.")
    print("----------------------------------------------------------------------------------")

    test_image_folder = 'test_images' # Folder to put your test images

    # Create the test_images folder if it doesn't exist
    if not os.path.exists(test_image_folder):
        os.makedirs(test_image_folder)
        print(f"\nCreated '{test_image_folder}' folder.")
        print(f"Please place images you want to test (e.g., a new plastic bottle image, a glass bottle image) into this folder.")
        print("Then run this script again.")
        sys.exit(0)

    # Get a list of all image files in the test folder
    image_files_to_test = [f for f in os.listdir(test_image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files_to_test:
        print(f"\nNo image files found in '{test_image_folder}'.")
        print("Please add some images to test (e.g., .png, .jpg, .jpeg files).")
    else:
        print(f"\nFound {len(image_files_to_test)} images in '{test_image_folder}' to test.")
        for img_file in image_files_to_test:
            full_img_path = os.path.join(test_image_folder, img_file)
            predict_litter_type(full_img_path)

    print("\n--- Prediction process complete. ---")