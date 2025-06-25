# predict_waste_type.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sys # To exit gracefully if model not found
import cv2  # Add this import at the top

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

def predict_litter_type_from_array(img_array):
    """
    Accepts a NumPy array (BGR image from OpenCV), preprocesses, and predicts waste type.
    """
    try:
        # Convert BGR (OpenCV) to RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        # Resize to model input size
        img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
        # Convert to float32 and normalize
        img_resized = img_resized.astype(np.float32) / 255.0
        # Expand dims for batch
        img_batch = np.expand_dims(img_resized, axis=0)
        # Predict
        predictions = model.predict(img_batch)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100
        return predicted_class_name, confidence, predictions[0]
    except Exception as e:
        print(f"Error during camera prediction: {e}")
        return None, None, None

# --- Example Usage ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='WasteWatch PH Prediction Tool')
    parser.add_argument('--camera', action='store_true', help='Use live camera feed for prediction')
    args = parser.parse_args()

    if args.camera:
        print("\nScanning for available cameras (indices 0-9)...")
        available_cams = []
        for cam_idx in range(10):
            temp_cap = cv2.VideoCapture(cam_idx)
            if temp_cap.isOpened():
                available_cams.append(cam_idx)
                temp_cap.release()
        if not available_cams:
            print("No available cameras found (indices 0-9). Exiting.")
            sys.exit(1)
        print(f"Available camera indices: {available_cams}")
        chosen_idx = None
        while chosen_idx not in available_cams:
            try:
                chosen_idx = int(input(f"Select camera index from {available_cams}: "))
            except ValueError:
                print("Invalid input. Please enter a valid camera index.")
        cap = cv2.VideoCapture(chosen_idx)
        print(f"Using camera index {chosen_idx}.")
        if not cap.isOpened():
            print(f"Error: Could not open webcam at index {chosen_idx}.")
            sys.exit(1)
        print("\nStarting live camera prediction mode. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            pred_class, conf, all_probs = predict_litter_type_from_array(frame)
            label = f"{pred_class if pred_class else 'Error'}: {conf:.1f}%" if conf else 'Prediction Error'
            # Overlay label
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            # Draw a rectangle in the center of the frame as a visual cue
            h, w, _ = frame.shape
            box_w, box_h = int(w * 0.5), int(h * 0.5)
            x1 = (w - box_w) // 2
            y1 = (h - box_h) // 2
            x2 = x1 + box_w
            y2 = y1 + box_h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Overlay label near the rectangle
            cv2.putText(frame, label, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow('WasteWatch Live Prediction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        print("\n--- Camera prediction session ended. ---")
        sys.exit(0)

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