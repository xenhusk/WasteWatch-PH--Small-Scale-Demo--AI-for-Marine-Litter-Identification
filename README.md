# WasteWatch PH (Small Scale Demo) - AI for Marine Litter Identification

WasteWatch PH is a machine learning project that uses deep learning to identify and classify marine litter from images. This demo leverages transfer learning with MobileNetV2 to recognize different types of waste, helping support environmental cleanup and waste management efforts.

## Features
- Image classification for six waste categories: cardboard, glass, metal, paper, plastic, and trash
- Transfer learning using MobileNetV2 for efficient and accurate predictions
- Data augmentation for robust model training
- Training and prediction scripts included

## Project Structure
```
├── data/                # Training images (organized by category, not tracked in git)
├── test_images/         # Example/test images (not tracked in git)
├── train_wastewatch_model.py   # Script to train the model
├── predict_waste_type.py       # Script to predict waste type from an image
├── wastewatch_model.h5         # Trained model file (generated after training)
├── wastewatch_env/      # Python virtual environment (not tracked in git)
└── .gitignore           # Git ignore rules
```

## Getting Started

### 1. Clone the Repository
```powershell
git clone https://github.com/xenhusk/WasteWatch-PH--Small-Scale-Demo--AI-for-Marine-Litter-Identification
cd "WasteWatch PH (Small Scale Demo) AI for Marine Litter Identification"
```

### 2. Set Up Python Environment
It is recommended to use a virtual environment:
```powershell
python -m venv wastewatch_env
.\wastewatch_env\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install tensorflow matplotlib numpy
```

### 4. Prepare Data
- Place your training images in the `data/` folder, organized into subfolders named after each category (e.g., `data/cardboard/`, `data/glass/`, etc.).
- Place test images in the `test_images/` folder.

### 5. Train the Model
```powershell
python train_wastewatch_model.py
```

### 6. Predict Waste Type
```powershell
python predict_waste_type.py
```

## Notes
- The `data/`, `test_images/`, and `wastewatch_env/` folders are excluded from version control via `.gitignore`.
- The model is saved as `wastewatch_model.h5` after training.

## License
This project is for educational and demonstration purposes.

## Acknowledgements
- [TensorFlow](https://www.tensorflow.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
