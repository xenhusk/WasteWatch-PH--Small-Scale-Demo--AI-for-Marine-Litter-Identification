# WasteWatch PH (Small Scale Demo) - AI for Marine Litter Identification

WasteWatch PH is a machine learning project that uses deep learning to identify and classify marine litter from images. This demo leverages transfer learning with MobileNetV2 to recognize different types of waste, helping support environmental cleanup and waste management efforts.

## Features
- Image classification for six waste categories: cardboard, glass, metal, paper, plastic, and trash
- Transfer learning using MobileNetV2 for efficient and accurate predictions
- Data augmentation for robust model training
- Training and prediction scripts included
- **Optimized for GPU acceleration via WSL2 for faster training**

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

## Getting Started (Recommended: WSL2 with GPU)

This guide provides steps for setting up the project using **Windows Subsystem for Linux 2 (WSL2) with GPU acceleration**, which offers significantly faster training times than CPU-only setups.

### Prerequisites

1.  **Windows 10/11 with WSL2 Enabled:** Ensure WSL2 is installed and an Ubuntu distribution (e.g., Ubuntu 22.04 LTS or 24.04 LTS) is set up. Follow Microsoft's official guide: [https://learn.microsoft.com/en-us/windows/wsl/install](https://learn.microsoft.com/en-us/windows/wsl/install)
2.  **NVIDIA GPU with Compatible Drivers:** Ensure you have an NVIDIA GPU and the latest recommended drivers for WSL from NVIDIA.
    * Check for updates: `wsl --update` in PowerShell.
    * Download NVIDIA WSL drivers: [https://developer.nvidia.com/cuda/wsl](https://developer.nvidia.com/cuda/wsl)

### 1. Clone the Repository (into WSL Linux Filesystem)

It is **highly recommended** to clone this repository directly into your WSL Ubuntu's Linux filesystem (e.g., in your home directory `~` or `~/projects`) for optimal performance. Working on files located on the Windows drive (`/mnt/c/`) from within WSL2 can be significantly slower.

```bash
# From your WSL2 Ubuntu terminal:
cd ~ # Navigate to your home directory (or create a 'projects' folder: mkdir projects && cd projects)
git clone https://github.com/xenhusk/WasteWatch-PH--Small-Scale-Demo--AI-for-Marine-Litter-Identification
cd "WasteWatch PH (Small Scale Demo) AI for Marine Litter Identification"
```

### 2. Set Up Python Environment

You need Python 3.12.x for TensorFlow 2.19.0. Ubuntu usually comes with `python3`. If `python3 -m venv` fails, you might need to install `python3.12-venv`.

```bash
# From your project directory in WSL2 Ubuntu
# Check your Python version (must be 3.12.x for TF 2.19.0 compatibility)
python3 --version

# If the virtual environment module is missing (e.g., "ensurepip is not available" error)
# You need to install the venv package for your specific Python 3.12 version:
sudo apt update
sudo apt install python3.12-venv # Use 'sudo' as this is a system package installation

# Create a Python virtual environment
python3 -m venv wastewatch_env

# Activate the virtual environment
source wastewatch_env/bin/activate
# Your prompt should change to show '(wastewatch_env)'
```

### 3. Install NVIDIA CUDA Toolkit 12.5

TensorFlow 2.19.0 specifically requires **CUDA Toolkit 12.5**. If you have a different version installed (e.g., 12.9), you *must* uninstall it first and then install 12.5.

#### a. Uninstall any existing incompatible CUDA Toolkit (if applicable)

```bash
# Deactivate virtual environment temporarily if active
deactivate

# Replace '12-9' with whatever version you have if it's not 12.5 (e.g., '12-x')
sudo apt --purge remove "cuda-toolkit-12-9*" "cuda-repo-wsl-ubuntu-12-9-local" "cuda-wsl-ubuntu-keyring"
sudo apt autoremove -y
sudo apt clean
sudo rm -rf /usr/local/cuda-12.9 # Remove residual files

# Reactivate your virtual environment after system changes
source wastewatch_env/bin/activate # Only if you deactivated it
```

#### b. Install CUDA Toolkit 12.5

1.  **Go to NVIDIA CUDA Toolkit Archives:** [https://developer.nvidia.com/cuda-downloads/archives](https://www.google.com/search?q=https://developer.nvidia.com/cuda-downloads/archives)

2.  **Select "CUDA Toolkit 12.5".**

3.  **Choose the following options on the 12.5 download page:**

      * Operating System: Linux
      * Architecture: x86_64
      * Distribution: WSL-Ubuntu
      * Version: Your Ubuntu version (e.g., 22.04 or 24.04)
      * Installer Type: deb (local)

4.  **Carefully copy and paste the EXACT installation commands provided on *that specific NVIDIA 12.5 download page* into your WSL2 terminal.** They will typically look like this:

    ```bash
    # (Example commands - copy from NVIDIA's website for accuracy!)
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda-repo-wsl-ubuntu-12-5-local_12.5.0-1_amd64.deb # Adjust URL if needed
    sudo dpkg -i cuda-repo-wsl-ubuntu-12-5-local_12.5.0-1_amd64.deb # Adjust filename
    sudo cp /var/cuda-repo-wsl-ubuntu-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt update
    sudo apt install cuda-toolkit-12-5 # IMPORTANT: Install 'cuda-toolkit-12-5'
    ```

5.  **Set Environment Variables for CUDA 12.5 in `~/.bashrc`:**

    ```bash
    nano ~/.bashrc
    # Add these lines at the end:
    export PATH=/usr/local/cuda-12.5/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH
    # Save (Ctrl+O, Enter, Ctrl+X)
    ```

6.  **Reload Bash environment and re-enter WSL:**

    ```bash
    source ~/.bashrc # In current shell
    exit             # Exit WSL
    wsl              # Re-enter WSL
    cd "WasteWatch PH (Small Scale Demo) AI for Marine Litter Identification" # Navigate back
    source wastewatch_env/bin/activate # Activate venv
    ```

7.  **Verify CUDA 12.5 installation:**

    ```bash
    nvcc --version # Should now show 'release 12.5'
    ```

### 4. Install NVIDIA cuDNN 9.3

TensorFlow 2.19.0 requires **cuDNN 9.3** to function with CUDA 12.5.

1.  **Go to NVIDIA cuDNN Downloads:** [https://developer.nvidia.com/cudnn/downloads](https://www.google.com/search?q=https://developer.nvidia.com/cudnn/downloads)

2.  **Log in to your NVIDIA Developer account** (create one if needed).

3.  **Accept the terms.**

4.  **Download "cuDNN v9.3.x for CUDA 12.5"** (specifically the "cuDNN Library for Linux (x86_64)" version - it will be a `.tar.xz` or `.tgz` file).

5.  **Transfer the downloaded file to your WSL2 Ubuntu home directory.** (Easiest: drag-and-drop from Windows File Explorer into your WSL terminal).

6.  **Extract the cuDNN archive:**

    ```bash
    tar -xvf cudnn-linux-x86_64-9.3.x.x_cuda12.5-archive.tar.xz # Adjust filename exactly
    ```

    This will create a `cuda/` directory.

7.  **Copy cuDNN files to your CUDA 12.5 installation:**

    ```bash
    sudo cp cuda/include/* /usr/local/cuda-12.5/include/
    sudo cp cuda/lib/* /usr/local/cuda-12.5/lib64/
    sudo chmod a+r /usr/local/cuda-12.5/lib64/*
    ```

### 5. Install Python Dependencies

With CUDA and cuDNN correctly installed, you can now install TensorFlow and other project dependencies into your active virtual environment.

```bash
# Ensure your virtual environment is active: (wastewatch_env) in your prompt
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 6. Verify GPU Detection by TensorFlow

```bash
# While in your project directory and virtual environment
python -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('Built with CUDA:', tf.test.is_built_with_cuda())"
```

You should see `Num GPUs Available: 1` (or more) and `Built with CUDA: True`.

### 7. Prepare Data

  - Place your training images in the `data/` folder, organized into subfolders named after each category (e.g., `data/cardboard/`, `data/glass/`, etc.).
  - Place test images in the `test_images/` folder.

### 8. Train the Model

```bash
python train_wastewatch_model.py
```

### 9. Predict Waste Type

```bash
python predict_waste_type.py
```

## YOLO Object Detection (Realtime Waste Detection)

This project now supports object detection using YOLOv8 for real-time waste detection via your webcam!

### How to Train YOLOv8

1. Prepare your dataset (images and YOLO-format labels) in the `yolo_dataset/` folder (already organized by the provided scripts).
2. Install Ultralytics YOLO:
   ```bash
   pip install ultralytics
   ```
3. Use the provided `data.yaml` for training:
   ```bash
   yolo detect train data=data.yaml model=yolov8m.pt epochs=50 imgsz=640 batch=32
   ```

### How to Run Real-Time Detection

After training, run this command to use your webcam for live detection:
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=0
```
- Replace `source=0` with a video file or image folder as needed.

### Outputs
- Training runs and results are saved in the `runs/` directory (ignored by git).
- Model weights are saved as `.pt` files in `runs/detect/train/weights/`.

## Dataset

The training and validation images used in this project are not included in the repository. You can download the original dataset from Kaggle:

[https://www.kaggle.com/datasets/harshpanwar/aquatrash/data](https://www.kaggle.com/datasets/harshpanwar/aquatrash/data)

After downloading, follow the instructions above to organize and preprocess the data for YOLO training.

## .gitignore
- The repository ignores all data, test images, virtual environments, YOLO outputs, model weights, and cache files for cleanliness and privacy.

## Troubleshooting Common Issues

  * **`Command 'python' not found, did you mean: command 'python3'`**: Use `python3` instead of `python` for system-level commands like `python3 -m venv`. Once the virtual environment is activated, `python` will correctly point to the virtual environment's Python.
  * **`E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)`**: You need administrative privileges for `apt` commands. Prefix the command with `sudo` (e.g., `sudo apt install python3.12-venv`).
  * **`wastewatch_env/bin/activate: No such file or directory`**: The virtual environment might not have been created correctly or was corrupted/deleted. Follow the steps to `rm -rf wastewatch_env` and then `python3 -m venv wastewatch_env` again.
  * **`Cannot dlopen some GPU libraries` / `Skipping registering GPU devices`**: This indicates a mismatch between your TensorFlow version's CUDA/cuDNN requirements and what's installed on your system, or incorrect `LD_LIBRARY_PATH`. Re-verify that CUDA Toolkit 12.5 and cuDNN 9.3 are installed and their paths are correctly added to `~/.bashrc`.
  * **`ptxas warning: Registers are spilled to local memory`**: These are warnings during GPU kernel compilation. They are generally harmless and do not prevent GPU usage. You can usually ignore them.
  * **`UserWarning: Your PyDataset class should call super().__init__(**kwargs)`**: This is a warning from Keras about your data loading pipeline. It's a best practice recommendation for more robust data handling and not directly a GPU issue.

## Notes

  - The `data/`, `test_images/`, and `wastewatch_env/` folders are excluded from version control via `.gitignore`.
  - The model is saved as `wastewatch_model.h5` after training.
  - If using VS Code, install the "WSL" extension to open your project directly from the WSL Linux filesystem for the best performance.

## License

This project is for educational and demonstration purposes.

## Acknowledgements

  - [TensorFlow](https://www.tensorflow.org/)
  - [MobileNetV2](https://arxiv.org/abs/1801.04381)

<!-- end list -->
