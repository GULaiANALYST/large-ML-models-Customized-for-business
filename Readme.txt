## 🎭 Facial Emotion Recognition using CNN

This project implements a Deep Learning model capable of identifying emotions from grayscale images of faces. It leverages the power of Convolutional Neural Networks to automatically extract facial features (like the curve of a lip or the furrow of a brow) to determine the user's mood.

## 🧠 Model Architecture

The model is a custom **Deep CNN** designed for high-performance image classification:

* **Four Convolutional Blocks:** * Each block contains a `Conv2D` layer with increasing filters (64, 128, 512, 512).
* `BatchNormalization` is used after every convolution to stabilize learning.
* `MaxPooling2D` reduces spatial dimensions to focus on the most important features.
* `Dropout` (25%) is applied to prevent the model from overfiting on the training data.


* **Fully Connected (Dense) Layers:**
* A `Flatten` layer converts the 2D feature maps into a 1D vector.
* Two Dense layers (256 and 512 neurons) perform the final reasoning.


* **Output Layer:**
* A final Dense layer with 7 neurons and a **Softmax** activation function to output probabilities for each emotion.



---

## 🛠️ Key Training Components

### 1. Data Augmentation

The script uses `ImageDataGenerator` to load images from directories. It converts images to **48x48 pixels** in **Grayscale** mode to ensure the model focuses on structural facial expressions rather than color or lighting noise.

### 2. Smart Callbacks

To ensure the best possible training results, the project uses three critical Keras callbacks:

* **ModelCheckpoint:** Automatically saves the "best" version of the model based on validation accuracy.
* **EarlyStopping:** Stops training if the validation loss stops improving, saving time and compute power.
* **ReduceLROnPlateau:** Lowers the learning rate when the model hits a plateau, allowing it to "fine-tune" its weights more carefully.

---

## 🚀 How to Use

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn tensorflow keras

```

### Dataset Structure

Ensure your images are organized as follows:

```text
images/
├── train/
│   ├── angry/
│   ├── disgust/
│   └── ... (other emotions)
└── validation/
    ├── angry/
    ├── disgust/
    └── ...

```

### Execution

1. **Configure Path:** Update the `folder_path` variable in the script to point to your dataset.
2. **Train:** Run the script. It will visualize sample images from the 'disgust' class and begin training for the specified number of epochs.
3. **Monitor:** After training, the script generates plots for **Loss** and **Accuracy** to help you visualize the model's performance.

---

## 📊 Results Visualization

The script produces a side-by-side comparison of:

* **Training vs. Validation Loss:** Helps identify if the model is learning effectively.
* **Training vs. Validation Accuracy:** Shows the percentage of correct predictions over time.

## Testing

For Testing use **untitled21.py** by simply entering the paths according to your system.  
