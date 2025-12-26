import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. DEFINE YOUR PATHS AND PARAMETERS ---

# This is the new test directory you just created
TEST_DATA_DIR = r'D:\Projects\Agri Detect\PlantVillage-Dataset-master\plant_dataset_split\test'

# Update this to the location of your saved model
MODEL_PATH = r'path\to\your_model_name.h5'  # <--- ⚠️ UPDATE THIS

# Use the same image dimensions you used for training
IMG_SIZE = (224, 224) # <--- ⚠️ UPDATE THIS if it's different
BATCH_SIZE = 32

# --- 2. LOAD THE MODEL ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Stop here if the model didn't load
    raise

# --- 3. SET UP THE TEST DATA GENERATOR ---
# IMPORTANT: Only rescale. No augmentation!
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # <-- IMPORTANT: Must be False for evaluation
)

# --- 4. MAKE PREDICTIONS ---
print("Making predictions on the test set...")
Y_pred = model.predict(test_generator)

# --- 5. GET TRUE AND PREDICTED LABELS ---
y_pred = np.argmax(Y_pred, axis=1) # Get the index of the highest probability
y_true = test_generator.classes     # Get the true class indices
class_labels = list(test_generator.class_indices.keys()) # Get the class names

print(f"Found {len(class_labels)} classes.")