import   tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os # Added os for path joining

# ==================================================================
#                       1. DEFINE PATHS AND PARAMETERS
# ==================================================================
MODEL_PATH = r'plant_disease_model.h5'
BASE_DIR = r'D:\Projects\Agri Detect\PlantVillage-Dataset-master\plant_dataset_split'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')

IMG_SIZE = (224, 224) # Use the same size you trained with
BATCH_SIZE = 32
INITIAL_EPOCHS = 10  # The number of epochs you used for the first training

# ==================================================================
#                       2. SET UP DATA GENERATORS
# ==================================================================

# Add augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# NO augmentation for validation, just rescale
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# ==================================================================
#                       3. LOAD AND UNFREEZE MODEL
# ==================================================================
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# We will fine-tune from 'block_13_expand' onwards.
# This means we will freeze all layers before it.
fine_tune_at_layer = 'block_13_expand'

# Unfreeze all layers first
model.trainable = True

# Find the index of the layer to fine-tune from
try:
    fine_tune_at_index = [i for i, layer in enumerate(model.layers) if layer.name == fine_tune_at_layer][0]
except IndexError:
    print(f"Error: Layer '{fine_tune_at_layer}' not found in model.")
    print("Please check the name against the summary.")
    raise

# Freeze all layers *before* that index
print(f"Freezing all layers up to layer #{fine_tune_at_index} ('{fine_tune_at_layer}')...")
for layer in model.layers[:fine_tune_at_index]:
    layer.trainable = False

print("Model unfreezing complete.")

# ==================================================================
#                       4. COMPILE WITH LOW LEARNING RATE
# ==================================================================
# This is the most important part of fine-tuning.
# We use a *very* low learning rate so we don't destroy
# the pre-trained weights.

print("Re-compiling model with a low learning rate...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # 0.00001
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print the new trainable status
model.summary()

# ==================================================================
#                       5. CONTINUE TRAINING (FINE-TUNE)
# ==================================================================
# We will train for 10 more epochs
fine_tune_epochs = 10
total_epochs = INITIAL_EPOCHS + fine_tune_epochs

# Add a callback to reduce the learning rate if it plateaus
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', 
                                 factor=0.2,
                                 patience=2,
                                 min_lr=1e-7)

print(f"Starting fine-tuning for {fine_tune_epochs} epochs...")
history_fine_tune = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=INITIAL_EPOCHS, # This tells Keras to start at epoch 10
    validation_data=validation_generator,
    callbacks=[lr_scheduler]
)

# ==================================================================
#                       6. SAVE YOUR NEW MODEL
# ==================================================================
NEW_MODEL_PATH = r'plant_disease_model_finetuned.h5'
model.save(NEW_MODEL_PATH)
print(f"Fine-tuning complete! New model saved to: {NEW_MODEL_PATH}")