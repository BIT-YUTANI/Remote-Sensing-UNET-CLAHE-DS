# %%
# ===================================================================
# Block 1: Imports, Configuration, and GPU Setup
# ===================================================================
#
# Description:
# This block handles all necessary imports and global configurations for the project.
# It includes standard libraries for data handling and deep learning, adds OpenCV
# for advanced image processing, and sets up project-specific parameters like
# file paths, model dimensions, and training hyperparameters.
#
# ===================================================================

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Add, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.utils import load_img, img_to_array

# NEW: Import OpenCV for CLAHE preprocessing and morphological post-processing
import cv2

# --- Project Configuration ---
SEED = 1998
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Paths and Dataset Configuration ---
DATASET_ROOT_DIR = Path('/kaggle/input/satellitechangedetectiondataset/ChangeDetectionDataset')
TRAINING_DATASETS = ['Model/with_shift', 'Model/without_shift', 'Real/subset']
TEST_DATASET_PATH = DATASET_ROOT_DIR / 'Real/subset'

IMAGE_SUBFOLDER = 'A'
CHANGED_SUBFOLDER = 'B'
MASK_SUBFOLDER = 'OUT'
ALLOWED_EXTS = {'.png', '.jpg', '.jpeg', 'tif', '.tiff', '.bmp'}

# --- Model and Image Dimensions ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
INPUT_CHANNELS = 6  # 3 for image A + 3 for image B
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS)
NUM_CLASSES = 1

# --- Training Hyperparameters ---
EPOCHS = 70
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DEEP_SUPERVISION = True
AUTOTUNE = tf.data.AUTOTUNE

# NEW: Deep supervision weights aligned with the paper's methodology.
DEEP_SUPERVISION_WEIGHTS = [1.13, 1.12, 1.112, 1.13, 1.13]

# --- Evaluation & Visualization Parameters ---
THRESH = 0.5
NUM_VIS_SAMPLES = 6

# --- GPU Configuration ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"✅ Enabled memory growth on {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"⚠️ Could not set memory growth: {e}")
else:
    print(" Bypassing GPU setup. No GPU detected; running on CPU.")

print("\n--- Block 1: Setup and Configuration complete. ---")


# %%
# ===================================================================
# Block 2: Data Pipeline, Preprocessing, and Dynamic Loss Function
# ===================================================================
#
# Description:
# This block defines the core data processing and loss functions. Key improvements include:
# 1. 'apply_clahe_cv': A new function to enhance image contrast, helping the model
#    discern features in varying lighting conditions.
# 2. 'dynamic_bce_dice_loss': The corrected loss function that dynamically calculates
#    class weights per batch to better handle class imbalance, as per the paper.
# 3. An updated data pipeline that integrates these enhancements and handles various image formats.
#
# ===================================================================

# --- Custom Loss Functions (Aligned with Paper) ---

def dice_coef(y_true, y_pred, smooth=1):
    """Computes the Dice coefficient, a measure of overlap between two samples."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """Computes the Dice loss, which is 1 - Dice coefficient."""
    return 1 - dice_coef(y_true, y_pred)

def dynamic_bce_dice_loss(y_true, y_pred):
    """
    Calculates a combined loss of dynamically weighted Binary Cross-Entropy (BCE) and Dice loss.
    This function dynamically calculates the balancing weight 'beta' for each batch based on the
    ratio of changed to unchanged pixels, as described in the paper. This is a critical
    change aimed at improving precision by reducing false positives.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Dynamically calculate beta per batch. beta = |Y-| / (|Y+| + |Y-|)
    # This weight is applied to the positive class (changed pixels).
    beta = tf.reduce_mean(1.0 - y_true)
    
    # Calculate weighted BCE
    bce = K.binary_crossentropy(y_true, y_pred)
    # Apply weight 'beta' to the positive class (y_true=1) and '1-beta' to the negative class (y_true=0)
    weighted_bce = K.mean(bce * beta * y_true + bce * (1.0 - beta) * (1.0 - y_true))

    # The paper combines weighted BCE with Dice loss using a lambda of 0.5.
    dice_loss = dice_coef_loss(y_true, y_pred)
    return weighted_bce + 0.559 * dice_loss


# --- Image Quality Enhancement & Data Loading Pipeline ---

# ✨ FIX: THIS FUNCTION IS MODIFIED TO WORK WITH BOTH TENSORS AND NUMPY ARRAYS ✨
def apply_clahe_cv(image):
    """
    Applies CLAHE using OpenCV to a single image (which can be a Tensor or a NumPy array).
    """
    # If the input is a TensorFlow Tensor, convert it. Otherwise, assume it's a NumPy array.
    if hasattr(image, 'numpy'):
        image_np = image.numpy()
    else:
        image_np = image
    
    # Ensure the image is uint8 for OpenCV operations
    image_np = image_np.astype(np.uint8)

    # Convert image from RGB to LAB color space to apply CLAHE only on the luminance channel
    lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(lab_image)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE-enhanced L-channel back with the original A and B channels
    limg = cv2.merge((cl, a, b))
    # Convert back to RGB color space
    final_image = cv2.cvtColor(limg, cv2.COLOR_Lab2RGB)
    return final_image.astype(np.uint8)

@tf.function
def tf_apply_clahe(image):
    """TensorFlow wrapper to integrate the OpenCV CLAHE function into the data pipeline."""
    im_shape = image.shape
    # tf.py_function wraps a Python function for use in a TensorFlow graph
    [image,] = tf.py_function(apply_clahe_cv, [image], [tf.uint8])
    image.set_shape(im_shape)
    return image

def _read_image_tf(path, channels=3):
    """Reads and decodes an image file into a TensorFlow tensor."""
    img_bytes = tf.io.read_file(path)
    # decode_image is format-agnostic (handles PNG, JPG, BMP, etc.)
    img = tf.image.decode_image(img_bytes, channels=channels, expand_animations=False)
    return img

def parse_and_preprocess(a_path, b_path, mask_path, img_size=(IMG_HEIGHT, IMG_WIDTH), augment=False):
    """Full data loading and preprocessing pipeline for one sample."""
    # 1. Read raw images as uint8 tensors
    a_uint8 = _read_image_tf(a_path, channels=3)
    b_uint8 = _read_image_tf(b_path, channels=3)
    # Read mask with channels=0 (auto-detect) to handle various formats like BMP correctly
    mask_uint8 = _read_image_tf(mask_path, channels=0) 
    
    # 2. Ensure mask is single-channel. If a grayscale image was saved as 3-channel, convert it.
    if mask_uint8.shape[-1] == 3:
        mask_uint8 = tf.image.rgb_to_grayscale(mask_uint8)

    # 3. Apply CLAHE for contrast enhancement
    a_clahe = tf_apply_clahe(a_uint8)
    b_clahe = tf_apply_clahe(b_uint8)
    
    # 4. Convert to float32 and normalize pixel values to [0, 1] range
    a = tf.cast(a_clahe, tf.float32) / 255.0
    b = tf.cast(b_clahe, tf.float32) / 255.0
    mask = tf.cast(mask_uint8, tf.float32) / 255.0

    # 5. Resize images and mask to the required model input size
    a = tf.image.resize(a, img_size, method=tf.image.ResizeMethod.BILINEAR)
    b = tf.image.resize(b, img_size, method=tf.image.ResizeMethod.BILINEAR)
    mask = tf.image.resize(mask, img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # 6. Binarize mask to ensure values are strictly 0 or 1
    mask = tf.where(mask >= THRESH, 1.0, 0.0)

    # 7. Concatenate images and mask for synchronized augmentation
    input_images = tf.concat([a, b], axis=-1)
    combined = tf.concat([input_images, mask], axis=-1)

    # 8. Apply augmentations if specified
    if augment:
        combined = tf.image.random_flip_left_right(combined)
        combined = tf.image.random_flip_up_down(combined)
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        combined = tf.image.rot90(combined, k=k)
        
        # Unpack, apply brightness augmentation, and repack
        imgs = combined[..., :INPUT_CHANNELS]
        msk = combined[..., INPUT_CHANNELS:]
        imgs = tf.image.random_brightness(imgs, max_delta=0.08)
        combined = tf.concat([imgs, msk], axis=-1)

    # 9. Unpack final data for model input and label
    input_images = combined[..., :INPUT_CHANNELS]
    mask = combined[..., INPUT_CHANNELS:]

    # 10. Ensure shape consistency for the model
    input_images = tf.ensure_shape(input_images, [IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS])
    mask = tf.ensure_shape(mask, [IMG_HEIGHT, IMG_WIDTH, 1])

    return input_images, mask

print("\n--- Block 2 (Corrected): Data Pipeline, Preprocessing, and Loss Functions defined. ---")



# %%
# ===================================================================
# Block 3: UNet++ Model Architecture with Residual Connections
# ===================================================================
#
# Description:
# This block defines the UNet++ architecture, modified according to the reference paper.
# - The 'standard_unit' now includes residual connections to improve gradient flow.
# - 'Nest_Net2' implements the full nested, densely connected skip pathways that define
#   UNet++, allowing it to capture features at multiple scales.
# - The model supports deep supervision by generating multiple side outputs.
#
# ===================================================================

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
    """Defines a standard convolutional unit with two conv layers and a residual connection."""
    
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation='selu', name='conv' + stage + '_1',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    
    # Residual connection path starts from the input to the first convolution
    x0 = input_tensor
    
    x = BatchNormalization(name='bn' + stage + '_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation='selu', name='conv' + stage + '_2',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization(name='bn' + stage + '_2')(x)
    
    # Add the residual connection. A 1x1 conv is used if channel counts differ.
    if K.int_shape(x0)[-1] != nb_filter:
        x0 = Conv2D(nb_filter, (1, 1), activation='selu', name='conv' + stage + '_res',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x0)
    
    x = Add(name='resi' + stage)([x, x0])
    return x

def Nest_Net2(input_shape, num_class=1, deep_supervision=False):
    """
    Constructs the UNet++ model (Nest_Net2).
    This architecture uses densely connected skip pathways, matching the implementation
    from your notebook for consistency.
    """
    # Import layers required for the functional API
    from tensorflow.keras import Input, Model
    
    nb_filter = [32, 64, 128, 256, 512]
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    inputs = Input(shape=input_shape)

    # --- Encoder & Nested Decoder Path ---
    conv1_1 = standard_unit(inputs, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])
    
    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])
    
    # --- Deep Supervision Outputs ---
    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer='he_normal', padding='same')(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer='he_normal', padding='same')(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer='he_normal', padding='same')(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer='he_normal', padding='same')(conv1_5)

    # Final fused output layer, a form of Multiple Side-Outputs Fusion (MSOF)
    conv_fuse = concatenate([conv1_2, conv1_3, conv1_4, conv1_5], name='merge_fuse', axis=bn_axis)
    nestnet_output_5 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_5', kernel_initializer='he_normal', padding='same')(conv_fuse)

    if deep_supervision:
        outputs = [nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4, nestnet_output_5]
    else:
        # If not using deep supervision, the final fused output is used as it combines all levels.
        outputs = [nestnet_output_5]

    model = Model(inputs=inputs, outputs=outputs)
    return model

print("\n--- Block 3 (Corrected): UNet++ Model Architecture defined. ---")



# %% [markdown]
# # Smoke Test

# %%
# ===================================================================
# Block 3.5: Utility Functions
# ===================================================================
#
# Description:
# This block defines helper functions for file handling and dataset creation.
# It is a prerequisite for both the smoke test and the main training pipeline.
#
# ===================================================================

def list_files_sorted(folder):
    """Lists image files in a folder, sorted alphabetically."""
    p = Path(folder)
    if not p.exists(): return []
    return sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTS])

def pair_a_b_mask(split_dir):
    """Pairs corresponding images from A, B, and OUT subfolders."""
    a_dir, b_dir, mask_dir = Path(split_dir)/IMAGE_SUBFOLDER, Path(split_dir)/CHANGED_SUBFOLDER, Path(split_dir)/MASK_SUBFOLDER
    a_files, b_files, mask_files = list_files_sorted(a_dir), list_files_sorted(b_dir), list_files_sorted(mask_dir)
    if not all((a_files, b_files, mask_files)): return [], [], []

    b_map, mask_map = {f.stem: str(f) for f in b_files}, {f.stem: str(f) for f in mask_files}
    paired_a, paired_b, paired_mask = [], [], []
    for a in a_files:
        if a.stem in b_map and a.stem in mask_map:
            paired_a.append(str(a))
            paired_b.append(b_map[a.stem])
            paired_mask.append(mask_map[a.stem])
    return paired_a, paired_b, paired_mask

def gather_paths(base_dir):
    """Gathers paired file paths for train, val, and test splits."""
    splits = {}
    for split in ('train', 'val', 'test'):
        splits[split] = pair_a_b_mask(Path(base_dir) / split)
    return splits

def make_tf_dataset(a_paths, b_paths, mask_paths, batch_size=BATCH_SIZE, shuffle=True, augment=False, repeat=False, outputs_count=1):
    """Creates a configured TensorFlow Dataset from file paths."""
    ds = tf.data.Dataset.from_tensor_slices((a_paths, b_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(a_paths), seed=SEED)
    ds = ds.map(lambda a, b, m: parse_and_preprocess(a, b, m, augment=augment), num_parallel_calls=AUTOTUNE)
    if outputs_count > 1:
        ds = ds.map(lambda x, y: (x, tuple([y] * outputs_count)), num_parallel_calls=AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

print("--- Block 3.5: Utility Functions defined. ---")

# %%
# ===================================================================
# Block 3.6: Quick Pipeline Verification (Smoke Test)
# ===================================================================
#
# Description:
# This test runs the entire pipeline on a tiny amount of data (one batch)
# for 5 epochs. It quickly finds errors in data loading, model building,
# or callbacks before you commit to a long training session.
#
# ===================================================================
print("\n" + "="*50 + "\n===== Starting Quick Pipeline Verification (Smoke Test) =====\n" + "="*50 + "\n")

try:
    # --- Step 1: Get a tiny subset of the data (just one batch) ---
    print("--- Step 1: Gathering a small batch of data paths...")
    test_train_a, test_train_b, test_train_m = gather_paths(DATASET_ROOT_DIR / TRAINING_DATASETS[0])['train']
    test_val_a, test_val_b, test_val_m = gather_paths(DATASET_ROOT_DIR / TRAINING_DATASETS[0])['val']

    one_batch_train_a = test_train_a[:BATCH_SIZE]
    one_batch_train_b = test_train_b[:BATCH_SIZE]
    one_batch_train_m = test_train_m[:BATCH_SIZE]
    one_batch_val_a = test_val_a[:BATCH_SIZE]
    one_batch_val_b = test_val_b[:BATCH_SIZE]
    one_batch_val_m = test_val_m[:BATCH_SIZE]
    print(f"Testing with {len(one_batch_train_a)} training and {len(one_batch_val_a)} validation samples.")

    # --- Step 2: Build and compile a fresh model for the test ---
    print("\n--- Step 2: Building and compiling a fresh test model...")
    test_model = Nest_Net2(INPUT_SHAPE, num_class=NUM_CLASSES, deep_supervision=DEEP_SUPERVISION)
    outputs_count = len(test_model.outputs)
    test_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=[dynamic_bce_dice_loss] * outputs_count,
        loss_weights=DEEP_SUPERVISION_WEIGHTS,
        metrics=['accuracy'] * outputs_count
    )
    print("Model compiled successfully.")

    # --- Step 3: Create datasets from the tiny subset ---
    print("\n--- Step 3: Creating TensorFlow datasets for the test...")
    test_train_ds = make_tf_dataset(one_batch_train_a, one_batch_train_b, one_batch_train_m, BATCH_SIZE, outputs_count=outputs_count)
    test_val_ds = make_tf_dataset(one_batch_val_a, one_batch_val_b, one_batch_val_m, BATCH_SIZE, outputs_count=outputs_count)
    print("Datasets created successfully.")

    # --- Step 4: Run training for 5 epochs to test the learning rate scheduler ---
    print("\n--- Step 4: Running model.fit() for 5 epochs (1 step each)... ---")
    def lr_scheduler_test(epoch, lr):
        if (epoch + 1) % 5 == 0: return lr * 0.9048374180359595
        return lr
    lr_callback_test = keras.callbacks.LearningRateScheduler(lr_scheduler_test)

    test_history = test_model.fit(
        test_train_ds,
        validation_data=test_val_ds,
        epochs=5,             # Run for 5 epochs to trigger the lr_scheduler
        steps_per_epoch=1,    # CRITICAL: Only process one batch per training epoch
        validation_steps=1,   # CRITICAL: Only process one batch per validation epoch
        callbacks=[lr_callback_test], # Explicitly test the scheduler callback
        verbose=1
    )
    print("\n✅ Verification successful! The pipeline is working correctly.")

except Exception as e:
    print(f"\n❌ Verification FAILED! An error occurred: {e}")
    raise e # Re-raise the exception to see the full error traceback

finally:
    print("\n" + "="*50 + "\n===== Quick Verification Complete =====\n" + "="*50 + "\n")

# %%
# ===================================================================
# Block 4: Main Training, Evaluation, and Visualization
# ===================================================================
#
# Description:
# This block contains the functions to run the full training pipeline,
# evaluate the final model, and visualize the results.
#
# ===================================================================

# --- Main Training, Plotting, and Evaluation Functions ---

def train_model(model, dataset_name, train_ds, val_ds, epochs):
    """Trains the model with specified callbacks, including a learning rate scheduler."""
    print(f"\n===== Starting Training on {dataset_name} =====")
    model_file_name = f"unetpp_change_best_{dataset_name.replace('/', '_')}.h5"

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        model_file_name, save_best_only=True, monitor="val_loss" if val_ds else "loss", mode='min')

    early_stopping_callback = keras.callbacks.EarlyStopping(
        patience=8, restore_best_weights=True, monitor="val_loss" if val_ds else "loss", mode='min')

    def lr_scheduler(epoch, lr):
        if (epoch + 1) % 5 == 0:
            return lr * 0.9048374180359595
        return lr
    lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)

    callbacks_list = [checkpoint_callback, early_stopping_callback, lr_callback]
    fit_args = {'x': train_ds, 'epochs': epochs, 'callbacks': callbacks_list, 'validation_data': val_ds} if val_ds else {'x': train_ds, 'epochs': epochs, 'callbacks': callbacks_list}
    history = model.fit(**fit_args)
    np.save(f'training_history_{dataset_name.replace("/", "_")}.npy', history.history)
    print(f"\n✅ Training on {dataset_name} complete. Best model saved to {model_file_name}")
    return model, history

def plot_history(history, dataset_name):
    """Plots metrics for single or deep-supervision models."""
    print(f"\n===== Plotting Learning Curves for {dataset_name} =====")
    if not history: return

    loss_key = 'output_5_loss' if 'output_5_loss' in history else 'loss'
    val_loss_key = 'val_output_5_loss' if 'val_output_5_loss' in history else 'val_loss'
    acc_key = 'output_5_accuracy' if 'output_5_accuracy' in history else 'accuracy'
    val_acc_key = 'val_output_5_accuracy' if 'val_output_5_accuracy' in history else 'val_accuracy'
    epochs_range = range(len(history[loss_key]))

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history[loss_key], label=f'Training Loss')
    if val_loss_key in history: plt.plot(epochs_range, history[val_loss_key], label=f'Validation Loss')
    plt.title(f'Loss - {dataset_name}'); plt.legend(loc='upper right')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history[acc_key], label=f'Training Accuracy')
    if val_acc_key in history: plt.plot(epochs_range, history[val_acc_key], label=f'Validation Accuracy')
    plt.title(f'Accuracy - {dataset_name}'); plt.legend(loc='lower right')
    plt.show()

def apply_morphology(mask, kernel_size=(3,3)):
    """Applies morphological opening then closing to a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

def visualize_augmentations(ds, num_samples=3):
    """Displays examples of augmented data from the training dataset."""
    print("\n===== Visualizing Data Augmentations =====")
    plt.figure(figsize=(12, 4 * num_samples))
    for i, (img_stack, mask) in enumerate(ds.take(num_samples)):
        plt.subplot(num_samples, 3, i*3 + 1); plt.imshow(img_stack[0,...,:3]); plt.title(f"Augmented A (Sample {i+1})"); plt.axis('off')
        plt.subplot(num_samples, 3, i*3 + 2); plt.imshow(img_stack[0,...,3:6]); plt.title(f"Augmented B (Sample {i+1})"); plt.axis('off')
        plt.subplot(num_samples, 3, i*3 + 3); plt.imshow(mask[0], cmap='gray'); plt.title(f"Augmented Mask (Sample {i+1})"); plt.axis('off')
    plt.tight_layout(); plt.show()

def run_training_pipeline():
    """Manages the full sequential training process."""
    model = None
    all_histories = {}
    final_model_file = f"unetpp_change_best_{TRAINING_DATASETS[-1].replace('/', '_')}.h5"
    if Path(final_model_file).exists():
        print(f"Found checkpoint for the final training stage: {final_model_file}. Skipping training.")
        try:
            model = keras.models.load_model(final_model_file, custom_objects={'dynamic_bce_dice_loss': dynamic_bce_dice_loss})
            return model, {}
        except Exception as e:
            print(f"⚠️ Could not load final model: {e}. Proceeding with full training.")
            model = None

    for dataset_name in TRAINING_DATASETS:
        dataset_path = DATASET_ROOT_DIR / dataset_name
        train_a, train_b, train_m = gather_paths(dataset_path)['train']
        val_a, val_b, val_m = gather_paths(dataset_path)['val']
        print(f"\n--- Processing Dataset: {dataset_name} ---")
        print(f"Found samples -> train: {len(train_a)}, val: {len(val_a)}")
        if not train_a: continue

        if model is None:
            print("Creating a new UNet++ model.")
            model = Nest_Net2(INPUT_SHAPE, num_class=NUM_CLASSES, deep_supervision=DEEP_SUPERVISION)

        outputs_count = len(model.outputs) if DEEP_SUPERVISION else 1
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss=[dynamic_bce_dice_loss] * outputs_count,
            loss_weights=DEEP_SUPERVISION_WEIGHTS if DEEP_SUPERVISION else None,
            metrics=['accuracy']*outputs_count
        )
        train_ds = make_tf_dataset(train_a, train_b, train_m, BATCH_SIZE, shuffle=True, augment=True, outputs_count=outputs_count)
        val_ds = make_tf_dataset(val_a, val_b, val_m, BATCH_SIZE, shuffle=False, outputs_count=outputs_count) if val_a else None
        model, history = train_model(model, dataset_name, train_ds, val_ds, EPOCHS)
        all_histories[dataset_name] = history.history
        plot_history(history.history, dataset_name)

    print("\n🎉 All training stages are complete.")
    return model, all_histories

def read_preprocess_pair_np(a_path, b_path, mask_path, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Reads and preprocesses an image/mask pair into NumPy arrays for evaluation."""
    a_img = img_to_array(load_img(a_path, target_size=img_size))
    b_img = img_to_array(load_img(b_path, target_size=img_size))
    mask_img = img_to_array(load_img(mask_path, color_mode='grayscale', target_size=img_size))
    a_cl = apply_clahe_cv(a_img)
    b_cl = apply_clahe_cv(b_img)
    a_norm = a_cl.astype(np.float32) / 255.0
    b_norm = b_cl.astype(np.float32) / 255.0
    mask_norm = mask_img.astype(np.float32) / 255.0
    mask_bin = np.where(mask_norm >= THRESH, 1.0, 0.0).astype(np.float32)
    x = np.concatenate([a_norm, b_norm], axis=-1)
    return x, mask_bin

def evaluate_and_visualize(model):
    """Performs final evaluation and visualizes results."""
    print("\n\n" + "="*50 + "\n===== Final Model Evaluation and Visualization =====\n" + "="*50 + "\n")
    test_a, test_b, test_m = gather_paths(TEST_DATASET_PATH)['test']
    if not test_a:
        print("No test samples found. Skipping evaluation."); return

    X_test, Y_test_raw = [], []
    for a_p, b_p, m_p in zip(test_a, test_b, test_m):
        x, y = read_preprocess_pair_np(a_p, b_p, m_p)
        X_test.append(x); Y_test_raw.append(y)
    X_test, Y_test = np.array(X_test), np.array(Y_test_raw)

    print("\nGenerating predictions on the test set...")
    preds = model.predict(X_test, batch_size=BATCH_SIZE)
    preds_raw = preds[-1] if isinstance(preds, list) else preds
    preds_bin_raw = (preds_raw >= THRESH).astype(np.uint8)

    print("Applying morphological post-processing...")
    preds_bin_processed = np.array([apply_morphology(p) for p in preds_bin_raw])

    def calculate_metrics(y_true, y_pred):
        flat_true, flat_pred = y_true.flatten(), y_pred.flatten()
        return {'Accuracy': accuracy_score(flat_true, flat_pred), 'Precision': precision_score(flat_true, flat_pred, zero_division=0),
                'Recall': recall_score(flat_true, flat_pred, zero_division=0), 'F1-score': f1_score(flat_true, flat_pred, zero_division=0)}

    raw_metrics = calculate_metrics(Y_test, preds_bin_raw)
    proc_metrics = calculate_metrics(Y_test, preds_bin_processed)

    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names, raw_values, proc_values = list(raw_metrics.keys()), list(raw_metrics.values()), list(proc_metrics.values())
    x = np.arange(len(metrics_names))
    width = 0.35
    rects1 = ax.bar(x - width/2, raw_values, width, label='Raw Prediction', color='skyblue')
    rects2 = ax.bar(x + width/2, proc_values, width, label='Post-Processed', color='coral')
    ax.set_ylabel('Scores'); ax.set_title('Comparative Pixel-wise Metrics'); ax.set_xticks(x); ax.set_xticklabels(metrics_names); ax.legend(); ax.set_ylim(0, 1.05)
    ax.bar_label(rects1, padding=3, fmt='%.3f'); ax.bar_label(rects2, padding=3, fmt='%.3f')
    fig.tight_layout(); plt.show()

    cm = confusion_matrix(Y_test.flatten(), preds_bin_processed.flatten()).ravel()
    cm_data = {'True Neg': cm[0], 'False Pos': cm[1], 'False Neg': cm[2], 'True Pos': cm[3]}
    plt.figure(figsize=(8, 5)); bars = plt.bar(cm_data.keys(), cm_data.values(), color=['#4CAF50', '#F44336', '#FF9800', '#2196F3'])
    plt.title('Confusion Matrix Counts (Post-Processed)'); plt.ylabel('Pixel Count')
    plt.bar_label(bars, fmt='{:,.0f}'); plt.show()

    visualize_all_steps(test_a, test_b, test_m, model, preds_bin_raw, preds_bin_processed)

def visualize_all_steps(test_a, test_b, test_m, model, preds_bin_raw, preds_bin_processed):
    """Generates a detailed visual comparison of the entire pipeline for a few samples."""
    print("\n===== Full Pipeline Visualization =====")
    indices = np.random.choice(len(test_a), size=min(NUM_VIS_SAMPLES, len(test_a)), replace=False)

    for i, idx in enumerate(indices):
        a_path, b_path, m_path = test_a[idx], test_b[idx], test_m[idx]
        _, y_np = read_preprocess_pair_np(a_path, b_path, m_path)
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        img_a_uint8 = img_to_array(load_img(a_path)).astype(np.uint8)
        axes[0].imshow(img_a_uint8); axes[0].set_title(f"Original A (idx={idx})")
        axes[1].imshow(apply_clahe_cv(img_a_uint8)); axes[1].set_title("A - After CLAHE")
        axes[2].imshow(y_np, cmap='gray'); axes[2].set_title("Ground Truth")
        axes[3].imshow(preds_bin_raw[idx], cmap='gray'); axes[3].set_title("Raw Prediction")
        axes[4].imshow(preds_bin_processed[idx], cmap='gray'); axes[4].set_title("Post-Processed")
        for ax in axes: ax.axis('off')
        plt.tight_layout(); plt.show()


# ===================================================================
# Main Execution Block
# ===================================================================
# --- Step 1: Run Full Training Pipeline ---
final_model, all_histories = run_training_pipeline()

# --- Step 2: Run Final Evaluation and Visualization ---
if final_model:
    evaluate_and_visualize(final_model)


