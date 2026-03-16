# %%
# ===================================================================
# Resume UNet++ Training from Checkpoint - Complete Notebook
# ===================================================================
# This notebook resumes training from the saved checkpoint and runs
# complete evaluation with all visualizations.
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
import cv2

print("✅ All imports successful!")

# ===================================================================
# CONFIGURATION - Keep exactly the same as original training
# ===================================================================

SEED = 1998
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
DATASET_ROOT_DIR = Path('/kaggle/input/satellitechangedetectiondataset/ChangeDetectionDataset')
TRAINING_DATASET = 'Real/subset'  # Only the final stage
TEST_DATASET_PATH = DATASET_ROOT_DIR / 'Real/subset'

# Checkpoint path
CHECKPOINT_PATH = '/kaggle/input/my-dataset/unetpp_change_best_Real_subset.h5'
# If checkpoint is in working directory, use:
# CHECKPOINT_PATH = 'unetpp_change_best_Real_subset.h5'

IMAGE_SUBFOLDER = 'A'
CHANGED_SUBFOLDER = 'B'
MASK_SUBFOLDER = 'OUT'
ALLOWED_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

# Model parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
INPUT_CHANNELS = 6
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS)
NUM_CLASSES = 1

# Training parameters
EPOCHS = 70  # Total epochs (training will continue from where it left off)
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DEEP_SUPERVISION = True
AUTOTUNE = tf.data.AUTOTUNE
DEEP_SUPERVISION_WEIGHTS = [1.13, 1.12, 1.112, 1.13, 1.13]

# Evaluation parameters
THRESH = 0.5
NUM_VIS_SAMPLES = 6

# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"✅ Enabled memory growth on {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"⚠️ Could not set memory growth: {e}")
else:
    print("⚠️ No GPU detected; running on CPU.")

print("\n" + "="*60)
print("Configuration loaded successfully!")
print("="*60 + "\n")

# ===================================================================
# LOSS FUNCTIONS - Must match original training exactly
# ===================================================================

def dice_coef(y_true, y_pred, smooth=1):
    """Computes the Dice coefficient."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """Computes the Dice loss."""
    return 1 - dice_coef(y_true, y_pred)

def dynamic_bce_dice_loss(y_true, y_pred):
    """Dynamic BCE + Dice loss with per-batch weighting."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    beta = tf.reduce_mean(1.0 - y_true)
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * beta * y_true + bce * (1.0 - beta) * (1.0 - y_true))
    dice_loss = dice_coef_loss(y_true, y_pred)
    
    return weighted_bce + 0.559 * dice_loss

print("✅ Loss functions defined!")

# ===================================================================
# DATA PREPROCESSING - Must match original exactly
# ===================================================================

def apply_clahe_cv(image):
    """Applies CLAHE preprocessing."""
    if hasattr(image, 'numpy'):
        image_np = image.numpy()
    else:
        image_np = image
    
    image_np = image_np.astype(np.uint8)
    lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(lab_image)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(limg, cv2.COLOR_Lab2RGB)
    return final_image.astype(np.uint8)

@tf.function
def tf_apply_clahe(image):
    """TensorFlow wrapper for CLAHE."""
    im_shape = image.shape
    [image,] = tf.py_function(apply_clahe_cv, [image], [tf.uint8])
    image.set_shape(im_shape)
    return image

def _read_image_tf(path, channels=3):
    """Reads and decodes an image file."""
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=channels, expand_animations=False)
    return img

def parse_and_preprocess(a_path, b_path, mask_path, img_size=(IMG_HEIGHT, IMG_WIDTH), augment=False):
    """Full preprocessing pipeline."""
    # Read images
    a_uint8 = _read_image_tf(a_path, channels=3)
    b_uint8 = _read_image_tf(b_path, channels=3)
    mask_uint8 = _read_image_tf(mask_path, channels=0)
    
    # Ensure mask is single channel
    if mask_uint8.shape[-1] == 3:
        mask_uint8 = tf.image.rgb_to_grayscale(mask_uint8)
    
    # Apply CLAHE
    a_clahe = tf_apply_clahe(a_uint8)
    b_clahe = tf_apply_clahe(b_uint8)
    
    # Normalize
    a = tf.cast(a_clahe, tf.float32) / 255.0
    b = tf.cast(b_clahe, tf.float32) / 255.0
    mask = tf.cast(mask_uint8, tf.float32) / 255.0
    
    # Resize
    a = tf.image.resize(a, img_size, method=tf.image.ResizeMethod.BILINEAR)
    b = tf.image.resize(b, img_size, method=tf.image.ResizeMethod.BILINEAR)
    mask = tf.image.resize(mask, img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    # Binarize mask
    mask = tf.where(mask >= THRESH, 1.0, 0.0)
    
    # Concatenate for synchronized augmentation
    input_images = tf.concat([a, b], axis=-1)
    combined = tf.concat([input_images, mask], axis=-1)
    
    # Augmentation
    if augment:
        combined = tf.image.random_flip_left_right(combined)
        combined = tf.image.random_flip_up_down(combined)
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        combined = tf.image.rot90(combined, k=k)
        
        imgs = combined[..., :INPUT_CHANNELS]
        msk = combined[..., INPUT_CHANNELS:]
        imgs = tf.image.random_brightness(imgs, max_delta=0.08)
        combined = tf.concat([imgs, msk], axis=-1)
    
    # Unpack
    input_images = combined[..., :INPUT_CHANNELS]
    mask = combined[..., INPUT_CHANNELS:]
    
    # Ensure shape
    input_images = tf.ensure_shape(input_images, [IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS])
    mask = tf.ensure_shape(mask, [IMG_HEIGHT, IMG_WIDTH, 1])
    
    return input_images, mask

print("✅ Data preprocessing functions defined!")

# ===================================================================
# UTILITY FUNCTIONS
# ===================================================================

def list_files_sorted(folder):
    """Lists image files in a folder, sorted."""
    p = Path(folder)
    if not p.exists():
        return []
    return sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTS])

def pair_a_b_mask(split_dir):
    """Pairs corresponding images from A, B, and OUT subfolders."""
    a_dir = Path(split_dir) / IMAGE_SUBFOLDER
    b_dir = Path(split_dir) / CHANGED_SUBFOLDER
    mask_dir = Path(split_dir) / MASK_SUBFOLDER
    
    a_files = list_files_sorted(a_dir)
    b_files = list_files_sorted(b_dir)
    mask_files = list_files_sorted(mask_dir)
    
    if not all((a_files, b_files, mask_files)):
        return [], [], []
    
    b_map = {f.stem: str(f) for f in b_files}
    mask_map = {f.stem: str(f) for f in mask_files}
    
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
        split_path = Path(base_dir) / split
        if split_path.exists():
            splits[split] = pair_a_b_mask(split_path)
        else:
            splits[split] = ([], [], [])
    return splits

def make_tf_dataset(a_paths, b_paths, mask_paths, batch_size=BATCH_SIZE, 
                    shuffle=True, augment=False, repeat=False, outputs_count=1):
    """Creates a TensorFlow Dataset."""
    if len(a_paths) == 0:
        return None
    
    ds = tf.data.Dataset.from_tensor_slices((a_paths, b_paths, mask_paths))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(a_paths), seed=SEED)
    
    ds = ds.map(
        lambda a, b, m: parse_and_preprocess(a, b, m, augment=augment),
        num_parallel_calls=AUTOTUNE
    )
    
    if outputs_count > 1:
        ds = ds.map(
            lambda x, y: (x, tuple([y] * outputs_count)),
            num_parallel_calls=AUTOTUNE
        )
    
    if repeat:
        ds = ds.repeat()
    
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

print("✅ Utility functions defined!")

# ===================================================================
# LOAD CHECKPOINT AND PREPARE FOR TRAINING
# ===================================================================

print("\n" + "="*60)
print("LOADING CHECKPOINT")
print("="*60 + "\n")

try:
    # Load the model with custom objects
    print(f"Loading model from: {CHECKPOINT_PATH}")
    model = keras.models.load_model(
        CHECKPOINT_PATH,
        custom_objects={'dynamic_bce_dice_loss': dynamic_bce_dice_loss},
        compile=False  # We'll recompile with our settings
    )
    print("✅ Model loaded successfully!")
    
    # Display model summary
    print("\nModel Summary:")
    model.summary()
    
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    print("\nPlease ensure:")
    print("1. The checkpoint file exists at the specified path")
    print("2. The path is correct (check if it's in /kaggle/input/ or working directory)")
    raise e

# Recompile the model with the same settings
outputs_count = len(model.outputs)
print(f"\nModel has {outputs_count} outputs (deep supervision: {DEEP_SUPERVISION})")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=[dynamic_bce_dice_loss] * outputs_count,
    loss_weights=DEEP_SUPERVISION_WEIGHTS if DEEP_SUPERVISION else None,
    metrics=['accuracy'] * outputs_count
)
print("✅ Model recompiled successfully!")

# ===================================================================
# PREPARE DATA FOR CONTINUED TRAINING
# ===================================================================

print("\n" + "="*60)
print("PREPARING DATA")
print("="*60 + "\n")

dataset_path = DATASET_ROOT_DIR / TRAINING_DATASET
print(f"Loading data from: {dataset_path}")

splits = gather_paths(dataset_path)
train_a, train_b, train_m = splits['train']
val_a, val_b, val_m = splits['val']
test_a, test_b, test_m = splits['test']

print(f"✅ Data loaded:")
print(f"   Training samples: {len(train_a)}")
print(f"   Validation samples: {len(val_a)}")
print(f"   Test samples: {len(test_a)}")

if len(train_a) == 0:
    raise ValueError("No training data found! Check dataset path.")

# Create datasets
print("\nCreating TensorFlow datasets...")
train_ds = make_tf_dataset(
    train_a, train_b, train_m,
    batch_size=BATCH_SIZE,
    shuffle=True,
    augment=True,
    outputs_count=outputs_count
)

val_ds = None
if len(val_a) > 0:
    val_ds = make_tf_dataset(
        val_a, val_b, val_m,
        batch_size=BATCH_SIZE,
        shuffle=False,
        outputs_count=outputs_count
    )
    print("✅ Validation dataset created")
else:
    print("⚠️ No validation data available")

print("✅ Training dataset created")

# ===================================================================
# VISUALIZE AUGMENTED SAMPLES
# ===================================================================

print("\n" + "="*60)
print("VISUALIZING AUGMENTED TRAINING SAMPLES")
print("="*60 + "\n")

plt.figure(figsize=(15, 12))
for i, (img_stack, mask) in enumerate(train_ds.take(3)):
    # Get first sample from batch
    img_a = img_stack[0, ..., :3].numpy()
    img_b = img_stack[0, ..., 3:6].numpy()
    mask_img = mask[0][0].numpy() if isinstance(mask, tuple) else mask[0].numpy()
    
    plt.subplot(3, 3, i*3 + 1)
    plt.imshow(img_a)
    plt.title(f"Augmented Image A - Sample {i+1}")
    plt.axis('off')
    
    plt.subplot(3, 3, i*3 + 2)
    plt.imshow(img_b)
    plt.title(f"Augmented Image B - Sample {i+1}")
    plt.axis('off')
    
    plt.subplot(3, 3, i*3 + 3)
    plt.imshow(mask_img, cmap='gray')
    plt.title(f"Augmented Mask - Sample {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('augmented_samples.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Augmentation visualization complete!")

# ===================================================================
# RESUME TRAINING
# ===================================================================

print("\n" + "="*60)
print("RESUMING TRAINING")
print("="*60 + "\n")

# Setup callbacks
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    'unetpp_change_best_Real_subset_continued.h5',
    save_best_only=True,
    monitor="val_loss" if val_ds else "loss",
    mode='min',
    verbose=1
)

early_stopping_callback = keras.callbacks.EarlyStopping(
    patience=8,
    restore_best_weights=True,
    monitor="val_loss" if val_ds else "loss",
    mode='min',
    verbose=1
)

def lr_scheduler(epoch, lr):
    """Learning rate scheduler - reduces LR every 5 epochs."""
    if (epoch + 1) % 5 == 0:
        new_lr = lr * 0.9048374180359595
        print(f"\n📉 Learning rate reduced to: {new_lr:.6f}")
        return new_lr
    return lr

lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)

callbacks_list = [checkpoint_callback, early_stopping_callback, lr_callback]

# Train the model
print(f"Training for {EPOCHS} epochs...")
print(f"Batch size: {BATCH_SIZE}")
print(f"Initial learning rate: {LEARNING_RATE}")
print()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_list,
    verbose=1
)

# Save history
np.save('training_history_continued.npy', history.history)
print("\n✅ Training complete! History saved.")

# ===================================================================
# PLOT TRAINING HISTORY
# ===================================================================

print("\n" + "="*60)
print("PLOTTING TRAINING HISTORY")
print("="*60 + "\n")

# Determine correct keys
loss_key = 'output_5_loss' if 'output_5_loss' in history.history else 'loss'
val_loss_key = 'val_output_5_loss' if 'val_output_5_loss' in history.history else 'val_loss'
acc_key = 'output_5_accuracy' if 'output_5_accuracy' in history.history else 'accuracy'
val_acc_key = 'val_output_5_accuracy' if 'val_output_5_accuracy' in history.history else 'val_accuracy'

epochs_range = range(1, len(history.history[loss_key]) + 1)

plt.figure(figsize=(16, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history[loss_key], 'b-', label='Training Loss', linewidth=2)
if val_loss_key in history.history and history.history[val_loss_key]:
    plt.plot(epochs_range, history.history[val_loss_key], 'r-', label='Validation Loss', linewidth=2)
plt.title('Model Loss (Continued Training)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history[acc_key], 'b-', label='Training Accuracy', linewidth=2)
if val_acc_key in history.history and history.history[val_acc_key]:
    plt.plot(epochs_range, history.history[val_acc_key], 'r-', label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy (Continued Training)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Training history plotted!")

# ===================================================================
# MORPHOLOGICAL POST-PROCESSING
# ===================================================================

def apply_morphology(mask, kernel_size=(3, 3)):
    """Applies morphological opening then closing."""
    mask_np = mask.squeeze().astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    opened = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed

# ===================================================================
# EVALUATION ON TEST SET
# ===================================================================

print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60 + "\n")

if len(test_a) == 0:
    print("⚠️ No test samples found. Using validation set for evaluation.")
    test_a, test_b, test_m = val_a, val_b, val_m

if len(test_a) == 0:
    print("❌ No data available for evaluation!")
else:
    print(f"Evaluating on {len(test_a)} test samples...")
    
    # Prepare test data
    X_test = []
    Y_test = []
    
    print("Loading and preprocessing test data...")
    for i, (a_p, b_p, m_p) in enumerate(zip(test_a, test_b, test_m)):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(test_a)} samples...")
        
        # Read and preprocess
        a_img = img_to_array(load_img(a_p, target_size=(IMG_HEIGHT, IMG_WIDTH)))
        b_img = img_to_array(load_img(b_p, target_size=(IMG_HEIGHT, IMG_WIDTH)))
        mask_img = img_to_array(load_img(m_p, color_mode='grayscale', target_size=(IMG_HEIGHT, IMG_WIDTH)))
        
        # Apply CLAHE
        a_cl = apply_clahe_cv(a_img)
        b_cl = apply_clahe_cv(b_img)
        
        # Normalize
        a_norm = a_cl.astype(np.float32) / 255.0
        b_norm = b_cl.astype(np.float32) / 255.0
        mask_norm = mask_img.astype(np.float32) / 255.0
        
        # Binarize mask
        mask_bin = np.where(mask_norm >= THRESH, 1.0, 0.0)
        
        # Concatenate
        x = np.concatenate([a_norm, b_norm], axis=-1)
        
        X_test.append(x)
        Y_test.append(mask_bin)
    
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    print(f"✅ Test data prepared: {X_test.shape}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    
    # Handle deep supervision output
    if isinstance(predictions, list):
        preds_raw = predictions[-1]  # Use final output
        print(f"Using final output from {len(predictions)} deep supervision outputs")
    else:
        preds_raw = predictions
    
    print(f"Raw predictions shape: {preds_raw.shape}")
    
    # Threshold predictions
    preds_bin_raw = (preds_raw >= THRESH).astype(np.uint8)
    
    # Apply morphological post-processing
    print("\nApplying morphological post-processing...")
    preds_bin_processed = np.array([apply_morphology(p) for p in preds_bin_raw])
    print("✅ Post-processing complete!")
    
    # ===================================================================
    # CALCULATE METRICS
    # ===================================================================
    
    print("\n" + "="*60)
    print("CALCULATING METRICS")
    print("="*60 + "\n")
    
    def calculate_metrics(y_true, y_pred):
        """Calculate pixel-wise metrics."""
        flat_true = y_true.flatten()
        flat_pred = y_pred.flatten()
        
        return {
            'Accuracy': accuracy_score(flat_true, flat_pred),
            'Precision': precision_score(flat_true, flat_pred, zero_division=0),
            'Recall': recall_score(flat_true, flat_pred, zero_division=0),
            'F1-Score': f1_score(flat_true, flat_pred, zero_division=0)
        }
    
    # Calculate metrics for both raw and processed predictions
    raw_metrics = calculate_metrics(Y_test, preds_bin_raw)
    proc_metrics = calculate_metrics(Y_test, preds_bin_processed)
    
    print("Raw Prediction Metrics:")
    for metric, value in raw_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nPost-Processed Metrics:")
    for metric, value in proc_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # ===================================================================
    # VISUALIZE METRICS COMPARISON
    # ===================================================================
    
    print("\n" + "="*60)
    print("VISUALIZING METRICS")
    print("="*60 + "\n")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metrics_names = list(raw_metrics.keys())
    raw_values = list(raw_metrics.values())
    proc_values = list(proc_metrics.values())
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, raw_values, width, label='Raw Prediction', 
                    color='skyblue', edgecolor='navy', linewidth=1.5)
    rects2 = ax.bar(x + width/2, proc_values, width, label='Post-Processed', 
                    color='coral', edgecolor='darkred', linewidth=1.5)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comparative Pixel-wise Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=9)
    ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Metrics comparison plotted!")
    
    # ===================================================================
    # CONFUSION MATRIX
    # ===================================================================
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60 + "\n")
    
    cm = confusion_matrix(Y_test.flatten(), preds_bin_processed.flatten())
    tn, fp, fn, tp = cm.ravel()
    
    print(f"True Negatives:  {tn:,}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    print(f"True Positives:  {tp:,}")
    
    # Plot confusion matrix as bar chart
    plt.figure(figsize=(10, 6))
    cm_data = {
        'True Neg': tn,
        'False Pos': fp,
        'False Neg': fn,
        'True Pos': tp
    }
    
    bars = plt.bar(cm_data.keys(), cm_data.values(), 
                   color=['#4CAF50', '#F44336', '#FF9800', '#2196F3'],
                   edgecolor='black', linewidth=1.5)
    
    plt.title('Confusion Matrix - Post-Processed Predictions', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Pixel Count', fontsize=12, fontweight='bold')
    plt.bar_label(bars, fmt='{:,.0f}', fontsize=10, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Confusion matrix plotted!")
    
    # ===================================================================
    # DETAILED VISUALIZATION - Full Pipeline
    # ===================================================================
    
    print("\n" + "="*60)
    print("DETAILED PIPELINE VISUALIZATION")
    print("="*60 + "\n")
    
    # Select random samples for visualization
    num_samples = min(NUM_VIS_SAMPLES, len(test_a))
    indices = np.random.choice(len(test_a), size=num_samples, replace=False)
    
    print(f"Visualizing {num_samples} random samples...")
    
    for i, idx in enumerate(indices):
        a_path = test_a[idx]
        b_path = test_b[idx]
        m_path = test_m[idx]
        
        # Load original images
        img_a_original = img_to_array(load_img(a_path)).astype(np.uint8)
        img_b_original = img_to_array(load_img(b_path)).astype(np.uint8)
        
        # Apply CLAHE
        img_a_clahe = apply_clahe_cv(img_a_original)
        img_b_clahe = apply_clahe_cv(img_b_original)
        
        # Get ground truth, raw prediction, and post-processed prediction
        ground_truth = Y_test[idx].squeeze()
        raw_pred = preds_bin_raw[idx].squeeze()
        proc_pred = preds_bin_processed[idx].squeeze()
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Sample {i+1} (Index: {idx}) - Complete Pipeline', 
                     fontsize=16, fontweight='bold')
        
        # Row 1: Image A progression
        axes[0, 0].imshow(img_a_original)
        axes[0, 0].set_title('Original Image A', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img_a_clahe)
        axes[0, 1].set_title('Image A After CLAHE', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(img_b_original)
        axes[0, 2].set_title('Original Image B', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(img_b_clahe)
        axes[0, 3].set_title('Image B After CLAHE', fontsize=12, fontweight='bold')
        axes[0, 3].axis('off')
        
        # Row 2: Predictions
        axes[1, 0].imshow(ground_truth, cmap='gray')
        axes[1, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(raw_pred, cmap='gray')
        axes[1, 1].set_title('Raw Prediction', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(proc_pred, cmap='gray')
        axes[1, 2].set_title('Post-Processed', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        # Difference visualization
        diff = np.abs(ground_truth.astype(float) - proc_pred.astype(float))
        axes[1, 3].imshow(diff, cmap='hot')
        axes[1, 3].set_title('Prediction Error (Red=Wrong)', fontsize=12, fontweight='bold')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'detailed_visualization_sample_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Sample {i+1} visualization complete")
    
    print("\n✅ All detailed visualizations complete!")
    
    # ===================================================================
    # SIDE-BY-SIDE COMPARISON GRID
    # ===================================================================
    
    print("\n" + "="*60)
    print("CREATING COMPARISON GRID")
    print("="*60 + "\n")
    
    # Create a large grid showing multiple samples
    grid_samples = min(6, len(test_a))
    grid_indices = np.random.choice(len(test_a), size=grid_samples, replace=False)
    
    fig, axes = plt.subplots(grid_samples, 5, figsize=(20, 4*grid_samples))
    fig.suptitle('Multi-Sample Comparison Grid', fontsize=18, fontweight='bold', y=0.995)
    
    for row, idx in enumerate(grid_indices):
        # Load images
        img_a = img_to_array(load_img(test_a[idx])).astype(np.uint8)
        img_b = img_to_array(load_img(test_b[idx])).astype(np.uint8)
        
        # Get predictions
        gt = Y_test[idx].squeeze()
        raw = preds_bin_raw[idx].squeeze()
        proc = preds_bin_processed[idx].squeeze()
        
        # Plot
        if grid_samples == 1:
            axes[0].imshow(img_a)
            axes[0].set_title('Image A', fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(img_b)
            axes[1].set_title('Image B', fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(gt, cmap='gray')
            axes[2].set_title('Ground Truth', fontweight='bold')
            axes[2].axis('off')
            
            axes[3].imshow(raw, cmap='gray')
            axes[3].set_title('Raw Pred', fontweight='bold')
            axes[3].axis('off')
            
            axes[4].imshow(proc, cmap='gray')
            axes[4].set_title('Post-Proc', fontweight='bold')
            axes[4].axis('off')
        else:
            axes[row, 0].imshow(img_a)
            axes[row, 0].set_title(f'Sample {row+1}: Image A' if row == 0 else '', fontweight='bold')
            axes[row, 0].axis('off')
            
            axes[row, 1].imshow(img_b)
            axes[row, 1].set_title(f'Image B' if row == 0 else '', fontweight='bold')
            axes[row, 1].axis('off')
            
            axes[row, 2].imshow(gt, cmap='gray')
            axes[row, 2].set_title(f'Ground Truth' if row == 0 else '', fontweight='bold')
            axes[row, 2].axis('off')
            
            axes[row, 3].imshow(raw, cmap='gray')
            axes[row, 3].set_title(f'Raw Pred' if row == 0 else '', fontweight='bold')
            axes[row, 3].axis('off')
            
            axes[row, 4].imshow(proc, cmap='gray')
            axes[row, 4].set_title(f'Post-Processed' if row == 0 else '', fontweight='bold')
            axes[row, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Comparison grid complete!")
    
    # ===================================================================
    # OVERLAY VISUALIZATIONS
    # ===================================================================
    
    print("\n" + "="*60)
    print("CREATING OVERLAY VISUALIZATIONS")
    print("="*60 + "\n")
    
    overlay_samples = min(3, len(test_a))
    overlay_indices = np.random.choice(len(test_a), size=overlay_samples, replace=False)
    
    for i, idx in enumerate(overlay_indices):
        img_b = img_to_array(load_img(test_b[idx])).astype(np.uint8)
        gt = Y_test[idx].squeeze()
        pred = preds_bin_processed[idx].squeeze()
        
        # Create colored overlays
        # Green = True Positive, Red = False Positive, Yellow = False Negative
        overlay = img_b.copy()
        
        # True Positives (Green)
        tp_mask = (gt == 1) & (pred == 1)
        overlay[tp_mask] = [0, 255, 0]
        
        # False Positives (Red)
        fp_mask = (gt == 0) & (pred == 1)
        overlay[fp_mask] = [255, 0, 0]
        
        # False Negatives (Yellow)
        fn_mask = (gt == 1) & (pred == 0)
        overlay[fn_mask] = [255, 255, 0]
        
        # Blend overlay with original image
        alpha = 0.5
        blended = cv2.addWeighted(img_b, 1-alpha, overlay, alpha, 0)
        
        # Visualize
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'Overlay Analysis - Sample {i+1}', fontsize=14, fontweight='bold')
        
        axes[0].imshow(img_b)
        axes[0].set_title('Original Image B', fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(gt, cmap='gray')
        axes[1].set_title('Ground Truth', fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title('Prediction', fontweight='bold')
        axes[2].axis('off')
        
        axes[3].imshow(blended)
        axes[3].set_title('Overlay (Green=TP, Red=FP, Yellow=FN)', fontweight='bold')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'overlay_analysis_sample_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Overlay {i+1} complete")
    
    print("\n✅ All overlay visualizations complete!")
    
    # ===================================================================
    # STATISTICAL ANALYSIS
    # ===================================================================
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60 + "\n")
    
    # Calculate per-sample metrics
    sample_accuracies = []
    sample_f1_scores = []
    sample_ious = []
    
    for i in range(len(Y_test)):
        gt_flat = Y_test[i].flatten()
        pred_flat = preds_bin_processed[i].flatten()
        
        acc = accuracy_score(gt_flat, pred_flat)
        f1 = f1_score(gt_flat, pred_flat, zero_division=0)
        
        # Calculate IoU
        intersection = np.sum((gt_flat == 1) & (pred_flat == 1))
        union = np.sum((gt_flat == 1) | (pred_flat == 1))
        iou = intersection / union if union > 0 else 0
        
        sample_accuracies.append(acc)
        sample_f1_scores.append(f1)
        sample_ious.append(iou)
    
    sample_accuracies = np.array(sample_accuracies)
    sample_f1_scores = np.array(sample_f1_scores)
    sample_ious = np.array(sample_ious)
    
    # Print statistics
    print("Per-Sample Statistics:")
    print(f"\nAccuracy:")
    print(f"  Mean: {sample_accuracies.mean():.4f}")
    print(f"  Std:  {sample_accuracies.std():.4f}")
    print(f"  Min:  {sample_accuracies.min():.4f}")
    print(f"  Max:  {sample_accuracies.max():.4f}")
    
    print(f"\nF1-Score:")
    print(f"  Mean: {sample_f1_scores.mean():.4f}")
    print(f"  Std:  {sample_f1_scores.std():.4f}")
    print(f"  Min:  {sample_f1_scores.min():.4f}")
    print(f"  Max:  {sample_f1_scores.max():.4f}")
    
    print(f"\nIoU (Intersection over Union):")
    print(f"  Mean: {sample_ious.mean():.4f}")
    print(f"  Std:  {sample_ious.std():.4f}")
    print(f"  Min:  {sample_ious.min():.4f}")
    print(f"  Max:  {sample_ious.max():.4f}")
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(sample_accuracies, bins=20, edgecolor='black', color='skyblue')
    axes[0].axvline(sample_accuracies.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sample_accuracies.mean():.3f}')
    axes[0].set_title('Accuracy Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Accuracy', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(sample_f1_scores, bins=20, edgecolor='black', color='lightcoral')
    axes[1].axvline(sample_f1_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sample_f1_scores.mean():.3f}')
    axes[1].set_title('F1-Score Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('F1-Score', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(sample_ious, bins=20, edgecolor='black', color='lightgreen')
    axes[2].axvline(sample_ious.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sample_ious.mean():.3f}')
    axes[2].set_title('IoU Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('IoU', fontsize=10)
    axes[2].set_ylabel('Frequency', fontsize=10)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('statistical_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Statistical distributions plotted!")
    
    # ===================================================================
    # BEST AND WORST PREDICTIONS
    # ===================================================================
    
    print("\n" + "="*60)
    print("BEST AND WORST PREDICTIONS")
    print("="*60 + "\n")
    
    # Find best and worst predictions based on F1-score
    best_indices = np.argsort(sample_f1_scores)[-3:][::-1]
    worst_indices = np.argsort(sample_f1_scores)[:3]
    
    print("Top 3 Best Predictions (by F1-Score):")
    for i, idx in enumerate(best_indices):
        print(f"  {i+1}. Sample {idx}: F1={sample_f1_scores[idx]:.4f}, IoU={sample_ious[idx]:.4f}")
    
    print("\nTop 3 Worst Predictions (by F1-Score):")
    for i, idx in enumerate(worst_indices):
        print(f"  {i+1}. Sample {idx}: F1={sample_f1_scores[idx]:.4f}, IoU={sample_ious[idx]:.4f}")
    
    # Visualize best predictions
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Best Predictions (Highest F1-Scores)', fontsize=16, fontweight='bold')
    
    for row, idx in enumerate(best_indices):
        img_b = img_to_array(load_img(test_b[idx])).astype(np.uint8)
        gt = Y_test[idx].squeeze()
        pred = preds_bin_processed[idx].squeeze()
        
        axes[row, 0].imshow(img_b)
        axes[row, 0].set_title(f'Sample {idx}: Image B', fontweight='bold')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(gt, cmap='gray')
        axes[row, 1].set_title(f'Ground Truth', fontweight='bold')
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(pred, cmap='gray')
        axes[row, 2].set_title(f'Prediction (F1={sample_f1_scores[idx]:.3f})', fontweight='bold')
        axes[row, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('best_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Visualize worst predictions
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Worst Predictions (Lowest F1-Scores)', fontsize=16, fontweight='bold')
    
    for row, idx in enumerate(worst_indices):
        img_b = img_to_array(load_img(test_b[idx])).astype(np.uint8)
        gt = Y_test[idx].squeeze()
        pred = preds_bin_processed[idx].squeeze()
        
        axes[row, 0].imshow(img_b)
        axes[row, 0].set_title(f'Sample {idx}: Image B', fontweight='bold')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(gt, cmap='gray')
        axes[row, 1].set_title(f'Ground Truth', fontweight='bold')
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(pred, cmap='gray')
        axes[row, 2].set_title(f'Prediction (F1={sample_f1_scores[idx]:.3f})', fontweight='bold')
        axes[row, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('worst_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Best and worst predictions visualized!")

# ===================================================================
# FINAL SUMMARY
# ===================================================================

print("\n" + "="*60)
print("TRAINING AND EVALUATION COMPLETE!")
print("="*60 + "\n")

print("📊 Summary:")
print(f"  - Model successfully resumed from checkpoint")
print(f"  - Training completed for {len(history.history[loss_key])} epochs")
print(f"  - Evaluated on {len(test_a) if len(test_a) > 0 else 'N/A'} test samples")
if len(test_a) > 0:
    print(f"  - Final Accuracy: {proc_metrics['Accuracy']:.4f}")
    print(f"  - Final F1-Score: {proc_metrics['F1-Score']:.4f}")
    print(f"  - Mean IoU: {sample_ious.mean():.4f}")

print("\n📁 Saved Files:")
print("  - unetpp_change_best_Real_subset_continued.h5 (trained model)")
print("  - training_history_continued.npy (training history)")
print("  - augmented_samples.png")
print("  - training_history.png")
print("  - metrics_comparison.png")
print("  - confusion_matrix.png")
print("  - detailed_visualization_sample_*.png")
print("  - comparison_grid.png")
print("  - overlay_analysis_sample_*.png")
print("  - statistical_distributions.png")
print("  - best_predictions.png")
print("  - worst_predictions.png")

print("\n✨ All visualizations and evaluations complete!")
print("="*60)

# %%



