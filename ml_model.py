import os
import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Fruit_dataset')
TRAIN_DIR   = os.path.join(DATASET_DIR, 'train1')
VAL_DIR     = os.path.join(DATASET_DIR, 'val1')
MODEL_DIR   = os.path.join(BASE_DIR, 'model')
STATIC_DIR  = os.path.join(BASE_DIR, 'static', 'charts')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

CLASSES = ['apple', 'banana', 'grape', 'guava', 'mango', 'papaya']
IMG_SIZE = (100, 100)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_color_features(img_bgr):
    """Extract mean & std of R, G, B channels and HSV channels."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    feats = []
    for ch in cv2.split(img_rgb):
        feats += [float(ch.mean()), float(ch.std())]
    for ch in cv2.split(img_hsv):
        feats += [float(ch.mean()), float(ch.std())]
    return feats  # 12 features


def extract_size_features(img_bgr):
    """Extract size / shape features using contour detection."""
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area    = float(cv2.contourArea(largest))
        peri    = float(cv2.arcLength(largest, True))
        x, y, w, h = cv2.boundingRect(largest)
        aspect = float(w) / float(h) if h > 0 else 1.0
        extent = area / float(w * h) if w * h > 0 else 0.0
        circularity = (4 * np.pi * area / (peri ** 2)) if peri > 0 else 0.0
    else:
        area, peri, aspect, extent, circularity = 0.0, 0.0, 1.0, 0.0, 0.0

    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    size_ratio   = area / float(total_pixels) if total_pixels > 0 else 0.0

    return [area, peri, aspect, extent, circularity, size_ratio]  # 6 features


def extract_features(img_bgr):
    """Combined feature vector: 18 features total."""
    img_resized = cv2.resize(img_bgr, IMG_SIZE)
    color_feats = extract_color_features(img_resized)
    size_feats  = extract_size_features(img_resized)
    return color_feats + size_feats


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(data_dir, classes=CLASSES):
    X, y, filenames = [], [], []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] Missing class folder: {cls_dir}")
            continue
        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Loading {len(imgs):>4} images for class '{cls}' ...")
        for fname in imgs:
            fpath = os.path.join(cls_dir, fname)
            img   = cv2.imread(fpath)
            if img is None:
                continue
            feats = extract_features(img)
            X.append(feats)
            y.append(cls)
            filenames.append(fpath)

    return np.array(X, dtype=np.float32), np.array(y), filenames


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_model():
    print("\n========================================")
    print(" Training KNN Fruit Classifier")
    print("========================================")

    # Load training data
    print("\n[1/5] Loading training data ...")
    X_train, y_train, _ = load_dataset(TRAIN_DIR)
    print(f"      Training samples: {len(X_train)}")

    # Load validation data
    print("\n[2/5] Loading validation data ...")
    X_val, y_val, _ = load_dataset(VAL_DIR)
    print(f"      Validation samples: {len(X_val)}")

    # Encode labels
    le = LabelEncoder()
    le.fit(CLASSES)
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    # Hyperparameter search: find best k
    print("\n[3/5] Searching best k (1-15) ...")
    best_k, best_acc = 3, 0.0
    k_scores = {}
    for k in range(1, 16):
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='uniform')
        scores = cross_val_score(knn, X_train_sc, y_train_enc, cv=5, scoring='accuracy')
        k_acc  = scores.mean()
        k_scores[k] = k_acc
        print(f"      k={k:2d}  CV accuracy = {k_acc:.4f}")
        if k_acc > best_acc:
            best_acc, best_k = k_acc, k

    print(f"\n      -> Best k = {best_k}  (CV accuracy = {best_acc:.4f})")

    # Train final model
    print("\n[4/5] Training final KNN model ...")
    knn_final = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', weights='uniform')
    knn_final.fit(X_train_sc, y_train_enc)

    # Evaluate on validation set
    y_pred = knn_final.predict(X_val_sc)
    val_acc = accuracy_score(y_val_enc, y_pred)
    val_f1  = f1_score(y_val_enc, y_pred, average='weighted')
    val_prec = precision_score(y_val_enc, y_pred, average='weighted', zero_division=0)
    val_rec  = recall_score(y_val_enc, y_pred, average='weighted', zero_division=0)

    print(f"\n      Validation Accuracy  : {val_acc:.4f}")
    print(f"      Validation F1-Score  : {val_f1:.4f}")
    print(f"      Validation Precision : {val_prec:.4f}")
    print(f"      Validation Recall    : {val_rec:.4f}")
    print("\n      Classification Report:")
    print(classification_report(y_val_enc, y_pred, target_names=le.classes_))

    # Save charts
    print("\n[5/5] Saving charts ...")
    _save_confusion_matrix(y_val_enc, y_pred, le.classes_)
    _save_k_selection_chart(k_scores, best_k)

    # Persist model artefacts
    metrics = {
        'accuracy'  : float(val_acc),
        'f1_score'  : float(val_f1),
        'precision' : float(val_prec),
        'recall'    : float(val_rec),
        'best_k'    : best_k,
        'classes'   : list(le.classes_),
        'k_scores'  : {str(k): float(v) for k, v in k_scores.items()},
        'n_train'   : int(len(X_train)),
        'n_val'     : int(len(X_val)),
        'report'    : classification_report(y_val_enc, y_pred,
                                            target_names=le.classes_,
                                            output_dict=True),
    }

    joblib.dump(knn_final, os.path.join(MODEL_DIR, 'knn_model.pkl'))
    joblib.dump(scaler,    os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(le,        os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    joblib.dump(metrics,   os.path.join(MODEL_DIR, 'metrics.pkl'))

    print("\n  Model saved to ./model/")
    print("========================================\n")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_title('Confusion Matrix – KNN Fruit Classifier', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'confusion_matrix.png'), dpi=120, bbox_inches='tight')
    plt.close()


def _save_k_selection_chart(k_scores, best_k):
    ks     = list(k_scores.keys())
    accs   = [k_scores[k] for k in ks]
    colors = ['#f97316' if k == best_k else '#3b82f6' for k in ks]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ks, accs, color=colors, edgecolor='white', linewidth=1.5)
    ax.axhline(y=k_scores[best_k], color='#ef4444', linestyle='--', alpha=0.7,
               label=f'Best k={best_k}')
    ax.set_xlabel('Number of Neighbors (k)', fontsize=12)
    ax.set_ylabel('Cross-Validation Accuracy', fontsize=12)
    ax.set_title('K Selection – 5-Fold Cross Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(ks)
    ax.legend()
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'k_selection.png'), dpi=120, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helper (used by Flask)
# ─────────────────────────────────────────────────────────────────────────────
def predict_image(img_bgr):
    """Return (predicted_class, confidence, all_probs) for a single image."""
    model_path = os.path.join(MODEL_DIR, 'knn_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please train the model first.")

    knn    = joblib.load(model_path)
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    le     = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    feats     = np.array(extract_features(img_bgr), dtype=np.float32).reshape(1, -1)
    feats_sc  = scaler.transform(feats)

    pred_enc  = knn.predict(feats_sc)[0]
    pred_cls  = le.inverse_transform([pred_enc])[0]

    # Distance-based confidence
    distances, indices = knn.kneighbors(feats_sc)
    neighbor_labels    = knn.predict(scaler.transform(feats_sc)) # reuse
    # Build per-class vote
    all_labels = le.inverse_transform(knn._y[indices[0]])
    counts     = {cls: 0 for cls in CLASSES}
    for lbl in all_labels:
        if lbl in counts:
            counts[lbl] += 1
    total      = len(all_labels)
    probs      = {cls: counts[cls] / total for cls in CLASSES}
    confidence = probs[pred_cls]

    return pred_cls, confidence, probs


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_model()
