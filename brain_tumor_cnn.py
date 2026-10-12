import os
import random
import copy
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

from PIL import Image, ImageEnhance, ImageTk

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# Configuration
# ============================================================
import kagglehub
import os

DATASET_PATH = kagglehub.dataset_download(
    "navoneel/brain-mri-images-for-brain-tumor-detection"
)
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
SEED = 42

MODEL_PATH = "best_brain_tumor_cnn.pth"
FINAL_MODEL_PATH = "brain_tumor_cnn_final.pth"


# ============================================================
# Reproducibility and device
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Data augmentation
# ============================================================

def augment_image(image):
    """Apply mild augmentation to a training MRI image."""

    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        image = image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR
        )

    if random.random() < 0.3:
        factor = random.uniform(0.9, 1.1)
        image = ImageEnhance.Brightness(image).enhance(factor)

    if random.random() < 0.3:
        factor = random.uniform(0.9, 1.1)
        image = ImageEnhance.Contrast(image).enhance(factor)

    return image


# ============================================================
# Dataset
# ============================================================

class BrainTumorDataset(Dataset):

    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("L")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

        if self.augment:
            image = augment_image(image)

        array = np.array(image, dtype=np.float32) / 255.0

        tensor = torch.tensor(
            array,
            dtype=torch.float32
        )

        # [224, 224] -> [1, 224, 224]
        tensor = tensor.unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return tensor, label


# ============================================================
# CNN Model
# ============================================================

class BrainTumorCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2)

        # 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(
            128 * 28 * 28,
            128
        )

        self.dropout = nn.Dropout(0.5)

        # One output neuron for binary classification
        self.fc2 = nn.Linear(
            128,
            1
        )

    def forward(self, x):

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# ============================================================
# Dataset preparation
# ============================================================

def load_samples(dataset_path):
    no_tumor_path = os.path.join(dataset_path, "no")
    tumor_path = os.path.join(dataset_path, "yes")

    if not os.path.isdir(no_tumor_path):
        raise FileNotFoundError(
            f"NO TUMOR directory not found: {no_tumor_path}"
        )

    if not os.path.isdir(tumor_path):
        raise FileNotFoundError(
            f"TUMOR directory not found: {tumor_path}"
        )

    samples = []
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    for filename in os.listdir(no_tumor_path):
        if filename.lower().endswith(valid_extensions):
            samples.append(
                (os.path.join(no_tumor_path, filename), 0)
            )

    for filename in os.listdir(tumor_path):
        if filename.lower().endswith(valid_extensions):
            samples.append(
                (os.path.join(tumor_path, filename), 1)
            )

    if not samples:
        raise RuntimeError("No images found in the dataset.")

    return samples


def create_dataloaders(dataset_path):

    samples = load_samples(dataset_path)

    paths = [path for path, _ in samples]
    labels = [label for _, label in samples]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=0.30,
        random_state=SEED,
        stratify=labels
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels
    )

    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))
    test_samples = list(zip(test_paths, test_labels))

    train_dataset = BrainTumorDataset(
        train_samples,
        augment=True
    )

    val_dataset = BrainTumorDataset(
        val_samples,
        augment=False
    )

    test_dataset = BrainTumorDataset(
        test_samples,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Total images:", len(samples))
    print("Training:", len(train_samples))
    print("Validation:", len(val_samples))
    print("Testing:", len(test_samples))

    return train_loader, val_loader, test_loader


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= 0.5
        ).float()

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            probabilities = torch.sigmoid(outputs)

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def train_model():

    print("=" * 60)
    print("BRAIN TUMOR CLASSIFICATION")
    print("=" * 60)
    print("Device:", device)

    train_loader, val_loader, test_loader = create_dataloaders(
        DATASET_PATH
    )

    model = BrainTumorCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    best_val_loss = float("inf")

    print("\nStarting training...\n")

    for epoch in range(NUM_EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print("  -> Best model saved")

    print("\nLoading best model...")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    # Save a final copy as well
    torch.save(
        model.state_dict(),
        FINAL_MODEL_PATH
    )

    print("Best model:", MODEL_PATH)
    print("Final model:", FINAL_MODEL_PATH)

    return (
        model,
        test_loader,
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        best_val_loss
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, test_loader):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= 0.5
            ).int()

            all_labels.extend(
                labels.numpy().astype(int)
            )

            all_predictions.extend(
                predictions.cpu().numpy().flatten()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy().flatten()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\n" + "=" * 40)
    print("TEST RESULTS")
    print("=" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }

# ============================================================
# Single image prediction
# ============================================================

def predict_image(image_path, model):

    model.eval()

    image = Image.open(
        image_path
    ).convert("L")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    array = np.array(
        image,
        dtype=np.float32
    )

    array = array / 255.0

    tensor = torch.tensor(
        array,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():

        output = model(tensor)

        tumor_probability = torch.sigmoid(
            output
        ).item()

    if tumor_probability >= 0.5:

        predicted_class = "TUMOR"
        confidence = tumor_probability

    else:

        predicted_class = "NO TUMOR"
        confidence = 1.0 - tumor_probability

    return predicted_class, confidence


# ============================================================
# Tkinter GUI
# ============================================================

class BrainTumorApp:

    def __init__(self, root, model):

        self.root = root
        self.model = model

        self.root.title(
            "Brain Tumor Classification"
        )

        self.root.geometry(
            "900x700"
        )

        self.root.minsize(
            800,
            600
        )

        self.current_image = None

        # ---------- Header ----------

        header = tk.Frame(
            self.root,
            bg="#17202A",
            height=90
        )

        header.pack(
            fill="x"
        )

        title = tk.Label(
            header,
            text="Brain Tumor Classification",
            font=("Arial", 24, "bold"),
            fg="white",
            bg="#17202A"
        )

        title.pack(
            pady=(18, 2)
        )

        subtitle = tk.Label(
            header,
            text="CNN-based Brain MRI Analysis",
            font=("Arial", 11),
            fg="#D5D8DC",
            bg="#17202A"
        )

        subtitle.pack()

        # ---------- Main frame ----------

        main = tk.Frame(
            self.root,
            bg="#F4F6F7"
        )

        main.pack(
            fill="both",
            expand=True
        )

        # ---------- Image area ----------

        image_frame = tk.Frame(
            main,
            bg="white",
            bd=1,
            relief="solid"
        )

        image_frame.pack(
            padx=40,
            pady=25,
            fill="both",
            expand=True
        )

        self.image_label = tk.Label(
            image_frame,
            text="No MRI image selected",
            font=("Arial", 16),
            fg="#7B7D7D",
            bg="white"
        )

        self.image_label.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20
        )

        # ---------- Buttons ----------

        button_frame = tk.Frame(
            main,
            bg="#F4F6F7"
        )

        button_frame.pack(
            pady=5
        )

        open_button = tk.Button(
            button_frame,
            text="Open MRI Image",
            command=self.open_image,
            font=("Arial", 13, "bold"),
            padx=20,
            pady=10,
            cursor="hand2"
        )

        open_button.grid(
            row=0,
            column=0,
            padx=8
        )

        clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear,
            font=("Arial", 13),
            padx=20,
            pady=10,
            cursor="hand2"
        )

        clear_button.grid(
            row=0,
            column=1,
            padx=8
        )

        # ---------- Result ----------

        self.result_label = tk.Label(
            main,
            text="Upload an MRI image to begin",
            font=("Arial", 18, "bold"),
            fg="#2C3E50",
            bg="#F4F6F7"
        )

        self.result_label.pack(
            pady=(10, 2)
        )

        self.confidence_label = tk.Label(
            main,
            text="",
            font=("Arial", 13),
            fg="#566573",
            bg="#F4F6F7"
        )

        self.confidence_label.pack(
            pady=(0, 20)
        )

        # ---------- Footer ----------

        footer = tk.Label(
            self.root,
            text="For educational/project use only — not a medical diagnosis.",
            font=("Arial", 9),
            fg="#7B7D7D",
            bg="#F4F6F7"
        )

        footer.pack(
            side="bottom",
            pady=8
        )

    def open_image(self):

        file_path = filedialog.askopenfilename(
            title="Select Brain MRI Image",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.bmp"
                ),
                (
                    "All files",
                    "*.*"
                )
            ]
        )

        if not file_path:
            return

        try:

            image = Image.open(
                file_path
            ).convert("L")

            self.display_image(image)

            predicted_class, confidence = predict_image(
                file_path,
                self.model
            )

            self.result_label.config(
                text=f"Prediction: {predicted_class}",
                fg=(
                    "#C0392B"
                    if predicted_class == "TUMOR"
                    else "#27AE60"
                )
            )

            self.confidence_label.config(
                text=f"Confidence: {confidence:.2%}"
            )

        except Exception as error:

            messagebox.showerror(
                "Prediction Error",
                f"Could not process the image.\n\n{error}"
            )

    def display_image(self, image):

        display = image.copy()

        display.thumbnail(
            (650, 420)
        )

        self.current_image = ImageTk.PhotoImage(
            display
        )

        self.image_label.config(
            image=self.current_image,
            text=""
        )

    def clear(self):

        self.current_image = None

        self.image_label.config(
            image="",
            text="No MRI image selected"
        )

        self.result_label.config(
            text="Upload an MRI image to begin",
            fg="#2C3E50"
        )

        self.confidence_label.config(
            text=""
        )


# ============================================================
# GUI launcher
# ============================================================

def launch_gui(model):

    root = tk.Tk()

    BrainTumorApp(
        root,
        model
    )

    root.mainloop()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Brain Tumor CNN classifier"
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI using an existing trained model"
    )

    args = parser.parse_args()

    if args.gui:

        if not os.path.exists(MODEL_PATH):

            print(
                f"Model file '{MODEL_PATH}' was not found."
            )

            print(
                "Run the script normally first to train the model."
            )

            return

        model = BrainTumorCNN().to(device)

        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=device
            )
        )

        print("Loaded:", MODEL_PATH)
        print("Starting Tkinter GUI...")

        launch_gui(model)

        return

    # Train model
    (
        model,
        test_loader,
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        best_val_loss
    ) = train_model()

    # Evaluate
    results = evaluate_model(
        model,
        test_loader
    )
    # Launch application
    print("\nStarting Tkinter application...")
    launch_gui(model)


if __name__ == "__main__":
    main()
