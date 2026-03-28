import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os
import cv2

from collections import Counter, defaultdict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, roc_curve, average_precision_score, cohen_kappa_score)


# ----------------< One-Time Preprocessing Module >---------------------------------------------------------------------

def preprocess_dataset_once(input_dir, output_dir):
    print("Starting one-time dataset preprocessing (PNG Lossless)...")

    for split in ['train', 'val', 'test']:
        split_input  = os.path.join(input_dir, split)
        split_output = os.path.join(output_dir, split)

        if not os.path.exists(split_input):
            continue

        for class_name in sorted(os.listdir(split_input)):
            class_input_dir  = os.path.join(split_input,  class_name)
            class_output_dir = os.path.join(split_output, class_name)
            os.makedirs(class_output_dir, exist_ok=True)

            images = [f for f in os.listdir(class_input_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for img_name in images:
                input_path  = os.path.join(class_input_dir, img_name)
                output_path = os.path.join(class_output_dir,
                                           os.path.splitext(img_name)[0] + '.png')
                try:
                    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
                    img   = clahe.apply(img)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    cv2.imwrite(output_path, img_rgb)
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

    print("Preprocessing complete!")


# ----------------< Custom Dataset Module >-----------------------------------------------------------------------------

class XrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.samples   = []
        self.classes   = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((
                            os.path.join(class_dir, img_name),
                            self.class_to_idx[class_name]
                        ))
        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ----------------< Focal Loss Module >---------------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.reduction       = reduction

    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            with torch.no_grad():
                true_dist = torch.zeros_like(inputs)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce_loss = -(true_dist * F.log_softmax(inputs, dim=1)).sum(dim=1)
            if self.alpha is not None:
                ce_loss = ce_loss * self.alpha[targets]
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            if self.alpha is not None:
                ce_loss = ce_loss * self.alpha[targets]

        pt          = torch.exp(-ce_loss)
        focal_loss  = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ----------------< Class Weights Module >------------------------------------------------------------------------------

def calculate_class_weights(dataset, device):
    class_counts = Counter(dataset.targets)
    total        = len(dataset)
    num_classes  = len(class_counts)
    weights      = torch.zeros(num_classes)
    for i in range(num_classes):
        weights[i] = total / (num_classes * class_counts[i])
    return weights.to(device)


# ----------------< Metrics Module >------------------------------------------------------------------------------------

def compute_and_save_metrics(model, test_loader, class_names, checkpoint_dir,
                              train_losses, val_losses, train_accs, val_accs,
                              lr_history, device):
    print("\n" + "=" * 80)
    print("METRICS MODULE — FULL TEST SET EVALUATION")
    print("=" * 80)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs   = F.softmax(outputs, dim=1)

            all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs) 

    # ── Accuracy ──────────────────────────────────────────────────────
    overall_acc = 100 * np.mean(all_preds == all_labels)
    print(f"\nOverall Test Accuracy : {overall_acc:.2f}%")

    # ── Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                 target_names=class_names, digits=4))

    # ── Cohen's Kappa ─────────────────────────────────────────────────
    kappa = cohen_kappa_score(all_labels, all_preds)
    print(f"Cohen's Kappa         : {kappa:.4f}")

    # ── Per-Class AUC (One-vs-Rest) ───────────────────────────────────
    print("\nPer-Class AUC (One vs Rest):")
    auc_scores = {}
    for i, class_name in enumerate(class_names):
        binary_labels    = (all_labels == i).astype(int)
        auc              = roc_auc_score(binary_labels, all_probs[:, i])
        auc_scores[class_name] = auc
        print(f"  {class_name:<20} AUC = {auc:.4f}")

    mean_auc = np.mean(list(auc_scores.values()))
    print(f"  {'Mean AUC':<20}     = {mean_auc:.4f}")

    # ── Per-Class Sensitivity & Specificity ───────────────────────────
    print("\nPer-Class Sensitivity & Specificity:")
    for i, class_name in enumerate(class_names):
        tp = np.sum((all_preds == i) & (all_labels == i))
        fn = np.sum((all_preds != i) & (all_labels == i))
        tn = np.sum((all_preds != i) & (all_labels != i))
        fp = np.sum((all_preds == i) & (all_labels != i))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  {class_name:<20} Sensitivity={sensitivity:.4f} | Specificity={specificity:.4f}")

    # ── Plot 1: Confusion Matrix ───────────────────────────────────────
    cm         = confusion_matrix(all_labels, all_preds)
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — Overall Accuracy: {overall_acc:.2f}%")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "confusion_matrix.png"), dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")

    # ── Plot 2: ROC Curves Per Class ───────────────────────────────────
    colors = ['#c0c8d0', '#ef4444', '#06b6d4', '#10b981', '#f59e0b']
    plt.figure(figsize=(10, 7))
    for i, class_name in enumerate(class_names):
        binary_labels = (all_labels == i).astype(int)
        fpr, tpr, _   = roc_curve(binary_labels, all_probs[:, i])
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'{class_name} (AUC={auc_scores[class_name]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — Per Class (One vs Rest)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "roc_curves.png"), dpi=150)
    plt.show()
    print("Saved: roc_curves.png")

    # ── Plot 3: Training Curves (Loss + Accuracy) ──────────────────────
    if train_losses:
        epochs_ran = len(train_losses)
        fig, axes  = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(range(1, epochs_ran + 1), train_losses,
                     label='Train Loss', color='#2563eb')
        axes[0].plot(range(1, epochs_ran + 1), val_losses,
                     label='Val Loss',   color='#06b6d4', linestyle='--')
        axes[0].set_title('Loss Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        axes[1].plot(range(1, epochs_ran + 1), train_accs,
                     label='Train Acc', color='#10b981')
        axes[1].plot(range(1, epochs_ran + 1), val_accs,
                     label='Val Acc',   color='#f59e0b', linestyle='--')
        axes[1].set_title('Accuracy Curves')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, "training_curves.png"), dpi=150)
        plt.show()
        print("Saved: training_curves.png")

    # ── Plot 4: LR Schedule ────────────────────────────────────────────
    if lr_history:
        plt.figure(figsize=(10, 4))
        plt.plot(lr_history, color='#f59e0b', lw=1.5)
        plt.title('OneCycleLR Schedule (Cosine Annealing Strategy)')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, "lr_schedule.png"), dpi=150)
        plt.show()
        print("Saved: lr_schedule.png")

    print(f"\nAll plots saved to: {checkpoint_dir}")
    print(f"\nSummary — Accuracy: {overall_acc:.2f}% | Mean AUC: {mean_auc:.4f} | Kappa: {kappa:.4f}")


# ----------------< Grad-CAM Module >-----------------------------------------------------------------------------------

def run_gradcam(model, test_dataset, val_transform, class_names, checkpoint_dir, device):
    """
    Generates top-5 Grad-CAM visualizations per class using SmoothGradCAMpp.
    Color-coded per pathology class. Saves grid to checkpoint_dir.
    """

    print("\n" + "=" * 80)
    print("GRAD-CAM MODULE")
    print("=" * 80)

    num_classes = len(class_names)
    model.eval()
    class_best = defaultdict(list)

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            img_path, true_label = test_dataset.samples[idx]
            original_pil  = Image.open(img_path).convert("RGB")
            input_tensor  = val_transform(original_pil).unsqueeze(0).to(device)
            out           = model(input_tensor)
            probs         = F.softmax(out, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

            if pred_idx.item() == true_label:
                class_best[true_label].append((conf.item(), idx))

    # Top 5 per class by confidence
    selected_indices = []
    for class_idx in range(num_classes):
        top5 = sorted(class_best[class_idx], reverse=True)[:25]
        selected_indices.extend([idx for _, idx in top5])

    color_map = {
        'No Finding':   'bone',
        'effusion':     'CMRmap',
        'atelectasis':  'viridis',
        'mass':         'hot',
        'pneumothorax': 'jet'
    }

    cam_extractor    = SmoothGradCAMpp(model, target_layer='features.7')
    all_visualizations = []

    for idx in selected_indices:
        img_path, _ = test_dataset.samples[idx]
        original_pil = Image.open(img_path).convert("RGB")
        viz_original = original_pil.resize((512, 512))
        input_tensor = val_transform(original_pil).unsqueeze(0).to(device)

        out              = model(input_tensor)
        probs            = F.softmax(out, dim=1)
        conf, pred_idx   = torch.max(probs, dim=1)
        pred_class       = test_dataset.classes[pred_idx.item()]

        act_map = cam_extractor(pred_idx.item(), out)
        cmap    = color_map.get(pred_class, 'jet')

        result_pil = overlay_mask(
            viz_original,
            to_pil_image(act_map[0].squeeze(0), mode='F'),
            alpha=0.5,
            colormap=cmap
        )
        combined = np.hstack((np.array(viz_original), np.array(result_pil)))
        all_visualizations.append((combined, f"Pred: {pred_class} ({conf.item():.2%})"))

    fig, axes = plt.subplots(5, 25, figsize=(100, 20))
    axes = axes.flatten()
    for i, (img, title) in enumerate(all_visualizations):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "gradcam_grid.png"), dpi=150)
    plt.show()
    cam_extractor.remove_hooks()
    print("Saved: gradcam_grid.png")


# ----------------< Environment + Training Module >---------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU  : {torch.cuda.get_device_name(0)}")
        print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA not available — using CPU")

    raw_data_dir          = r"C:\Users\nexus\Documents\Radiology_v5"
    preprocessed_data_dir = r"C:\Users\nexus\Documents\Radiology_v5_preprocessed"
    checkpoint_dir        = r"C:\Users\nexus\Documents\Radiology_v5\checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(preprocessed_data_dir):
        print("Preprocessed data not found — running one-time preprocessing...")
        preprocess_dataset_once(raw_data_dir, preprocessed_data_dir)

    start_time = time.time()

    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = XrayDataset(f"{preprocessed_data_dir}/train", transform=train_transform)
    val_dataset   = XrayDataset(f"{preprocessed_data_dir}/val",   transform=val_transform)
    test_dataset  = XrayDataset(f"{preprocessed_data_dir}/test",  transform=val_transform)

    class_names  = train_dataset.classes
    num_classes  = len(class_names)
    print(f"\nClasses : {class_names}")
    print(f"Train   : {len(train_dataset)} images")
    print(f"Val     : {len(val_dataset)} images")
    print(f"Test    : {len(test_dataset)} images")

    train_class_dist = Counter(train_dataset.targets)
    print("\nTraining set distribution:")
    for i, name in enumerate(class_names):
        count = train_class_dist[i]
        print(f"  {name}: {count} ({100 * count / len(train_dataset):.1f}%)")

    class_weights = calculate_class_weights(train_dataset, device)
    multipliers   = torch.tensor([0.5, 2.5, 1.5, 3.0, 1.0]).to(device)
    class_weights = class_weights * multipliers
    print(f"\nAdjusted class weights: {class_weights.cpu().numpy()}")

    pin_memory   = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=4, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=4, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False, num_workers=4)

    # ----------------< Model Setup Module >----------------------------------------------------------------------------

    model     = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    num_ftrs  = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)

    for param in model.features.parameters():
        param.requires_grad = True

    criterion  = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.01)
    optimizer  = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    num_epochs = 1
    scheduler  = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-4, epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3, anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler('cuda')

    # ----------------< Checkpoint Resume Module >----------------------------------------------------------------------

    checkpoint_path = os.path.join(checkpoint_dir, "radiology_model_checkpoint.pth")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    start_epoch     = 0

    # Metric history lists to continue across resumed runs
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    lr_history               = []

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch  = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        val_losses   = checkpoint.get('val_losses',   [])
        train_accs   = checkpoint.get('train_accs',   [])
        val_accs     = checkpoint.get('val_accs',     [])
        lr_history   = checkpoint.get('lr_history',   [])
        print(f"Resuming from epoch {start_epoch}")

    # ----------------< Early Stopping Module >-------------------------------------------------------------------------

    best_val_acc    = 0.0
    best_val_loss   = float('inf')
    patience_counter = 0
    patience         = 15

    # ----------------< Training Module >-------------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TRAINING START")
    print("=" * 80)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss   = 0.0
        correct_train  = 0
        total_train    = 0
        all_train_preds, all_train_labels = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Track LR every step
            lr_history.append(scheduler.get_last_lr()[0])

            running_loss  += loss.item() * inputs.size(0)
            _, predicted   = torch.max(outputs, 1)
            total_train   += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        epoch_loss     = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_loss      = 0.0
        correct       = 0
        total         = 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs        = model(inputs)
                loss           = criterion(outputs, labels)
                val_loss      += loss.item() * inputs.size(0)
                _, predicted   = torch.max(outputs, 1)
                total         += labels.size(0)
                correct       += (predicted == labels).sum().item()
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        avg_val_loss   = val_loss / len(val_loader.dataset)
        val_accuracy   = 100 * correct / total

        # Append to metric history
        train_losses.append(epoch_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss  : {avg_val_loss:.4f} | Val Acc  : {val_accuracy:.2f}%")

        all_val_preds  = np.array(all_val_preds)
        all_val_labels = np.array(all_val_labels)
        print("Per-class Val Acc:", end=" ")
        for i, name in enumerate(class_names):
            mask = all_val_labels == i
            if mask.sum() > 0:
                acc = 100 * np.mean(all_val_preds[mask] == all_val_labels[mask])
                print(f"{name}: {acc:.1f}%", end=" | ")
        print()

        # ----------------< Save Best Module >---------------------------------------------------------------------------
        if val_accuracy > best_val_acc:
            best_val_acc  = val_accuracy
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'val_acc':             val_accuracy,
                'loss':                epoch_loss,
                'train_losses':        train_losses,
                'val_losses':          val_losses,
                'train_accs':          train_accs,
                'val_accs':            val_accs,
                'lr_history':          lr_history,
            }, best_model_path)
            print(f"✓ New best model saved — Val Acc: {val_accuracy:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                print(f"Best Val Acc: {best_val_acc:.2f}% | Best Val Loss: {best_val_loss:.4f}")
                break

        # ----------------< Rolling Checkpoint Module >-----------------------------------------------------------------
        torch.save({
            'epoch':               epoch,
            'model_state_dict':    model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':                epoch_loss,
            'train_losses':        train_losses,
            'val_losses':          val_losses,
            'train_accs':          train_accs,
            'val_accs':            val_accs,
            'lr_history':          lr_history,
        }, checkpoint_path)

    # ----------------< Load Best Model Module >------------------------------------------------------------------------

    print("\nLoading best model for evaluation...")
    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    # Restore metric history from best checkpoint in case early stopping cut the run
    train_losses = best_checkpoint.get('train_losses', train_losses)
    val_losses   = best_checkpoint.get('val_losses',   val_losses)
    train_accs   = best_checkpoint.get('train_accs',   train_accs)
    val_accs     = best_checkpoint.get('val_accs',     val_accs)
    lr_history   = best_checkpoint.get('lr_history',   lr_history)

    # ----------------< Metrics Module >-------------------------------------------------------------------------------

    compute_and_save_metrics(
        model       = model,
        test_loader = test_loader,
        class_names = class_names,
        checkpoint_dir = checkpoint_dir,
        train_losses = train_losses,
        val_losses   = val_losses,
        train_accs   = train_accs,
        val_accs     = val_accs,
        lr_history   = lr_history,
        device       = device
    )

    # ----------------< Grad-CAM Module >------------------------------------------------------------------------------

    run_gradcam(
        model        = model,
        test_dataset = test_dataset,
        val_transform= val_transform,
        class_names  = class_names,
        checkpoint_dir = checkpoint_dir,
        device       = device
    )

    # ----------------< Summary Module >-------------------------------------------------------------------------------

    print(f"\n" + "=" * 80)
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes")
    print(f"Best validation accuracy : {best_val_acc:.2f}%")
    print(f"Checkpoints saved to     : {checkpoint_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
