# ============================================================
# 1. Install & Import Library
# ============================================================
!pip install spikingjelly==0.0.0.0.14 torchmetrics scikit-learn matplotlib

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score

# ============================================================
# 2. Mount Google Drive
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

data_dir = "/content/drive/MyDrive/Identifikasi Retina/Cataract/Dataset"  # Update path dataset sesuai lokasi Anda

# ============================================================
# 3. Data Transform & Loader
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
test_dataset  = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 4. Model CNN (Pembanding)
# ============================================================
cnn_model = models.resnet18(pretrained=False)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
cnn_model = cnn_model.to(device)

# ============================================================
# 5. Training Function with Accuracy & Confusion Matrix
# ============================================================
def train_model(model, train_loader, optimizer, criterion, epochs=5, label="Model"):
    history = []
    model.train()
    all_train_true, all_train_pred = [], []
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_train_true.extend(targets.cpu().numpy())
            all_train_pred.extend(predicted.cpu().numpy())

        acc = 100. * correct / total
        print(f"{label} - Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} Acc: {acc:.2f}%")
        history.append(acc)

        # Confusion Matrix for Training
        cm_train = confusion_matrix(all_train_true, all_train_pred)
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{label} Training Confusion Matrix - Epoch {epoch+1}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    return history

# ============================================================
# 6. Testing + Metrics Function with Accuracy
# ============================================================
def test_model_metrics(model, test_loader, label="Model"):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    acc_test = 100. * correct / total
    print(f"{label} Test Accuracy: {acc_test:.2f}%")

    precision = BinaryPrecision()(torch.tensor(y_pred), torch.tensor(y_true))
    recall    = BinaryRecall()(torch.tensor(y_pred), torch.tensor(y_true))
    f1        = BinaryF1Score()(torch.tensor(y_pred), torch.tensor(y_true))

    print(f"\n{label} Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Confusion Matrix for Test
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{label} Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{label} ROC Curve")
    plt.legend()
    plt.show()

    print("\nClassification Report:\n", classification_report(y_true, y_pred))

# ============================================================
# 7. Train & Test CNN
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer_cnn = optim.Adamax(cnn_model.parameters(), lr=0.00005)  # Using Adamax instead of Adam

# Training the model
cnn_history = train_model(cnn_model, train_loader, optimizer_cnn, criterion, epochs=50, label="CNN")

# Testing the model
test_model_metrics(cnn_model, test_loader, label="CNN")
