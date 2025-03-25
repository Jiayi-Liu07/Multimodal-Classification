import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from OCTDataset import OCTDataset
import matplotlib.pyplot as plt
#import timm
import torchvision.models as models

# ===============================
# 1. Image Model (RETFound Encoder)
# ===============================
# class RETFoundEncoder(nn.Module):
#     def __init__(self, model_name="timm/retfound"):
#         super(RETFoundEncoder, self).__init__()
#         # Load the RETFound model and remove the classification head
#         self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
#         self.feature_dim = self.model.num_features

#     def forward(self, x):
#         return self.model(x)  # Return image features

# ===============================

class ResNetClassifier(nn.Module):
    def __init__(self):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)

    def forward(self, x):
        return self.model(x)

# ===============================
# 2. Multiclass Classification Module
# ===============================
class MulticlassClassifier(nn.Module):
    def __init__(self, image_feature_dim, num_classes):
        super(MulticlassClassifier, self).__init__()
        self.fc = nn.Linear(image_feature_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

# ===============================
# 6. Training Function
# ===============================
def train_model(model, image_encoder, train_loader, val_loader, criterion, optimizer, device):
    train_losses, val_losses = [], []
    for epoch in range(10):
        model.train()
        total_loss, correct, total_auc = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                img_features = image_encoder(images)  # [batch, image_feature_dim]
            outputs = model(img_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            auc = roc_auc_score(labels.cpu().numpy(), torch.softmax(outputs, dim=1).cpu().numpy(), multi_class='ovr')
            total_auc += auc
        accuracy = correct / len(train_loader.dataset)
        auc_score = total_auc / len(train_loader)
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        correct = 0
        total_auc = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                img_features = image_encoder(images)  # [batch, image_feature_dim]
                outputs = model(img_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                auc = roc_auc_score(labels.cpu().numpy(), torch.softmax(outputs, dim=1).cpu().numpy(), multi_class='ovr')
                total_auc += auc
        val_accuracy = correct / len(val_loader.dataset)
        val_auc_score = total_auc / len(val_loader)
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Accuracy = {accuracy:.4f}, Train AUC = {auc_score:.4f}')
        print(f'Epoch {epoch+1}: Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, Val AUC = {val_auc_score:.4f}')

    return train_losses, val_losses

# ===============================
# 7. Evaluation Function
# ===============================
def evaluate(model, image_encoder, test_loader, device):
    model.eval()
    correct = 0
    total_auc = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            img_features = image_encoder(images)  # [batch, image_feature_dim]
            outputs = model(img_features)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            auc = roc_auc_score(labels.cpu().numpy(), torch.softmax(outputs, dim=1).cpu().numpy(), multi_class='ovr')
            total_auc += auc
    accuracy = correct / len(test_loader.dataset)
    auc_score = total_auc / len(test_loader)
    return accuracy, auc_score

# ===============================
# 8. Main Program
# ===============================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    dataset = OCTDataset(root_dir='path_to_data', label_path='path_to_labels')

    # Train-val-test split
    train_idx, val_test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model and optimizer
    image_encoder = ResNetClassifier()
    multiclass_classifier = MulticlassClassifier(image_encoder.model.fc.in_features, num_classes=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(multiclass_classifier.parameters(), lr=1e-4)

    # Train the model
    train_losses, val_losses = train_model(multiclass_classifier, image_encoder, train_loader, val_loader, criterion, optimizer, device)

    # Plot train and val loss
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.show()

    # Evaluate on test set
    test_accuracy, test_auc_score = evaluate(multiclass_classifier, image_encoder, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test AUC Score: {test_auc_score:.4f}')

    # Display hyperparameters
    print('Hyperparameters:')
    print(f'Batch Size: {16}')
    print(f'Number of Epochs: {10}')
    print(f'Learning Rate: {1e-4}')
    print(f'Optimizer: Adam')
    print(f'Model: RETFoundEncoder + MulticlassClassifier')