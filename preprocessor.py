import torchvision.transforms as transforms
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torchvision import models
import numpy as np
from tqdm import tqdm


# ----------------------- Data Loading and Preprocessing ----------------------- #
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_df = pd.read_csv('data/train_synth_data.csv')
val_df = pd.read_csv('data/val_data.csv')

# ----------------------- Tabular Data Preprocessing ----------------------- #
print("Preprocessing tabular data...")
numeric_features = ['age']
categorical_features = ['sex', 'localization']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ], remainder='drop'
    )

train_meta_processed = preprocessor.fit_transform(train_df)
val_meta_processed = preprocessor.transform(val_df)

joblib.dump(preprocessor, 'models/tabular_preprocessor.pkl')
print("Preprocessor saved to 'models/tabular_preprocessor.pkl'")

# ----------------------- Image Transformation ----------------------- #
print("Setting up image transformations...")
class CustomDataset(Dataset):
    def __init__(self, df, processed_meta, image_transform=None):
        self.df = df
        self.processed_meta = processed_meta
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = f'data/{self.df.iloc[idx]['path']}'
        image = Image.open(image_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        
        meta = torch.tensor(self.processed_meta[idx].toarray().flatten(), dtype=torch.float32)
        
        label = torch.tensor(self.df.iloc[idx]['target'], dtype=torch.long)

        return {
            'image': image,
            'meta': meta,
            'label': label  
        }

train_dataset = CustomDataset(train_df, train_meta_processed, image_transform=image_transform)
val_dataset = CustomDataset(val_df, val_meta_processed, image_transform=image_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ---------------------------- NEURAL NETWORK ---------------------------- #
print("Setting up the neural network...")
class MultiInputNN(nn.Module):
    def __init__(self, num_meta_features, num_classes):
        super(MultiInputNN, self).__init__()

        # Image branch (CNN)
        self.image_branch = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        for param in self.image_branch.parameters():
            param.requires_grad = False
        
        num_ftrs = self.image_branch.classifier[1].in_features
        self.image_branch.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 128)
        )

        # Metadata branch (Fully Connected)
        self.meta_branch = nn.Sequential(
            nn.Linear(num_meta_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )

        # Combined layers
        self.classifier_head = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, meta):
        image_features = self.image_branch(image)
        meta_features = self.meta_branch(meta)
        combined = torch.cat((image_features, meta_features), dim=1)
        output = self.classifier_head(combined)
        return output
    
num_meta_features = train_meta_processed.shape[1]
num_classes = len(train_df['target'].unique())

model = MultiInputNN(num_meta_features, num_classes)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
PATIENCE = 5
best_val_loss = np.inf
patience_counter = 0
model_checkpoint_path = 'models/best_model.pth'

# ---------------------------- TRAINING LOOP ---------------------------- #
print("Starting training...")
model.to(device)
for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
        images = batch['image'].to(device)
        metas = batch['meta'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images, metas)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() 
    epoch_loss = running_loss / len(train_loader)

    # ---------------------------- VALIDATION LOOP ---------------------------- #
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
            images = batch['image'].to(device)
            metas = batch['meta'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, metas)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = correct / total if total > 0 else 0
    val_acc = val_accuracy * 100

    print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), model_checkpoint_path)
        print(f"✅ Validation loss improved to {best_val_loss:.4f}. Saving model to {model_checkpoint_path}")
    else:
        patience_counter += 1
        print(f"⚠️ No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping triggered after {epoch + 1} epochs.")
            break # Exit the training loop

print("Training finished.")