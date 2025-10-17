import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import os
import re
from collections import defaultdict
import cv2

from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Custom Dataset for tropical fruit images and spectral data
class tropic_dataset(Dataset):
    def __init__(self, num_days, transform, mode, df):
        """
        Args:
            num_days (int): The number of days used to discretize the labels.
            transform (callable): A function/transform to apply to the images.
            mode (str): 'train' or 'test' to specify the dataset type.
            df (DataFrame): Pandas DataFrame containing the spectral data.
        """
        self.transform = transform
        self.mode = mode
        self.root_train = 'Directory where the processed training images will be saved'
        self.root_test = 'Directory where the processed test images will be saved'
        self.num_days = num_days
        self.df = df

        if mode == 'test':
            self.image_folder = self.root_test
            self.labels, self.texts = get_labels_text(self.image_folder, self.num_days, self.df)
            self.image_paths = [os.path.join(self.image_folder, fname) for fname in self.labels.keys()]
        else: # train mode
            self.image_folder = self.root_train
            self.labels, self.texts = get_labels_text(self.image_folder, self.num_days, self.df)
            self.image_paths = [os.path.join(self.image_folder, fname) for fname in self.labels.keys()]
            
        print(f"{self.mode} data has a size of {len(self.image_paths)}")

    def __getitem__(self, index):
        """
        Returns a single sample from the dataset.
        """
        img_path = self.image_paths[index]
        # Use os.path.basename for robustly getting the filename
        filename = os.path.basename(img_path)
        
        target = self.labels[filename]
        text_data = self.texts[filename]
        
        image = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(image)
        
        return img_tensor, target, text_data

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_paths)

def get_labels_text(image_folder_path, num_days, df):
    """
    Parses filenames to generate labels and retrieve corresponding spectral text data.
    The label represents the remaining days to maturity, bucketed into classes.
    """
    max_days_per_id = defaultdict(int)
    image_labels = {}
    image_text = {}

    # Regex to parse image_id, day, and view from filenames like '10_1_1.jpg'
    pattern = re.compile(r'(\d+)_(\d+)_(\d+)\.jpg')

    # First pass: find the maximum observation day for each fruit ID
    for filename in os.listdir(image_folder_path):
        match = pattern.match(filename)
        if match:
            image_id, day, _ = map(int, match.groups())
            if day > max_days_per_id[image_id]:
                max_days_per_id[image_id] = day

    # Second pass: calculate labels and fetch spectral data
    for filename in os.listdir(image_folder_path):
        match = pattern.match(filename)
        if match:
            image_id, day, _ = map(int, match.groups())
            
            # Create a search key like '10-1' for the dataframe
            search_value = f"{image_id}-{day}"
            result = df.loc[df['1'] == search_value, 'Combined'].values
            
            # The label is the number of remaining days, divided into bins
            label = max_days_per_id[image_id] - day
            image_labels[filename] = int(label / num_days)
            
            # Store the spectral data as a string
            if len(result) > 0:
                image_text[filename] = str(result[0])
            else:
                image_text[filename] = "" # Handle cases where data might be missing

    return image_labels, image_text


# Define the multi-modal fusion model
class MultiModalResNet(nn.Module):
    def __init__(self, resnet, num_classes, bert_feature_dim):
        super(MultiModalResNet, self).__init__()
        self.resnet = resnet
        self.bert_feature_dim = bert_feature_dim
        
        # Get the number of input features from ResNet's original classifier
        num_ftrs = self.resnet.fc.in_features
        
        # Replace the ResNet classifier with an Identity layer to extract features
        self.resnet.fc = nn.Identity()

        # Define a new classifier for the concatenated image and text features
        self.fc = nn.Linear(num_ftrs + self.bert_feature_dim, num_classes)

    def forward(self, image, bert_features):
        # Extract features from the image using ResNet
        resnet_features = self.resnet(image)

        # Concatenate image features and BERT text features
        combined_features = torch.cat((resnet_features, bert_features), dim=1)
        
        # Classify the combined features
        out = self.fc(combined_features)
        
        return out
    
def evaluate_model(model, bert_model, tokenizer, test_loader, criterion, device, epoch, num_days, file_path='resnet18_mm_evaluation_results.txt'):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    running_mse = 0.0
    running_mae = 0.0

    progress_bar = tqdm(test_loader, desc=f'Evaluating Epoch {epoch}')
    with torch.no_grad():
        for images, labels, texts in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Tokenize text and get BERT features
            inputs = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = bert_model(**inputs)
            cls_features = outputs.last_hidden_state[:, 0, :]

            # Get model predictions
            predictions = model(images, cls_features)
            loss = criterion(predictions, labels)
            _, preds_class = torch.max(predictions, 1)

            # Update metrics
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds_class == labels.data)
            
            mse = mean_squared_error(labels.cpu().numpy(), preds_class.cpu().numpy())
            mae = mean_absolute_error(labels.cpu().numpy(), preds_class.cpu().numpy())
            running_mse += mse * images.size(0)
            running_mae += mae * images.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), mse=mse, mae=mae)

    acc = running_corrects.double() / len(test_loader.dataset)
    mse = running_mse / len(test_loader.dataset)
    mae = running_mae / len(test_loader.dataset)

    print(f'Test Acc: {acc:.4f}, Test MSE: {mse:.4f}, Test MAE: {mae:.4f}')

    # Log results to a file
    with open(f'day{num_days}_{file_path}', 'a') as f:
        f.write(f'Epoch {epoch}: Test Acc: {acc:.4f}, Test MSE: {mse:.4f}, Test MAE: {mae:.4f}\n')

    return mae


def train_model(model, bert_model, tokenizer, train_loader, test_loader, criterion, optimizer, device, num_days, num_epochs=10):
    best_mae = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_mse = 0.0
        running_mae = 0.0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
        for images, labels, texts in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Tokenize text and get BERT features (with no gradient calculation for BERT)
            inputs = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = bert_model(**inputs)
            cls_features = outputs.last_hidden_state[:, 0, :]

            # Forward pass through the fusion model
            predictions = model(images, cls_features)
            loss = criterion(predictions, labels)
            _, preds_class = torch.max(predictions, 1)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running metrics
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds_class == labels.data)
            
            mse = mean_squared_error(labels.cpu().numpy(), preds_class.cpu().numpy())
            mae = mean_absolute_error(labels.cpu().numpy(), preds_class.cpu().numpy())
            running_mse += mse * images.size(0)
            running_mae += mae * images.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), mse=mse, mae=mae)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        epoch_mse = running_mse / len(train_loader.dataset)
        epoch_mae = running_mae / len(train_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, MSE: {epoch_mse:.4f}, MAE: {epoch_mae:.4f}')

        # Evaluate the model on the test set
        current_mae = evaluate_model(model, bert_model, tokenizer, test_loader, criterion, device, epoch, num_days)

        # Save the model if it has the best MAE so far
        if current_mae < best_mae:
            best_mae = current_mae
            torch.save(model.state_dict(), f'best_mm_resnet18_day{num_days}.pth')
            print(f'Saved new best model with MAE: {best_mae:.4f}')


if __name__ == '__main__':
    # --- 1. Data Preprocessing for Spectral Data ---
    df = pd.read_csv('Path to your Spectrum Data', skiprows=1, header=None)
    df = df.iloc[:, 2:]
    df.columns = [f'{i}' for i in range(1, df.shape[1] + 1)]
    # Combine all spectral columns into a single space-separated string
    df['Combined'] = df.apply(lambda row: ' '.join(row[1:].astype(str)), axis=1)
    df = df[['1', 'Combined']]

    # --- 2. Setup Models and Tokenizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model_path = 'Path to your Bert Model'

    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    bert_model = BertModel.from_pretrained(local_model_path)
    # Move BERT to the selected device and set it to evaluation mode
    bert_model = bert_model.to(device)
    bert_model.eval()

    # --- 3. Dataset and DataLoader Configuration ---
    transform_tropic = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # These are likely dataset-specific normalization values
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
    ])

    num_days = 3 # The interval for creating class labels
    train_dataset = tropic_dataset(num_days, transform_tropic, 'train', df)
    test_dataset = tropic_dataset(num_days, transform_tropic, 'test', df)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # --- 4. Model Initialization ---
    # A dictionary to map the day interval to the number of classes
    class_dict = {'1': 59, '3': 20, '5': 12, '7': 9, '9': 7}
    num_class = class_dict[str(num_days)]
    
    # Load a ResNet-18 model with pretrained weights from ImageNet
    resnet = models.resnet18(pretrained=True)
    
    bert_feature_dim = 768 # For 'bert-base-uncased'
    
    # Instantiate the multi-modal model
    multi_modal_model = MultiModalResNet(resnet, num_class, bert_feature_dim)
    model = multi_modal_model.to(device)

    # --- 5. Training Setup and Execution ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    train_model(model, bert_model, tokenizer, train_loader, test_loader, criterion, optimizer, device, num_days, num_epochs=20)