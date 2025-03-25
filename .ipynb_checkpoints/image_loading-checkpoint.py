######## Modify "root_directory" first before running the code #######

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def find_b_scans_directory(root_path):
    """
    Recursively searches for the 'B-Scans' directory.
    Since each level contains only one subdirectory, traverse through until reach the 'B-Scans' folder.
    
    :param root_path: The starting directory path (e.g., a specific scan date folder).
    :return: Full path to the 'B-Scans' directory or None if not found.
    """
    while True:
        subdirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        if "B-Scans" in subdirs:
            return os.path.join(root_path, "B-Scans")
        elif len(subdirs) == 1:  # If there's only one subdirectory, continue traversing
            root_path = os.path.join(root_path, subdirs[0])
        else:
            return None  # Unexpected structure, return None

class OCTDataset(Dataset):
    def __init__(self, root_dir, label_path, transform=None):
        """
        Custom PyTorch Dataset for loading OCT images from a hierarchical directory structure.
        
        :param root_dir: Root directory containing patient folders (e.g., 'cirrus_OCT_Imaging_Data').
        :param transform: Optional image transformations for data augmentation.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        if not os.path.isfile(label_path):
            print("Not a valid label path")
            return
            
        labels_df = pd.read_excel(label_path)
        le = LabelEncoder()
        labels_df['label'] = le.fit_transform(labels_df["Diagnosis Label"])
        self.class_mapping = le.classes_
        
        # Iterate over all patient folders
        for patient_id in os.listdir(root_dir):
            if labels_df['Patient Number'].isin([int(patient_id)]).any():
                patient_df = labels_df[labels_df['Patient Number'] == int(patient_id)]
                patient_path = os.path.join(root_dir, patient_id)
                if not os.path.isdir(patient_path):
                    continue  # Skip non-directory files
                # Iterate over left ('L') and right ('R') eye folders
                for eye in ["L", "R"]:
                    if patient_df['Laterality'].isin([eye]).any(): 
                        eye_path = os.path.join(patient_path, eye)
                        eye_df = patient_df[patient_df['Laterality'] == eye]
                        if not os.path.isdir(eye_path):
                            continue
                        # Iterate over different scan dates
                        for scan_date in os.listdir(eye_path):
                            if eye_df['Diagnosis Date'].isin([int(scan_date)]).any():
                                scan_date_path = os.path.join(eye_path, scan_date)
                                scan_date_df = eye_df[eye_df['Diagnosis Date'] == int(scan_date)]
                                if not os.path.isdir(scan_date_path):
                                    continue
                                # Recursively find the 'B-scans' directory
                                b_scans_path = find_b_scans_directory(scan_date_path)
                                if b_scans_path and os.path.isdir(b_scans_path):
                                    for img_name in os.listdir(b_scans_path):
                                        img_path = os.path.join(b_scans_path, img_name)
                                        if img_path.endswith(".jpg") or img_path.endswith(".png"):  # Process only image files
                                            self.image_paths.append(img_path)
                                            self.labels.append(scan_date_df['label'].iloc[0])  # Store Diagnosis Label
    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label.
        
        :param idx: Index of the image to fetch.
        :return: Tuple (image, label) where:
                 - image is a transformed tensor.
                 - label is a tuple (patient_id, eye, scan_date).
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image and convert to grayscale
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)

        return image, label
    def get_label_map(self):
        return self.class_mapping
