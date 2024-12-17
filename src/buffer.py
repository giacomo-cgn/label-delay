import torch
from torch.utils.data import Dataset
from collections import defaultdict

# TODO: check if transforms are changed correctly
class ClassBalancedBuffer(Dataset):
    def __init__(self, buffer_size):
        """
        Initialize the class-balanced buffer.
        
        Args:
            buffer_size (int): Maximum number of samples the buffer can hold.
        """
        self.buffer_size = buffer_size
        self.buffer = defaultdict(list)  # Stores samples grouped by class (label)
        self.data = []  # Full dataset for torch.Dataset compatibility
        self.labels = []  # Corresponding labels for the dataset
        self.task_ids = []  # Task IDs corresponding to each sample
        self.transforms = None  # Transformations from the original dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label, task_id = self.data[idx], self.labels[idx], self.task_ids[idx]
        if self.transforms:
            img = self.transforms(img)
        return img, label, task_id

    def update(self, new_dataset):
        """
        Update the buffer with new samples, maintaining class balance.

        Args:
            new_dataset (torch.utils.data.Dataset): A dataset containing new samples.
                Each item in the dataset is a tuple (img, label, task_id).
        """
        # Store transforms from the first dataset
        if self.transforms is None and hasattr(new_dataset, "transforms"):
            self.transforms = new_dataset.transforms

        # Extract samples, labels, and task_ids from the new dataset
        for img, label, task_id in new_dataset:
            self.buffer[label].append((img, label, task_id))

        # Calculate the number of classes and per-class limit
        classes = list(self.buffer.keys())
        num_classes = len(classes)
        per_class_limit = self.buffer_size // num_classes

        # Truncate samples per class to maintain class balance
        for cls in classes:
            self.buffer[cls] = self.buffer[cls][-per_class_limit:]

        # Flatten the buffer to update data, labels, and task IDs
        self._refresh_data()

    def _refresh_data(self):
        """
        Refresh the buffer's flat data, labels, and task IDs to maintain compatibility
        with torch.Dataset.
        """
        self.data = []
        self.labels = []
        self.task_ids = []
        for cls_samples in self.buffer.values():
            for sample, label, task_id in cls_samples:
                self.data.append(sample)
                self.labels.append(label)
                self.task_ids.append(task_id)

        # No tensor conversion here because transforms might need raw data
