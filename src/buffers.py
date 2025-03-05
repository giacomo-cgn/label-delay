import torch
from torch.utils.data import Dataset
from collections import defaultdict

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label, task_id = self.data[idx], self.labels[idx], self.task_ids[idx]
        return img, label, task_id

    def update(self, new_dataset):
        """
        Update the buffer with new samples, maintaining class balance.

        Args:
            new_dataset (torch.utils.data.Dataset): A dataset containing new samples.
                Each item in the dataset is a tuple (img, label, task_id).
        """

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


class ReservoirBuffer(Dataset):
    def __init__(self, buffer_size):
        """
        Initialize the reservoir buffer.

        Args:
            buffer_size (int): Maximum number of samples the buffer can hold.
        """
        self.buffer_size = buffer_size
        self.data = []  # Full dataset for torch.Dataset compatibility
        self.labels = []  # Corresponding labels for the dataset
        self.task_ids = []  # Task IDs corresponding to each sample
        self.total_seen = 0  # Total number of samples seen so far

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label, task_id = self.data[idx], self.labels[idx], self.task_ids[idx]
        return img, label, task_id

    def update(self, new_dataset):
        """
        Update the buffer with new samples using reservoir sampling.

        Args:
            new_dataset (torch.utils.data.Dataset): A dataset containing new samples.
                Each item in the dataset is a tuple (img, label, task_id).
        """
        for img, label, task_id in new_dataset:
            self.total_seen += 1

            if len(self.data) < self.buffer_size:
                # Add to buffer if there's space
                self.data.append(img)
                self.labels.append(label)
                self.task_ids.append(task_id)
            else:
                # Perform reservoir sampling
                replace_idx = torch.randint(0, self.total_seen, (1,)).item()
                if replace_idx < self.buffer_size:
                    self.data[replace_idx] = img
                    self.labels[replace_idx] = label
                    self.task_ids[replace_idx] = task_id