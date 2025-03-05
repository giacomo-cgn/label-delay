import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Subset, Dataset, Sampler

from avalanche.benchmarks.classic import SplitCIFAR100, SplitCIFAR10, SplitImageNet
from avalanche.benchmarks.scenarios import NCExperience

class SemiSupBenchmark:
    def __init__(self, supervised_tr_stream, unsupervised_tr_stream, test_stream, valid_stream, image_size, num_classes, num_exps):
        self.supervised_tr_stream = supervised_tr_stream
        self.unsupervised_tr_stream = unsupervised_tr_stream
        self.test_stream = test_stream
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_exps = num_exps

        if len(valid_stream) > 0:
            self.valid_stream = valid_stream
        else:
            self.valid_stream = None

class UnsupervisedDataset(Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor, label, task_label = self.data[idx]

        weak_x, strong_x = self.transforms(input_tensor)
        return [weak_x, strong_x], label
    
class SupervisedDataset(Dataset):
    def __init__(self, data, transforms, num_views):
        self.data = data
        self.transforms = transforms
        self.num_views = num_views

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor, label, task_label = self.data[idx]
        
        views = []
        for _ in range(self.num_views):
            views.append(self.transforms(input_tensor))

        return views, label



def get_benchmark_label_delay(dataset_name : str,
                        dataset_root: str,
                        num_exps : int = 20,
                        valid_ratio : float = 0.0,
                        supervised_ratio: float = 0.05,
                        seed : int = 42,
                        delay : int = 1
                            ) -> SemiSupBenchmark:
    """
    The function `get_benchmark_label_delay` creates a labeled and unlabeled split of a given dataset using Avalanche.

    Parameters:
    - dataset_name: Name of the dataset to use
    - dataset_root: Root directory where the dataset is stored
    - num_exps: Number of experiences to split the dataset into
    - supervised_ratio: Ratio of supervised data to unsupervised data in each experience
    - seed: Seed for reproducibility
    - unsupervised_next_task_ratio: Ratio of unsupervised data to include in the next experience

    Returns:
    - SemiSupBenchmark: Object containing test, validation, supervised and unsupervised training streams
    """
        
        
    if dataset_name == 'cifar100':
        avalanche_benchmark = SplitCIFAR100(
                n_experiences=num_exps,
                seed=seed, # Fixed seed for reproducibility
                return_task_id=False,
                shuffle=True,
                train_transform=None,
                eval_transform=None,
            )
        image_size = 32
        num_classes = 100
    elif dataset_name == 'cifar10':
        avalanche_benchmark = SplitCIFAR10(
                n_experiences=num_exps,
                seed=seed, # Fixed seed for reproducibility
                return_task_id=False,
                shuffle=True,
                train_transform=None,
                eval_transform=None,
            )
        image_size = 32
        num_classes = 10

    # # Extract classes in each experience
    # exp_classes_list = []
    # for experience in avalanche_benchmark.train_stream:
    #     exp_classes_list.append(experience.classes_in_this_experience)

    # Split validation and training datasets
    tr_stream = []
    valid_stream = []    
    for experience in avalanche_benchmark.train_stream:
        if valid_ratio > 0:
            tr_exp_dataset, val_exp_dataset = class_balanced_split(split_size=valid_ratio, dataset=experience)
            tr_stream.append(tr_exp_dataset)
            valid_stream.append(val_exp_dataset)
        else:
            tr_stream.append(experience.dataset)

    test_stream = []
    for experience in avalanche_benchmark.test_stream:
        test_stream.append(experience.dataset)


    supervised_tr_stream = []
    unsupervised_tr_stream = []
    # Split supervised and unsupervised portions of the stream
    for experience in avalanche_benchmark.train_stream:
        superv_dataset, unsup_dataset = class_balanced_split(split_size=supervised_ratio, dataset=experience)
        supervised_tr_stream.append(superv_dataset)
        unsupervised_tr_stream.append(unsup_dataset)
        # Each task has unsupervised part only of the subsequent task

    for _ in range(delay):
        unsupervised_tr_stream.append(None)
        supervised_tr_stream.insert(0, None)
        num_exps += 1


    # FOR MIXED DELAY EXPERIENCES
    # if unsup_anticipate_ratio == 1:
    #     # Anticipate the unsupervised part of the data 1 exp backward
    #     unsupervised_tr_stream = unsupervised_tr_stream[1:] + [unsupervised_tr_stream[0]]

    # elif unsup_anticipate_ratio > 0:
    #     # Split unsupervised data into 2 parts: for the current task and for the previous task
    #     for_prev_task = []
    #     for_curr_task = []
    #     for experience in unsupervised_tr_stream:
    #         for_prev_task_dataset, for_curr_task_dataset = class_balanced_split(split_size=unsup_anticipate_ratio, dataset=experience)
    #         for_prev_task.append(for_prev_task_dataset)
    #         for_curr_task.append(for_curr_task_dataset)

    #     # Combine unsupervised data for the current task and the next task
    #     unsupervised_tr_stream = [torch.utils.data.ConcatDataset([for_curr_task[i], for_prev_task[i+1]]) for i in range(0, len(for_curr_task)-1)]

    # if drop_last_exp:
    #     supervised_tr_stream = supervised_tr_stream[:-1]
    #     unsupervised_tr_stream = unsupervised_tr_stream[:-1]
    #     test_stream = test_stream[:-1]
    #     exp_classes_list = exp_classes_list[:-1]
    #     if valid_ratio > 0:
    #         valid_stream = valid_stream[:-1]
    #     num_exps -= 1
    


    return SemiSupBenchmark(supervised_tr_stream, unsupervised_tr_stream, test_stream, valid_stream, image_size, num_classes, num_exps)   



def class_balanced_split(split_size, dataset):
    # Adapted from Avalanche.benchmarks
    """Class-balanced train/validation splits.

    This splitting strategy splits `experience` into two experiences
    (`split` and `remaining`) of size `split_size` using a class-balanced
    split. Sample of each class are chosen randomly.

    """
    if not 0.0 <= split_size <= 1.0:
        raise ValueError("split_size must be a float in [0, 1].")
    
    if isinstance(dataset, NCExperience):
        # Split for Avlanche dataset
        avalanche_dataset_wrapper = dataset
        dataset = avalanche_dataset_wrapper.dataset

        exp_indices = list(range(len(dataset)))
        exp_classes = avalanche_dataset_wrapper.classes_in_this_experience

        # shuffle exp_indices
        exp_indices = torch.as_tensor(exp_indices)[torch.randperm(len(exp_indices))]
        # shuffle the targets as well
        exp_targets = torch.as_tensor(avalanche_dataset_wrapper.dataset.targets)[exp_indices]

        remaining_exp_indices = []
        split_exp_indices = []
        for cid in exp_classes:  # split indices for each class separately.
            c_indices = exp_indices[exp_targets == cid]
            split_n_instances = int(split_size * len(c_indices))
            split_exp_indices.extend(c_indices[:split_n_instances])
            remaining_exp_indices.extend(c_indices[split_n_instances:])

    else:
        num_tot = len(dataset)
        num_split = int(split_size * num_tot)

        # Get the labels for each data point in the training set
        try:
            labels = dataset.labels
        except AttributeError:
            labels = [dataset[i][1] for i in range(num_tot)]

        # Perform stratified split to ensure the validation set is class balanced
        remaining_exp_indices, split_exp_indices = train_test_split(
            np.arange(num_tot),
            test_size=num_split,
            stratify=labels,  # this ensures class balance
            random_state=42
        )

    if isinstance(dataset, torch.utils.data.Dataset):
        # Use Subset for older versions of Avalanche where AvalancheDataset is a subclass of torch Dataset
        remaining_dataset = Subset(dataset, remaining_exp_indices)
        split_dataset = Subset(dataset, split_exp_indices)
    else:
        # Use .subset for newer versions of Avalanche where AvalancheDataset is not a subclass of torch Dataset
        remaining_dataset = dataset.subset(remaining_exp_indices)
        split_dataset = dataset.subset(split_exp_indices)


    return split_dataset, remaining_dataset


class ClassStratifiedSampler(Sampler):
    def __init__(self, data_source, batch_size=1, classes_per_batch=10, epochs=1):
        """
        Simple Class-Stratified Sampler

        Samples batches with a fixed number of images per class, ensuring that each batch
        contains images from a fixed number of randomly selected classes.

        :param data_source: A PyTorch dataset with a `.targets` attribute containing class labels.
        :param batch_size: Number of images to sample from each class per batch.
        :param classes_per_batch: Number of classes to include in each batch.
        :param epochs: Number of iterations over the dataset.
        """
        super(ClassStratifiedSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.classes_per_batch = classes_per_batch
        self.epochs = epochs

        # Extract unique classes and indices for each class
        self.targets = np.array(data_source.targets)
        self.classes = np.unique(self.targets)
        self.class_indices = {
            cls: np.where(self.targets == cls)[0].tolist() for cls in self.classes
        }

    def __iter__(self):
        for epoch in range(self.epochs):
            # Shuffle the classes at the start of each epoch
            shuffled_classes = np.random.permutation(self.classes)

            # Iterate through classes in groups of `classes_per_batch`
            for i in range(0, len(shuffled_classes), self.classes_per_batch):
                selected_classes = shuffled_classes[i:i + self.classes_per_batch]
                
                # Collect samples from each selected class
                class_samples = []
                for cls in selected_classes:
                    cls_indices = self.class_indices[cls]
                    cls_indices = np.random.choice(cls_indices, self.batch_size, replace=True)
                    class_samples.extend(cls_indices)

                # Yield a batch of indices
                yield class_samples

    def __len__(self):
        """
        Compute the number of batches per epoch.

        :return: Total number of batches across all epochs.
        """
        batches_per_epoch = len(self.classes) // self.classes_per_batch
        return batches_per_epoch * self.epochs


class SimpleClassBalancedSampler(Sampler):
    """
    A batch sampler that returns class-balanced mini-batches.
    """
    def __init__(self, labels, batch_size):
        """
        Args:
            labels (list or array-like): A list or array of labels corresponding to the dataset.
            batch_size (int): The size of each mini-batch.
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size

        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[label].append(idx)

        # Shuffle indices within each class
        for label in self.class_indices:
            np.random.shuffle(self.class_indices[label])

        self.iter_per_class = {label: iter(indices) for label, indices in self.class_indices.items()}

        # Compute approximate samples per class
        self.num_classes = len(self.class_indices)
        self.samples_per_class = max(1, self.batch_size // self.num_classes)

    def __iter__(self):
        batch = []
        while True:
            for label in self.class_indices:
                # Replenish iterator if exhausted
                try:
                    batch.extend([next(self.iter_per_class[label]) for _ in range(self.samples_per_class)])
                except StopIteration:
                    # Reshuffle and restart the iterator
                    np.random.shuffle(self.class_indices[label])
                    self.iter_per_class[label] = iter(self.class_indices[label])
                    batch.extend([next(self.iter_per_class[label]) for _ in range(self.samples_per_class)])

            # If the batch exceeds batch_size, truncate it
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        min_class_samples = min(len(indices) for indices in self.class_indices.values())
        total_samples = sum(len(indices) for indices in self.class_indices.values())
        return total_samples // self.batch_size









