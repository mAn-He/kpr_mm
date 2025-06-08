from __future__ import division, absolute_import
import copy
import numpy as np
import random
from collections import defaultdict
import torch
from torch.utils.data.sampler import Sampler

class AugmentedRatioBatchSampler(Sampler):
    """
    Yields batches of indices, ensuring a specific ratio of augmented images per batch.

    Args:
        data_source (list): List of dicts, where each dict contains at least 'is_augmented' boolean flag.
        batch_size (int): Size of each batch.
        augmented_ratio (float): Desired ratio of augmented images in each batch (e.g., 0.1 for 10%).
    """
    def __init__(self, data_source, batch_size, augmented_ratio=0.1):
        if not (0.0 <= augmented_ratio <= 1.0):
            raise ValueError("augmented_ratio must be between 0.0 and 1.0")

        self.data_source = data_source
        self.batch_size = batch_size
        self.augmented_ratio = augmented_ratio

        # Separate indices
        self.original_indices = [i for i, sample in enumerate(data_source) if not sample.get('is_augmented', False)]
        self.augmented_indices = [i for i, sample in enumerate(data_source) if sample.get('is_augmented', False)]
        self.num_original = len(self.original_indices)
        self.num_augmented = len(self.augmented_indices)

        # Calculate number of samples per batch based on ratio
        # Ensure at least one augmented image if ratio > 0 and possible
        if augmented_ratio > 0 and self.num_augmented > 0:
            self.num_augmented_per_batch = max(1, int(round(batch_size * augmented_ratio)))
        else:
            self.num_augmented_per_batch = 0
        self.num_original_per_batch = batch_size - self.num_augmented_per_batch

        # --- Input Validation and Edge Case Handling ---
        if batch_size <= 0:
             raise ValueError("batch_size must be positive")

        # Ensure batch size calculation is valid
        if self.num_original_per_batch < 0:
             raise ValueError(f"Calculated num_original_per_batch ({self.num_original_per_batch}) is negative. Check batch_size and augmented_ratio.")
        if self.num_original_per_batch == 0 and self.num_augmented_per_batch == 0 and batch_size > 0:
             raise ValueError("Batch size calculation resulted in zero samples for both types, but batch_size > 0.")


        # Handle cases where one type is requested but none exist
        if self.num_original_per_batch > 0 and self.num_original == 0:
            raise ValueError("Requested original images per batch, but none found in the dataset.")
        if self.num_augmented_per_batch > 0 and self.num_augmented == 0:
            print("Warning: Requested augmented images per batch, but none found in the dataset. Batches will only contain original images.")
            # Adjust batch composition if no augmented images are available
            self.num_original_per_batch = self.batch_size
            self.num_augmented_per_batch = 0 # Already 0 if num_augmented is 0, but explicit

        # --- Calculate Sampler Length ---
        # Length is determined by the number of full batches we can form
        if self.num_original_per_batch > 0 and self.num_augmented_per_batch > 0:
            # Limited by whichever runs out first
            batches_from_original = self.num_original // self.num_original_per_batch if self.num_original_per_batch > 0 else float('inf')
            batches_from_augmented = self.num_augmented // self.num_augmented_per_batch if self.num_augmented_per_batch > 0 else float('inf')
            self.length = int(min(batches_from_original, batches_from_augmented))
        elif self.num_original_per_batch > 0:
            # Only original images
            self.length = int(self.num_original // self.num_original_per_batch)
        elif self.num_augmented_per_batch > 0:
            # Only augmented images
            self.length = int(self.num_augmented // self.num_augmented_per_batch)
        else:
            # Should not happen based on checks above
            self.length = 0

        if self.length == 0 and (self.num_original > 0 or self.num_augmented > 0):
             print(f"Warning: Sampler length is 0. Not enough data to form even one full batch with the specified counts.")
             print(f"  Original needed/available: {self.num_original_per_batch}/{self.num_original}")
             print(f"  Augmented needed/available: {self.num_augmented_per_batch}/{self.num_augmented}")


    def __iter__(self):
        orig_indices = copy.deepcopy(self.original_indices)
        aug_indices = copy.deepcopy(self.augmented_indices)
        random.shuffle(orig_indices)
        random.shuffle(aug_indices)

        orig_ptr = 0
        aug_ptr = 0

        for _ in range(self.length):
            batch = []

            # Sample original images
            if self.num_original_per_batch > 0 and self.num_original > 0:
                orig_needed = self.num_original_per_batch
                indices_to_take = []
                while orig_needed > 0:
                    available = len(orig_indices) - orig_ptr
                    take = min(orig_needed, available)
                    indices_to_take.extend(orig_indices[orig_ptr : orig_ptr + take])
                    orig_ptr += take
                    orig_needed -= take
                    if orig_ptr >= len(orig_indices): # Wrap around
                        random.shuffle(orig_indices)
                        orig_ptr = 0
                batch.extend(indices_to_take)


            # Sample augmented images
            if self.num_augmented_per_batch > 0 and self.num_augmented > 0:
                aug_needed = self.num_augmented_per_batch
                indices_to_take = []
                while aug_needed > 0:
                     available = len(aug_indices) - aug_ptr
                     take = min(aug_needed, available)
                     indices_to_take.extend(aug_indices[aug_ptr : aug_ptr + take])
                     aug_ptr += take
                     aug_needed -= take
                     if aug_ptr >= len(aug_indices): # Wrap around
                         random.shuffle(aug_indices)
                         aug_ptr = 0
                batch.extend(indices_to_take)


            if len(batch) != self.batch_size:
                 # This should ideally not happen with the wrap-around logic unless dataset is very small
                 print(f"Warning: Final batch size {len(batch)} does not match target {self.batch_size}. Check data counts and batch size.")
                 # Continue with the batch as is, or could skip/pad if needed

            random.shuffle(batch) # Shuffle the final batch indices
            yield batch

    def __len__(self):
        # Returns the number of batches that can be yielded
        return self.length
