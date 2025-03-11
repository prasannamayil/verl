import os
import numpy
import random
import torch

# compute pass@k


def compute_pass_at_k(advantages):
    """
    Compute the pass@k for a given list of advantages.
    """
    pass_at_k = 0
    pass_at_1 = 0
    total_groups = 0
    for sample_id in advantages['samples']:
        group = advantages['samples'][sample_id]
        total_groups += 1

        # compute pass@k for each group
        for sample in group:
            if sample['score'] == 1.0:
                pass_at_k += 1
                break
        
        # compute pass@1 for each group
        random.shuffle(group)
        sample = group[0]
        if sample['score'] == 1.0:
            pass_at_1 += 1

    return pass_at_1 / total_groups, pass_at_k / total_groups, total_groups


def load_advantages(checkpoints, split='train', stage='pre_training', epoch=None):
    """
    Load advantages for all checkpoints for a specific data split and training stage.
    
    Args:
        split (str): Data split - 'train' or 'val'
        stage (str): Training stage - 'pre_training' or specific epoch
        epoch (int, optional): If loading advantages for a specific epoch
    
    Returns:
        dict: Dictionary mapping checkpoint names to their advantage tensors
    """
    advantages = {}
    
    # Construct filename based on parameters
    if epoch is not None:
        filename = f'advantages_{split}_epoch_{epoch}.pt'
    else:
        filename = f'advantages_{split}_{stage}.pt'
    
    for name, checkpoint_dir in checkpoints.items():
        advantage_path = os.path.join(checkpoint_dir, 'advantage_tracking', filename)
        try:
            # Use weights_only=True to address the FutureWarning
            advantages[name] = torch.load(advantage_path)
            print(f"Loaded {split} advantages for {name}")
        except Exception as e:
            print(f"Error loading advantages for {name}: {e}")
    
    return advantages
