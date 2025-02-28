import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple

def load_dataset_advantage_data(advantage_path: str, epoch: Optional[int] = None, dataset_type: str = 'train'):
    """Load advantage data for entire datasets.
    
    Args:
        advantage_path: Path to the directory containing advantage data
        epoch: Specific epoch to load, or None to load all epochs
        dataset_type: 'train' or 'val'
        
    Returns:
        List of advantage data dictionaries, one per epoch
    """
    files = os.listdir(advantage_path)
    
    # Handle pre-training data (epoch -1)
    pre_training_file = f'advantages_{dataset_type}_pre_training.pt'
    epoch_files = [f for f in files if f.startswith(f'advantages_{dataset_type}_epoch_')]
    
    if pre_training_file in files:
        advantage_files = [pre_training_file] + epoch_files
    else:
        advantage_files = epoch_files
    
    if epoch is not None:
        if epoch == -1:
            advantage_files = [f for f in advantage_files if 'pre_training' in f]
        else:
            advantage_files = [f for f in advantage_files if f'_epoch_{epoch}.pt' in f]
    
    # Sort by epoch, with pre-training first
    def get_epoch_num(filename):
        if 'pre_training' in filename:
            return -1
        return int(filename.split('_epoch_')[1].split('.')[0])
    
    advantage_files.sort(key=get_epoch_num)
    
    data_list = []
    for file in advantage_files:
        filepath = os.path.join(advantage_path, file)
        data = torch.load(filepath)
        data_list.append(data)
    
    return data_list

def create_dataset_advantage_summary(advantage_data: List[Dict]) -> pd.DataFrame:
    """Create a summary of advantage data across epochs for entire datasets.
    
    Args:
        advantage_data: List of advantage data dictionaries, one per epoch
        
    Returns:
        DataFrame with advantage statistics per epoch
    """
    summary_data = []
    
    for data in advantage_data:
        epoch = data['epoch']
        dataset_type = data['dataset_type']
        samples = data['samples']
        
        all_advantages = []
        all_rewards = []
        
        # Collect all advantages and rewards
        for sample_id, sample_entries in samples.items():
            for entry in sample_entries:
                adv = entry['advantage']
                mask = entry['response_mask'].astype(bool)
                
                # Only consider advantages for valid tokens
                valid_adv = adv[mask]
                all_advantages.extend(valid_adv)
                
                if entry['token_reward'] is not None:
                    reward = entry['token_reward']
                    valid_reward = reward[mask]
                    all_rewards.extend(valid_reward)
        
        all_advantages = np.array(all_advantages)
        
        epoch_label = "pre_training" if epoch == -1 else str(epoch)
        
        summary_entry = {
            'epoch': epoch_label,
            'dataset_type': dataset_type,
            'num_samples': len(samples),
            'num_tokens': len(all_advantages),
            'advantage_mean': np.mean(all_advantages),
            'advantage_std': np.std(all_advantages),
            'advantage_min': np.min(all_advantages),
            'advantage_max': np.max(all_advantages),
        }
        
        if all_rewards:
            all_rewards = np.array(all_rewards)
            summary_entry.update({
                'reward_mean': np.mean(all_rewards),
                'reward_std': np.std(all_rewards),
                'reward_min': np.min(all_rewards),
                'reward_max': np.max(all_rewards),
            })
        
        summary_data.append(summary_entry)
    
    return pd.DataFrame(summary_data)

def plot_dataset_advantage_trends(summary_df: pd.DataFrame, dataset_type: str = 'train', save_path: Optional[str] = None):
    """Plot trends in advantage statistics over epochs for entire datasets.
    
    Args:
        summary_df: DataFrame with advantage statistics
        dataset_type: 'train' or 'val' to filter the data
        save_path: Path to save the plot, or None to display
    """
    # Filter for the specified dataset type
    df = summary_df[summary_df['dataset_type'] == dataset_type].copy()
    
    # Convert epoch to numeric for proper ordering
    df['epoch_num'] = df['epoch'].apply(lambda x: -1 if x == 'pre_training' else int(x))
    df = df.sort_values('epoch_num')
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(df['epoch_num'], df['advantage_mean'])
    plt.title(f'{dataset_type.capitalize()} Mean Advantage')
    plt.xlabel('Epoch')
    plt.xticks(df['epoch_num'], df['epoch'])
    
    plt.subplot(2, 3, 2)
    plt.plot(df['epoch_num'], df['advantage_std'])
    plt.title(f'{dataset_type.capitalize()} Advantage Standard Deviation')
    plt.xlabel('Epoch')
    plt.xticks(df['epoch_num'], df['epoch'])
    
    plt.subplot(2, 3, 3)
    plt.plot(df['epoch_num'], df['advantage_min'], label='Min')
    plt.plot(df['epoch_num'], df['advantage_max'], label='Max')
    plt.title(f'{dataset_type.capitalize()} Min/Max Advantage')
    plt.xlabel('Epoch')
    plt.xticks(df['epoch_num'], df['epoch'])
    plt.legend()
    
    plt.subplot(2, 3, 4)
    plt.plot(df['epoch_num'], df['num_samples'])
    plt.title(f'{dataset_type.capitalize()} Number of Samples')
    plt.xlabel('Epoch')
    plt.xticks(df['epoch_num'], df['epoch'])
    
    plt.subplot(2, 3, 5)
    plt.plot(df['epoch_num'], df['num_tokens'])
    plt.title(f'{dataset_type.capitalize()} Number of Valid Tokens')
    plt.xlabel('Epoch')
    plt.xticks(df['epoch_num'], df['epoch'])
    
    if 'reward_mean' in df.columns:
        plt.subplot(2, 3, 6)
        plt.plot(df['epoch_num'], df['reward_mean'])
        plt.title(f'{dataset_type.capitalize()} Mean Reward')
        plt.xlabel('Epoch')
        plt.xticks(df['epoch_num'], df['epoch'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def compare_dataset_advantages(train_summary: pd.DataFrame, val_summary: pd.DataFrame, save_path: Optional[str] = None):
    """Compare advantage trends between training and validation datasets.
    
    Args:
        train_summary: DataFrame with train advantage statistics
        val_summary: DataFrame with validation advantage statistics
        save_path: Path to save the plot, or None to display
    """
    # Ensure both dataframes have the same epochs
    train_df = train_summary.copy()
    val_df = val_summary.copy()
    
    # Convert epoch to numeric for proper ordering
    train_df['epoch_num'] = train_df['epoch'].apply(lambda x: -1 if x == 'pre_training' else int(x))
    val_df['epoch_num'] = val_df['epoch'].apply(lambda x: -1 if x == 'pre_training' else int(x))
    
    train_df = train_df.sort_values('epoch_num')
    val_df = val_df.sort_values('epoch_num')
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_df['epoch_num'], train_df['advantage_mean'], label='Train')
    plt.plot(val_df['epoch_num'], val_df['advantage_mean'], label='Validation')
    plt.title('Mean Advantage')
    plt.xlabel('Epoch')
    plt.xticks(train_df['epoch_num'], train_df['epoch'])
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(train_df['epoch_num'], train_df['advantage_std'], label='Train')
    plt.plot(val_df['epoch_num'], val_df['advantage_std'], label='Validation')
    plt.title('Advantage Standard Deviation')
    plt.xlabel('Epoch')
    plt.xticks(train_df['epoch_num'], train_df['epoch'])
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(train_df['epoch_num'], train_df['advantage_min'], label='Train Min')
    plt.plot(train_df['epoch_num'], train_df['advantage_max'], label='Train Max')
    plt.plot(val_df['epoch_num'], val_df['advantage_min'], label='Val Min', linestyle='--')
    plt.plot(val_df['epoch_num'], val_df['advantage_max'], label='Val Max', linestyle='--')
    plt.title('Min/Max Advantage')
    plt.xlabel('Epoch')
    plt.xticks(train_df['epoch_num'], train_df['epoch'])
    plt.legend()
    
    if 'reward_mean' in train_df.columns and 'reward_mean' in val_df.columns:
        plt.subplot(2, 2, 4)
        plt.plot(train_df['epoch_num'], train_df['reward_mean'], label='Train')
        plt.plot(val_df['epoch_num'], val_df['reward_mean'], label='Validation')
        plt.title('Mean Reward')
        plt.xlabel('Epoch')
        plt.xticks(train_df['epoch_num'], train_df['epoch'])
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()



if __name__ == "__main__":
    # Load advantage data for all epochs
    train_advantage_data = load_dataset_advantage_data("/fast/pmayilvahanan/post_training/verl_checkpoints/qwen2_7b_function_rm/advantages_train", dataset_type='train')
    val_advantage_data = load_dataset_advantage_data("/fast/pmayilvahanan/post_training/verl_checkpoints/qwen2_7b_function_rm/advantages_val", dataset_type='val')

    # Create summaries
    train_summary = create_dataset_advantage_summary(train_advantage_data)
    val_summary = create_dataset_advantage_summary(val_advantage_data)

    # Plot trends for training data
    plot_dataset_advantage_trends(train_summary, dataset_type='train', save_path="train_advantage_trends.png")

    # Plot trends for validation data
    plot_dataset_advantage_trends(val_summary, dataset_type='val', save_path="val_advantage_trends.png")

    # Compare train and validation trends
    compare_dataset_advantages(train_summary, val_summary, save_path="train_vs_val_advantages.png") 