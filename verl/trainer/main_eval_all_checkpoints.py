#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluate all checkpoints in a given directory with high pass@k.

This script iterates through all checkpoints in a checkpoint directory,
checks which ones have already been evaluated (by reading evals_high_pass.jsonl),
and evaluates the remaining ones.
"""

import os
import json
import glob
import sys
from pathlib import Path
from typing import Set, List, Dict, Any
import re

import hydra
from omegaconf import OmegaConf, open_dict
import ray

from verl.trainer.main_ppo import run_ppo


def get_checkpoint_dirs(parent_dir: str) -> List[str]:
    """
    Get all checkpoint directories in sorted order.
    
    Args:
        parent_dir: Parent directory containing checkpoints
        
    Returns:
        List of checkpoint directory paths sorted by step number
    """
    checkpoint_pattern = os.path.join(parent_dir, "global_step_*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    # Filter to only directories
    checkpoint_dirs = [d for d in checkpoint_dirs if os.path.isdir(d)]
    
    # Sort by step number
    def extract_step(path):
        match = re.search(r'global_step_(\d+)', path)
        return int(match.group(1)) if match else 0
    
    checkpoint_dirs = sorted(checkpoint_dirs, key=extract_step)
    
    return checkpoint_dirs


def get_evaluated_checkpoints(eval_file: str) -> Set[int]:
    """
    Read evals_high_pass.jsonl and return set of already evaluated checkpoint steps.
    
    Args:
        eval_file: Path to evals_high_pass.jsonl
        
    Returns:
        Set of checkpoint step numbers that have been evaluated
    """
    evaluated_steps = set()
    
    if not os.path.exists(eval_file):
        return evaluated_steps
    
    try:
        with open(eval_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        step = entry.get('log_step')
                        if step is not None:
                            evaluated_steps.add(int(step))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Warning: Could not read {eval_file}: {e}")
    
    return evaluated_steps


def evaluate_checkpoint(config: Any, checkpoint_path: str, checkpoint_step: int, eval_filename: str = "evals_high_pass.jsonl") -> bool:
    """
    Evaluate a single checkpoint.
    
    Args:
        config: Base hydra config
        checkpoint_path: Path to checkpoint directory
        checkpoint_step: Checkpoint step number
        eval_filename: Name of the eval file (default: evals_high_pass.jsonl)
        
    Returns:
        True if evaluation succeeded, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Step: {checkpoint_step}")
    print(f"{'='*80}\n")
    
    # Create a copy of the config for this checkpoint
    checkpoint_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    
    with open_dict(checkpoint_config):
        # Set validation-only mode
        checkpoint_config.trainer.val_only = True
        checkpoint_config.trainer.val_before_train = True
        
        # Set checkpoint path
        checkpoint_config.trainer.resume_mode = "resume_path"
        checkpoint_config.trainer.resume_from_path = checkpoint_path
        
        # Set custom eval filename
        checkpoint_config.trainer.eval_filename = eval_filename
        
        # Update experiment name to include checkpoint info
        original_exp_name = checkpoint_config.trainer.experiment_name
        checkpoint_config.trainer.experiment_name = f"{original_exp_name}_highpass"
        
    try:
        # Run validation
        run_ppo(checkpoint_config)
        print(f"\n✓ Successfully evaluated checkpoint at step {checkpoint_step}\n")
        return True
    except Exception as e:
        print(f"\n✗ Failed to evaluate checkpoint at step {checkpoint_step}: {e}\n")
        return False
    finally:
        # Shutdown Ray between checkpoints to avoid issues
        if ray.is_initialized():
            ray.shutdown()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """
    Main entry point for evaluating all checkpoints.
    
    Args:
        config: Hydra configuration
        
    Config Parameters:
        trainer.checkpoint_parent_dir: Directory containing checkpoints (required)
        trainer.eval_filename: Name of evaluation file (default: evals_high_pass.jsonl)
        trainer.force_reeval: Force re-evaluation (default: False)
    """
    # Get evaluation parameters from config
    # Use OmegaConf.select to safely access potentially undefined fields
    checkpoint_parent_dir = OmegaConf.select(config, 'trainer.checkpoint_parent_dir')
    if checkpoint_parent_dir is None:
        checkpoint_parent_dir = OmegaConf.select(config, 'trainer.default_local_dir', default='checkpoints')
    
    eval_filename = OmegaConf.select(config, 'trainer.eval_filename', default='evals_high_pass.jsonl')
    force_reeval = OmegaConf.select(config, 'trainer.force_reeval', default=False)
    
    # Convert to absolute path
    if not os.path.isabs(checkpoint_parent_dir):
        checkpoint_parent_dir = os.path.join(os.getcwd(), checkpoint_parent_dir)
    
    print(f"Checkpoint directory: {checkpoint_parent_dir}")
    print(f"Eval filename: {eval_filename}")
    print(f"Force re-evaluation: {force_reeval}")
    
    if not os.path.exists(checkpoint_parent_dir):
        print(f"Error: Checkpoint directory does not exist: {checkpoint_parent_dir}")
        sys.exit(1)
    
    # Get all checkpoint directories
    checkpoint_dirs = get_checkpoint_dirs(checkpoint_parent_dir)
    
    if not checkpoint_dirs:
        print(f"No checkpoints found in {checkpoint_parent_dir}")
        sys.exit(1)
    
    print(f"Found {len(checkpoint_dirs)} checkpoints")
    
    # Check which checkpoints have been evaluated
    eval_file = os.path.join(checkpoint_parent_dir, eval_filename)
    if not force_reeval:
        evaluated_steps = get_evaluated_checkpoints(eval_file)
        print(f"Already evaluated {len(evaluated_steps)} checkpoints")
    else:
        evaluated_steps = set()
        print("Force re-evaluation enabled")
    
    # Filter checkpoints to evaluate
    checkpoints_to_eval = []
    for ckpt_dir in checkpoint_dirs:
        step = int(re.search(r'global_step_(\d+)', ckpt_dir).group(1))
        if force_reeval or step not in evaluated_steps:
            checkpoints_to_eval.append((ckpt_dir, step))
    
    print(f"Will evaluate {len(checkpoints_to_eval)} checkpoints")
    
    if not checkpoints_to_eval:
        print("All checkpoints have already been evaluated!")
        return
    
    # Evaluate each checkpoint
    successful = 0
    failed = 0
    
    for ckpt_dir, step in checkpoints_to_eval:
        success = evaluate_checkpoint(config, ckpt_dir, step, eval_filename)
        if success:
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Evaluation Summary")
    print(f"{'='*80}")
    print(f"Total checkpoints: {len(checkpoints_to_eval)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

