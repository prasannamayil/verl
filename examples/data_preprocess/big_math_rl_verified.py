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
Preprocess the SynthLabsAI/Big-Math-RL-Verified dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    """Extract the final answer from a solution string."""
    try:
        return remove_boxed(last_boxed_only_string(solution_str))
    except:
        # If the boxed extraction fails, return the original answer
        return solution_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/big_math_rl_verified')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=None, help='Number of examples for training set. If None, uses all available data.')
    parser.add_argument('--test_size', type=int, default=None, help='Number of examples for test set. If None, no test set is created.')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of data to use for test set if test_size is not specified.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset splitting')

    args = parser.parse_args()

    data_source = 'SynthLabsAI/Big-Math-RL-Verified'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source)

    full_dataset = dataset['train']
    print(f"Loaded dataset with {len(full_dataset)} examples")

    # Split dataset into train and test if needed
    if args.test_size is not None or args.test_ratio > 0:
        # Determine test size
        test_size = args.test_size if args.test_size is not None else int(len(full_dataset) * args.test_ratio)
        train_size = args.train_size if args.train_size is not None else (len(full_dataset) - test_size)
        
        # Make sure we don't exceed the dataset size
        total_size = train_size + test_size
        if total_size > len(full_dataset):
            print(f"Warning: Requested {total_size} examples but only {len(full_dataset)} available.")
            # Adjust sizes proportionally
            ratio = len(full_dataset) / total_size
            train_size = int(train_size * ratio)
            test_size = len(full_dataset) - train_size
        
        # Split the dataset
        split_dataset = full_dataset.train_test_split(
            test_size=test_size,
            train_size=train_size,
            seed=args.seed
        )
        train_dataset = split_dataset['train']
        test_dataset = split_dataset['test']
        print(f"Split dataset into {len(train_dataset)} training and {len(test_dataset)} test examples")
    else:
        # Use all data for training
        train_size = args.train_size if args.train_size is not None else len(full_dataset)
        if train_size < len(full_dataset):
            train_dataset = full_dataset.select(range(train_size))
        else:
            train_dataset = full_dataset
        test_dataset = None
        print(f"Using {len(train_dataset)} examples for training")

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # Process function for each data item
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop('problem')
            question = problem + ' ' + instruction_following

            answer = example.pop('answer')
            
            # Try to extract additional fields if they exist
            solution = example.pop('solution', None)
            problem_type = example.pop('problem_type', None)
            difficulty = example.pop('difficulty', None)
            source = example.pop('source', None)
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'full_solution': solution,
                    'problem_type': problem_type,
                    'difficulty': difficulty,
                    'source': source
                }
            }
            
            # Add type if problem_type exists
            if problem_type is not None:
                data['type'] = problem_type
                
            return data

        return process_fn

    # Process train dataset
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    
    # Process test dataset if it exists
    if test_dataset is not None:
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create directory if it doesn't exist
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)
    
    # Save train dataset
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    print(f"Saved train dataset with {len(train_dataset)} examples to {os.path.join(local_dir, 'train.parquet')}")
    
    # Save test dataset if it exists
    if test_dataset is not None:
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
        print(f"Saved test dataset with {len(test_dataset)} examples to {os.path.join(local_dir, 'test.parquet')}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"Copied data to HDFS: {hdfs_dir}") 