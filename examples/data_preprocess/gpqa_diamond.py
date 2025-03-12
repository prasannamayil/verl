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
Preprocess the GPQA Diamond dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/gpqa_diamond')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'hendrydong/gpqa_diamond'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source)

    test_dataset = dataset['test']
    print(f"Loaded dataset with {len(test_dataset)} examples")

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # Process function for each data item
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop('problem')
            question = problem + ' ' + instruction_following

            solution = example.pop('solution')
            # Extract the answer from the solution
            answer = extract_solution(solution)
            domain = example.pop('domain', None)
            
            data = {
                "type": domain,
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
                    'domain': domain,
                    'full_solution': solution
                }
            }
            return data

        return process_fn

    # Process the dataset
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    print(f"Processed {len(test_dataset)} examples")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create directory if it doesn't exist
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)
    
    # Save test dataset
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    print(f"Saved test dataset with {len(test_dataset)} examples to {os.path.join(local_dir, 'test.parquet')}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"Copied data to HDFS: {hdfs_dir}") 