#!/usr/bin/env python3
"""
View and analyze evaluation results from evals.jsonl or evals_high_pass.jsonl files.

Usage:
    python view_eval_results.py <eval_file> [--metrics <metric_names>]
    
Examples:
    # View all metrics
    python view_eval_results.py /path/to/checkpoint_dir/evals_high_pass.jsonl
    
    # View specific metrics
    python view_eval_results.py /path/to/checkpoint_dir/evals_high_pass.jsonl --metrics pass@1024 pass@512
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import sys


def read_eval_file(filepath: str) -> List[Dict[str, Any]]:
    """Read evaluation results from a JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    results.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line: {e}")
    return results


def extract_metrics(results: List[Dict[str, Any]], metric_filter: List[str] = None) -> Dict[int, Dict[str, float]]:
    """Extract metrics organized by checkpoint step."""
    step_metrics = defaultdict(dict)
    
    for entry in results:
        step = entry.get('log_step', entry.get('step', None))
        if step is None:
            continue
        
        metrics = entry.get('metrics', {})
        
        for key, value in metrics.items():
            # Filter metrics if specified
            if metric_filter:
                if not any(f in key for f in metric_filter):
                    continue
            
            # Store the metric
            if isinstance(value, (int, float)):
                step_metrics[step][key] = value
    
    return dict(step_metrics)


def print_metrics_table(step_metrics: Dict[int, Dict[str, float]], metric_filter: List[str] = None):
    """Print metrics in a formatted table."""
    if not step_metrics:
        print("No metrics found.")
        return
    
    # Get all unique metric keys
    all_metrics = set()
    for metrics in step_metrics.values():
        all_metrics.update(metrics.keys())
    
    # Filter metrics if specified
    if metric_filter:
        all_metrics = {m for m in all_metrics if any(f in m for f in metric_filter)}
    
    all_metrics = sorted(all_metrics)
    
    if not all_metrics:
        print("No matching metrics found.")
        return
    
    # Print header
    print(f"\n{'Step':<10}", end='')
    for metric in all_metrics:
        # Shorten metric names for display
        display_name = metric.split('/')[-1] if '/' in metric else metric
        print(f"{display_name:<25}", end='')
    print()
    print('-' * (10 + 25 * len(all_metrics)))
    
    # Print data
    for step in sorted(step_metrics.keys()):
        metrics = step_metrics[step]
        print(f"{step:<10}", end='')
        for metric in all_metrics:
            value = metrics.get(metric, float('nan'))
            if isinstance(value, float):
                print(f"{value:<25.6f}", end='')
            else:
                print(f"{value:<25}", end='')
        print()
    print()


def print_summary_stats(step_metrics: Dict[int, Dict[str, float]], metric_filter: List[str] = None):
    """Print summary statistics for metrics."""
    if not step_metrics:
        return
    
    # Collect values for each metric
    metric_values = defaultdict(list)
    for metrics in step_metrics.values():
        for key, value in metrics.items():
            if metric_filter and not any(f in key for f in metric_filter):
                continue
            if isinstance(value, (int, float)):
                metric_values[key].append(value)
    
    if not metric_values:
        return
    
    # Check if we have timing info
    has_timing = any('validation_time' in k for k in metric_values.keys())
    if has_timing:
        timing_key = next((k for k in metric_values.keys() if 'validation_time_minutes' in k), None)
        if timing_key:
            timing_values = metric_values[timing_key]
            total_time_minutes = sum(timing_values)
            avg_time_minutes = total_time_minutes / len(timing_values)
            print(f"\n=== Timing Summary ===")
            print(f"Total evaluation time: {total_time_minutes:.1f} minutes ({total_time_minutes/60:.1f} hours)")
            print(f"Average per checkpoint: {avg_time_minutes:.1f} minutes")
            print(f"Number of checkpoints: {len(timing_values)}")
    
    print("\n=== Summary Statistics ===")
    print(f"{'Metric':<50} {'Min':<12} {'Max':<12} {'Mean':<12} {'Last':<12}")
    print('-' * 98)
    
    for metric in sorted(metric_values.keys()):
        values = metric_values[metric]
        display_name = metric.split('/')[-1] if '/' in metric else metric
        min_val = min(values)
        max_val = max(values)
        mean_val = sum(values) / len(values)
        last_val = values[-1]
        
        print(f"{display_name:<50} {min_val:<12.6f} {max_val:<12.6f} {mean_val:<12.6f} {last_val:<12.6f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="View evaluation results")
    parser.add_argument("eval_file", type=str, help="Path to evals.jsonl or evals_high_pass.jsonl")
    parser.add_argument("--metrics", nargs='+', help="Filter metrics (e.g., pass@1024 pass@512)")
    parser.add_argument("--summary-only", action="store_true", help="Show only summary statistics")
    parser.add_argument("--full-names", action="store_true", help="Show full metric names")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.eval_file).exists():
        print(f"Error: File not found: {args.eval_file}")
        sys.exit(1)
    
    print(f"Reading evaluation results from: {args.eval_file}")
    results = read_eval_file(args.eval_file)
    print(f"Found {len(results)} evaluation entries")
    
    # Extract metrics
    step_metrics = extract_metrics(results, args.metrics)
    print(f"Found results for {len(step_metrics)} checkpoints")
    
    # Print results
    if not args.summary_only:
        print_metrics_table(step_metrics, args.metrics)
    
    print_summary_stats(step_metrics, args.metrics)


if __name__ == "__main__":
    main()

