import os
import json
import time
from typing import Dict, Any

class JSONLogger:
    """Logger that saves metrics to JSON files."""
    
    def __init__(self, log_dir: str):
        """
        Initialize the JSON logger.
        
        Args:
            log_dir: Directory to save the JSON log files
        """
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, "metrics.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create metrics file or clear it if it exists
        with open(self.metrics_file, 'w') as f:
            pass
            
        print(f"JSONLogger initialized. Metrics will be saved to {self.metrics_file}")
        
    def log(self, data: Dict[str, Any], step: int):
        """
        Log metrics to a JSON file.
        
        Args:
            data: Dictionary of metrics to log
            step: Current step number
        """
        # Add timestamp and step to the data
        log_entry = {
            "timestamp": time.time(),
            "log_step": step,
            "metrics": data
        }
        
        # Append to the JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n') 