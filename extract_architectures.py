#!/usr/bin/env python3
"""
Script to extract Python code from 106.json into individual files in pytorch_arch/
"""
import json
import os
from pathlib import Path

def extract_architectures():
    """Extract Python code from 106.json into individual files"""
    input_file = "106.json"
    output_dir = Path("pytorch_arch")
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Load the JSON file
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} architectures")
    
    # Extract each architecture
    extracted_count = 0
    for item in data:
        name = item.get("name")
        program = item.get("program")
        
        if not name or not program:
            print(f"Skipping item missing name or program: {item.keys()}")
            continue
        
        # Create filename
        filename = f"{name}.py"
        filepath = output_dir / filename
        
        # Write the Python code
        print(f"Extracting {filename}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(program)
        
        extracted_count += 1
    
    print(f"\nExtracted {extracted_count} architectures to {output_dir}/")
    return extracted_count

if __name__ == "__main__":
    extract_architectures()