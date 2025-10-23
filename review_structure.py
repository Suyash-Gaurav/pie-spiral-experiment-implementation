#!/usr/bin/env python3
"""Script to review the complete project structure and list all files."""

import os
from pathlib import Path

def list_directory_tree(directory, prefix="", max_depth=5, current_depth=0):
    """Recursively list directory structure."""
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(Path(directory).iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and not item.name.startswith('.'):
                extension = "    " if is_last else "│   "
                list_directory_tree(item, prefix + extension, max_depth, current_depth + 1)
    except PermissionError:
        pass

def list_all_files(directory):
    """List all files with their paths."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for filename in filenames:
            if not filename.startswith('.'):
                filepath = os.path.join(root, filename)
                files.append(filepath)
    return sorted(files)

if __name__ == "__main__":
    print("=" * 80)
    print("PROJECT STRUCTURE TREE")
    print("=" * 80)
    list_directory_tree("./final")
    
    print("\n" + "=" * 80)
    print("ALL FILES LIST")
    print("=" * 80)
    files = list_all_files("./final")
    for f in files:
        size = os.path.getsize(f)
        print(f"{f:<60} ({size:>8} bytes)")
    
    print(f"\nTotal files: {len(files)}")
