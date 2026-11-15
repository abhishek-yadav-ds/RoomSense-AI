"""
RoomSense AI - One-Click Project Packager
=========================================
Creates a clean, submission-ready ZIP file containing all datasets,
executed notebooks, trained models, visualizations, reports, and code,
excluding cache and temporary build files.
"""

import os
import zipfile

def export_zip():
    output_filename = "RoomSense_AI_Submission.zip"
    print(f"Creating project submission package: {output_filename}...")
    
    ignore_dirs = {'__pycache__', '.ipynb_checkpoints', '.git', '.vscode', '.idea', 'venv', 'env'}
    ignore_exts = {'.pyc', '.pyo', '.pyd', '.log'}
    
    file_count = 0
    total_bytes = 0
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
            
            for file in files:
                if file == output_filename:
                    continue
                _, ext = os.path.splitext(file)
                if ext in ignore_exts:
                    continue
                if file.startswith('.') and file != '.gitignore':
                    continue
                    
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, '.')
                
                zipf.write(full_path, rel_path)
                file_count += 1
                total_bytes += os.path.getsize(full_path)
                
    mb_size = round(total_bytes / (1024 * 1024), 2)
    zip_size = round(os.path.getsize(output_filename) / (1024 * 1024), 2)
    
    print("=" * 60)
    print(f"SUCCESS: Package created successfully!")
    print(f"ZIP File:       {output_filename}")
    print(f"Total Files:    {file_count} files archived")
    print(f"Uncompressed:   {mb_size} MB")
    print(f"Compressed:     {zip_size} MB")
    print("=" * 60)
    print("Ready to copy to USB drive or email to your professor!")

if __name__ == '__main__':
    export_zip()

