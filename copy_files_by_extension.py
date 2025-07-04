import os
import shutil
import sys

def copy_files_by_extension(source_dir, dest_dir, extension):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(extension):
                source_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)
                shutil.copy2(source_file, dest_file)
                print(f"Copied {source_file} to {dest_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python copy_files_by_extension.py <source_dir> <dest_dir> <extension>")
        sys.exit(1)

    source_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    extension = sys.argv[3]

    copy_files_by_extension(source_dir, dest_dir, extension)