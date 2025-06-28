import os
import glob

def delete_bert_files(directory):
    # Find all files with the .bert.pt extension
    files = glob.glob(os.path.join(directory, '**', '*.bert.pt'), recursive=True)

    # Delete each file
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

# Example usage
# delete_bert_files('/path/to/directory')
