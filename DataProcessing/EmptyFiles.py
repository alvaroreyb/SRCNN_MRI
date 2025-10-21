
import os

def delete_empty_files(target_dir):
    for root, dirs, files in os.walk(target_dir):
        for filename in files:
            filepath = os.path.join(root, filename)

            if os.path.getsize(filepath) == 0:
                os.remove(filepath)

if __name__ == "__main__":
    target_dir = r""
    delete_empty_files(target_dir)

