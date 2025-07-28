import os
import shutil
import random

# Configure paths
source_positive_dir = r'C:\Users\anasj\Desktop\dlia-practice\Concrete Crack Images dataset\Positive'
source_negative_dir = r'C:\Users\anasj\Desktop\dlia-practice\Concrete Crack Images dataset\Negative'
destination_dir = r'C:\Users\anasj\Desktop\dlia-practice\Concrete Crack Images 250'
num_files_to_copy = 250

def copy_random_files(source_folder, dest_folder, num_files):
    """Selects and copies a random set of files from source to destination."""
    os.makedirs(dest_folder, exist_ok=True)
    
    # Get a list of all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_files = [f for f in os.listdir(source_folder) if os.path.splitext(f)[1].lower() in image_extensions]
    
    # Check if you have enough files to copy
    if len(all_files) < num_files:
        print(f"Warning: Not enough files in {source_folder}. Found {len(all_files)}, but need {num_files}.")
        num_files = len(all_files)

    # Randomly select the files
    files_to_copy = random.sample(all_files, num_files)
    
    # Copy the files
    print(f"Copying {num_files} files from {source_folder} to {dest_folder}...")
    for filename in files_to_copy:
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(dest_folder, filename)
        shutil.copy2(source_path, destination_path)
    
    print("Copying complete.")

# Create the destination subfolders
dest_positive = os.path.join(destination_dir, 'Positive')
dest_negative = os.path.join(destination_dir, 'Negative')

# Run the function for both classes
copy_random_files(source_positive_dir, dest_positive, num_files_to_copy)
copy_random_files(source_negative_dir, dest_negative, num_files_to_copy)

print(f"\nSuccessfully created your new dataset at: {destination_dir}")
