import os
from PIL import Image
import numpy as np

def convert_images_to_npy(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over all the files in the input directory
    for img_name in os.listdir(input_dir):
        if img_name.endswith('.jpg'):
            # Construct the full path to the image
            img_path = os.path.join(input_dir, img_name)
            
            # Open the image file
            img = Image.open(img_path).convert('RGB')
            
            # Convert the image to a numpy array
            img_array = np.array(img)
            
            # Construct the output file path
            output_file_path = os.path.join(output_dir, img_name.replace('.jpg', '.npy'))
            
            # Save the numpy array as a .npy file
            np.save(output_file_path, img_array)
            
            print(f"Converted {img_name} to {output_file_path}")

if __name__ == "__main__":
    # Paths to the COCO dataset directories
    train_dir = r'C:\Users\new_d\Downloads\train2017\train2017'
    val_dir = r'C:\Users\new_d\Downloads\val2017\val2017'
    test_dir = r'C:\Users\new_d\Downloads\test2017\test2017'

    # Output directories for binary files
    train_output_dir = r'D:\Felipe\train2017_converted'
    val_output_dir = r'D:\Felipe\val2017_converted'
    test_output_dir = r'D:\Felipe\test2017_converted'

    # Convert all images
    #convert_images_to_npy(train_dir, train_output_dir)
    convert_images_to_npy(val_dir, val_output_dir)
    convert_images_to_npy(test_dir, test_output_dir)