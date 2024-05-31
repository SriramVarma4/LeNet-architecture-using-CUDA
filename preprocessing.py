import numpy as np
import cv2
import os

# Path to the original folder containing images
folder_path = 'img'

#file:  Path to the new folder to save processed imagee(.txt file)
new_folder_path = 'pre-proc-img'

#file: where you want to save your name of your .txt files
output_file = 'precessed_test_neg.txt'
np.set_printoptions(linewidth=np.inf,formatter={'float': '{: 0.6f}'.format})
os.makedirs(new_folder_path, exist_ok=True)

# List all files in the original folder
files = os.listdir(folder_path)
count=100
for file in files:
    # Read the image
    img_path = os.path.join(folder_path, file)
    img = cv2.imread(img_path, 0)

    # Check if the image shape is not [28, 28] and resize if necessary
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))

    # Normalize the image to 0-1 range
    img = img / 255.0

    # Save the image pixel values to a text file
    with open(os.path.join(new_folder_path, file.replace('.png', '.txt')), 'w') as f:
        for row in img:
            for pixel_value in row:
                f.write(str(pixel_value) + '\n')

    # Print the image pixel values
    #count=count-1
    if(count==0):
        break
    #print(img)


with open(output_file, 'w') as file:
    for filename in os.listdir(new_folder_path):
        file.write(filename + '\n')
