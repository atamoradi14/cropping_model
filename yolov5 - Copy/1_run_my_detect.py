import os
import subprocess

# Directory containing the images
image_dir = '../../1kTesting'

# Define the command to be executed
command = [
    "python",
    "my_detect.py",
    "--weights",
    "runs/train/Second_good/weights/best.pt",
    "--conf",
    "0.5",
    "--augment",
    "--project",
    "../../1kTesting/Labeled_images_Second_model",
    "--save-txt"
]

# Get a list of all files in the image directory
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Run the command for each image file
for image_file in image_files:
    # Add source argument to the command
    command_with_source = command + ["--source", os.path.join(image_dir, image_file)]
    
    # Run the command
    try:
        subprocess.run(command_with_source, check=True)
        print(f"Detection completed successfully for {image_file}.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
