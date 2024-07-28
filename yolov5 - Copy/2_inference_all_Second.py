import os
import cv2
import pandas as pd
import numpy as np

input_dir = "../../1kTesting/Original"  # Folder containing the images
txt_dir = "../../1kTesting/Labeled_images_Second_model/labels"  # Folder containing the corresponding txt files

output_dir = '../../1kTesting/Cropped_images_Second_model_new'
# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to Calculate Distance
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to Crop Image
def crop_img(img, threshold=30):
    try:
        img_blur = cv2.blur(img, (3, 3))
        img_blur = cv2.blur(img_blur, (3, 3))
        img_blur = img_blur.astype('uint8')
        ret, threshed_img = cv2.threshold(cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)
        vertical, horizontal = np.nonzero(threshed_img)
        top, bot = np.min(vertical), np.max(vertical)
        left, right = np.min(horizontal), np.max(horizontal)
        cropped_image = img[top:bot, left:right]
        return cropped_image
    except Exception as err_msg:
        print(err_msg)
        return img

# Iterate over each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg'):
        # Read Image
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Parse corresponding txt file
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(txt_dir, txt_filename)

        # Skip processing if the corresponding txt file doesn't exist
        if not os.path.exists(txt_path):
            print(f"No associated text file found for {filename}. Skipping image processing.")
            continue

        # Initialize an empty list to store bounding box data for this image
        bounding_boxes_data = []

        # Read txt file line by line and parse bounding box data
        with open(txt_path, 'r') as file:
            for line in file:
                # Split the line and extract bounding box information
                parts = line.strip().split(' ')
                class_id, x_center, y_center, width, height, confidence = map(float, parts)

                # Calculate bounding box coordinates
                class_id = int(class_id)
                x_center = int(x_center)
                y_center = int(y_center)
                w = int(width)
                h = int(height)

                # Append bounding box data to the list
                bounding_boxes_data.append([class_id, x_center, y_center, w, h, confidence])

        # Create DataFrame from the collected bounding box data for this image
        columns = ['class', 'x_center', 'y_center', 'w', 'h', 'confidence']
        bounding_boxes_df = pd.DataFrame(bounding_boxes_data, columns=columns)
        print(bounding_boxes_df)

        disc_box = bounding_boxes_df[bounding_boxes_df['class'] == 1]
        if len(disc_box) == 0 or disc_box.values[0][5] < 0.2:
            print(f"Disc not detected with sufficient confidence in {filename}. Skipping image processing.")
            continue

        # Extract Bounding Box Coordinates
        disc_box = disc_box.values[0][0:6].tolist()
        macula_box = bounding_boxes_df[bounding_boxes_df['class'] == 0].values[0][0:6].tolist() if 0 in bounding_boxes_df['class'].values else None

        if macula_box and macula_box[5] > 0.4:
            bounding_boxes = {"disc": disc_box, "center": macula_box}
        else:
            retina_box = bounding_boxes_df[bounding_boxes_df['class'] == 2]
            if len(retina_box) == 0:
                print(f"Retina bounding box not found in {filename}. Skipping image processing.")
                continue
            bounding_boxes = {"disc": disc_box, "center": retina_box.values[0][0:6].tolist()}

        # Calculate Center and Radius
        macula_center = ((bounding_boxes["center"][1]), 
                         (bounding_boxes["center"][2]))
        disc_center = ((bounding_boxes["disc"][1]), 
                       (bounding_boxes["disc"][2]))
        centers_distance = distance(macula_center, disc_center)
        radius = int(1.5 * centers_distance) if macula_box and macula_box[5] > 0.5 else int(2 * centers_distance)

        # Create Circular Mask
        mask = np.zeros(gray_img.shape, dtype=np.uint8)
        cv2.circle(mask, (int(macula_center[0]), int(macula_center[1])), radius, 255, -1)

        # Apply Mask
        cropped_background = cv2.bitwise_and(img, img, mask=mask)

        # Crop Image
        cropped_image = crop_img(cropped_background)

        # Resize Image
        resized_image = cv2.resize(cropped_image, (700, 700))

        # Write Result
        output_path = os.path.join(output_dir, f"cropped_{filename}")
        cv2.imwrite(output_path, resized_image)

        print(f"Processed {filename} and saved as {output_path}")

print("All images processed.")