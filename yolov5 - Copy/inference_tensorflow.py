import os
import cv2
import numpy as np
import tensorflow as tf

# Load TensorFlow model
model_path = 'crop/retinagpuS/weights/TFModel'  # Replace with actual path
try:
    model = tf.saved_model.load(model_path)
    signature = model.signatures['serving_default']  # Assuming 'serving_default' is the correct signature
except Exception as e:
    print(f"Error loading the TensorFlow model from {model_path}: {e}")
    exit(1)

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

# Process each image in the folder
input_dir = '../../Test_new/Original'
output_dir = '../../Test_new/Cropped_TF'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.jpg'):
        # Read Image
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        # Convert to Grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Prepare input for TensorFlow model
        input_tensor = tf.convert_to_tensor(gray_img[np.newaxis, ..., np.newaxis], dtype=tf.float32)

        # Inference
        try:
            results = signature(input_tensor=input_tensor)
        except Exception as e:
            print(f"Error occurred during inference for {filename}: {e}")
            continue

        # Example processing: assuming output is in 'detection_boxes'
        boxes = results['detection_boxes'][0].numpy()

        # Example processing: assuming output is in 'detection_classes' and 'detection_scores'
        classes = results['detection_classes'][0].numpy().astype(np.int32)
        scores = results['detection_scores'][0].numpy()

        # Convert boxes to xyxy format
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 1] * img.shape[1]  # ymin
        boxes_xyxy[:, 1] = boxes[:, 0] * img.shape[0]  # xmin
        boxes_xyxy[:, 2] = boxes[:, 3] * img.shape[1]  # ymax
        boxes_xyxy[:, 3] = boxes[:, 2] * img.shape[0]  # xmax

        # Convert boxes to pandas DataFrame for easier manipulation
        import pandas as pd
        boxes_df = pd.DataFrame(boxes_xyxy, columns=['xmin', 'ymin', 'xmax', 'ymax'])

        # Example processing: filtering based on confidence and class
        disc_box = boxes_df[(classes == 0) & (scores > 0.55)]
        macula_box = boxes_df[(classes == 1) & (scores >= 0.7)]
        retina_box = boxes_df[(classes == 2) & (scores > 0.55)]

        if len(disc_box) == 0:
            print(f"Disc not detected with sufficient confidence in {filename}. Skipping image processing.")
            continue

        disc_box = disc_box.values[0].tolist()

        if len(macula_box) > 0:
            macula_box = macula_box.values[0].tolist()
            bounding_boxes = {"disc": disc_box, "center": macula_box}
        else:
            if len(retina_box) == 0:
                print(f"Macula and retina bounding boxes not found in {filename}. Skipping image processing.")
                continue
            bounding_boxes = {"disc": disc_box, "center": retina_box.values[0].tolist()}

        # Calculate Center and Radius
        macula_center = ((bounding_boxes["center"][0] + bounding_boxes["center"][2]) // 2, 
                         (bounding_boxes["center"][1] + bounding_boxes["center"][3]) // 2)
        disc_center = ((bounding_boxes["disc"][0] + bounding_boxes["disc"][2]) // 2, 
                       (bounding_boxes["disc"][1] + bounding_boxes["disc"][3]) // 2)
        centers_distance = distance(macula_center, disc_center)
        radius = int(1.5 * centers_distance) if len(macula_box) > 0 else int(2 * centers_distance)

        # Create Circular Mask
        mask = np.zeros(gray_img.shape, dtype=np.uint8)
        cv2.circle(mask, (int(macula_center[0]), int(macula_center[1])), radius, 255, -1)

        # Apply Mask
        cropped_background = cv2.bitwise_and(img, img, mask=mask)

        # Crop Image
        cropped_image = crop_img(cropped_background)

        # Resize Image
        resized_image = cv2.resize(cropped_image, (1200, 1200))

        # Write Result
        output_path = os.path.join(output_dir, f"cropped_{filename}")
        cv2.imwrite(output_path, resized_image)

        print(f"Processed {filename} and saved as {output_path}")

print("All images processed.")