import torch
import cv2
import numpy as np

# Load Model
model = torch.hub.load('', 'custom', path='runs/train/Initial_good/weights/best.pt', source='local')  # local repo

# Read Image
img = cv2.imread('../../1kTesting/G008047_20201106174142_1.jpg')

# Convert to Grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Inference
results = model(gray_img, size=1200)

# Extract Bounding Boxes
boxes = results.pandas().xyxy[0]
print(boxes)

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

# Extract Bounding Box Coordinates
disc_box = boxes[boxes['name'] == 'disc'].values[0][0:5].tolist()
macula_box = boxes[boxes['name'] == 'macula'].values[0][0:5].tolist() if 'macula' in boxes['name'].values else None

if macula_box and macula_box[4] > 0.5:
    bounding_boxes = {"disc": disc_box, "center": macula_box}
else:
    bounding_boxes = {"disc": disc_box, "center": boxes[boxes['name'] == 'retina'].values[0][0:5].tolist()}

# Calculate Center and Radius
macula_center = ((bounding_boxes["center"][0] + bounding_boxes["center"][2]) // 2, 
                 (bounding_boxes["center"][1] + bounding_boxes["center"][3]) // 2)
disc_center = ((bounding_boxes["disc"][0] + bounding_boxes["disc"][2]) // 2, 
               (bounding_boxes["disc"][1] + bounding_boxes["disc"][3]) // 2)
centers_distance = distance(macula_center, disc_center)
radius = int(1.5 * centers_distance) if macula_box and macula_box[4] > 0.5 else int(2 * centers_distance)

# Create Circular Mask
mask = np.zeros(gray_img.shape, dtype=np.uint8)
cv2.circle(mask, (int(macula_center[0]), int(macula_center[1])), radius, 255, -1)

# Apply Mask
cropped_background = cv2.bitwise_and(img, img, mask=mask)

# Crop Image
cropped_image = crop_img(cropped_background)

# Resize Image
resized_image = cv2.resize(cropped_image, (700, 700))

# Show Result
# cv2.imwrite('runs/detect/exp2/cropped_study_90.jpg',resized_image)
cv2.imshow("Cropped Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()