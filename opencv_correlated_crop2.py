
import cv2
import os

# Folder paths
input_folder = r"C:\Users\user\0391TS"
output_folder = r"C:\Users\user\cropped_images"
# Load the template
template = cv2.imread(r"C:\Users\user\template_7019.jpg", 0)
h, w = template.shape[::-1]  # Fix the shape tuple

# Loop through images from 7002 to 8356
for image_number in range(7002, 8357):
    try:
        # Input image path
        input_image_path = os.path.join(input_folder, f"DSCF{image_number}.JPG")
    
        # Read the input image
        img_rgb = cv2.imread(input_image_path)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    

    
        # Match template
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
    
        # Crop the rectangle part
        cropped_image = img_rgb[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
        # Output image path
        output_image_path = os.path.join(output_folder, f"cropped_DSCF{image_number}.jpg")
    
        # Save the cropped image
        cv2.imwrite(output_image_path, cropped_image)
    except Exception as e:
            print(f"Error processing image {image_number}: {e}")

print("Cropping and saving complete.")