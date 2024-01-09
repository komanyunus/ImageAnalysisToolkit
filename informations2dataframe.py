

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
import pandas as pd
import openpyxl

# Define the range of image indices
image_indices = range(7002, 8357)
# Create a list to store results
results_list = []

# Create an empty Pandas DataFrame to store results
columns = ['Image_Index', 'Mean_Distance', 'Left_Line', 'Right_Line', 'Mean_Pixel_Value']
df_results = pd.DataFrame(columns=columns)

for image_index in image_indices:
    try:
        # Read an image
        image_path = f"C:\\Users\\user\\cropped_images\\cropped_DSCF{image_index}.jpg"
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply multi-Otsu threshold
        thresholds = threshold_multiotsu(gray, classes=3)

        # Digitize (segment) original image into multiple classes.
        regions = np.digitize(gray, bins=thresholds)
        output = np.uint8(regions)  # Convert 64-bit integer values to uint8

        # Find the index of the threshold with the maximum pixel count
        max_count_index = np.argmax(np.bincount(output.flat))

        # Create a mask to extract only the region corresponding to the maximum pixel count
        max_count_mask = output == max_count_index

        # Apply the mask to the original image
        segmented_part = np.copy(image)
        segmented_part[~np.stack([max_count_mask]*3, axis=-1)] = 0  # Set non-segmented pixels to black

        # Apply edge detection (you can use any edge detection method of your choice)
        edges = cv2.Canny(segmented_part, 50, 150, apertureSize=3)

        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=150, maxLineGap=100)

        # Filter lines to keep only those close to vertical (e.g., within a range of angles)
        angle_threshold = 10  # Adjust as needed
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if np.abs(angle - 90) < angle_threshold:
                vertical_lines.append(line)

        # Sort the vertical lines based on their x-coordinates
        vertical_lines.sort(key=lambda x: (x[0][0] + x[0][2]) / 2)

        # Calculate mean distance between the vertical lines closest to left and right of the image
        left_line = vertical_lines[0][0]
        right_line = vertical_lines[-1][0]
        mean_distance = np.sqrt((left_line[2] - right_line[0])**2 + (left_line[3] - right_line[1])**2)

        # Calculate mean pixel value of the segmented part
        mean_pixel_value = np.mean(segmented_part[max_count_mask])

         # Append the results to the list
        results_list.append({
            'Image_Index': image_index,
            'Mean_Distance': mean_distance,
            'Left_Line': left_line,
            'Right_Line': right_line,
            'Mean_Pixel_Value': mean_pixel_value
        })

    except Exception as e:
        print(f"Error processing image {image_index}: {e}")

# Create the DataFrame once with all the data
df_results = pd.DataFrame(results_list)

# Print the resulting DataFrame
print(df_results)

df2 = df_results.copy()
with pd.ExcelWriter('output2.xlsx') as writer:  
    df2.to_excel(writer, sheet_name='Sheet1')
    
    
    
    



#from sklearn.ensemble import IsolationForest
# import matplotlib.pyplot as plt

# # Assuming df_results is your DataFrame containing 'Mean_Distance' column
# # If not, replace it with your DataFrame name
# X = df_results[['Mean_Distance']]

# # Fit the Isolation Forest model
# clf = IsolationForest(contamination=0.25)  # Adjust contamination as needed
# clf.fit(X)

# # Predict the anomalies
# df_results['Anomaly'] = clf.predict(X)

# # Plot the anomalies
# plt.figure(figsize=(10, 6))
# plt.scatter(df_results['Image_Index'], df_results['Mean_Distance'], c=df_results['Anomaly'], cmap='viridis')
# plt.title('Anomaly Detection on Mean Distance')
# plt.xlabel('Image Index')
# plt.ylabel('Mean Distance')
# plt.show()

# X = df_results[['Mean_Distance']]

# # Fit the Isolation Forest model
# clf = IsolationForest(contamination=0.25)  # You can adjust the contamination parameter
# clf.fit(X)

# # Predict whether each instance is an outlier or not
# df_results['Is_Outlier'] = clf.predict(X)

# # Plotting
# plt.figure(figsize=(10, 6))
# colors = np.where(df_results['Is_Outlier'] == 1, 'blue', 'red')  # Blue for inliers, red for outliers

# plt.scatter(df_results['Image_Index'], df_results['Mean_Distance'], c=colors, label='Anomaly')
# plt.xlabel('Image Index')
# plt.ylabel('Mean Distance')
# plt.title('Anomaly Detection using Isolation Forest')
# plt.legend()
# plt.show()