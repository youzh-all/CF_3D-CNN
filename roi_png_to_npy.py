#%%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Folder path setting
folder_name = "CF_V4_20240821_193920"
folder_path = f"output_without/{folder_name}"

# cultivar_name = TripleD, ga20ox2, ga20ox4
# cultivar_name setting
cultivar_name = "TripleD"


# Get list of files in the folder
file_list = [file for file in os.listdir(folder_path) if file.endswith('.png')]
file_list.sort()  # Sorting (optional)
print(f"PNG file order in the folder: {file_list}")

if len(file_list) == 0:
    raise ValueError("There are no PNG files in the folder.")

# Read the last file
last_image_path = os.path.join(folder_path, file_list[-14]) # -14 : NPQ_L2
last_image = cv2.imread(last_image_path, cv2.IMREAD_COLOR)
last_image = cv2.cvtColor(last_image, cv2.COLOR_BGR2RGB)

roi = None  # Variable to store ROI information
# List to store ROI information
rois = []
# ROI size setting
n = 5

# ROI selection function
def on_mouse(event, x, y, flags, param):
    global preview_img  # Preview image
    
    if event == cv2.EVENT_MOUSEMOVE:
        preview_img = clone.copy()  # Copy the original image
        # Preview box display
        cv2.rectangle(preview_img, (x, y), (x + n, y + n), (0, 255, 0), 1)
        cv2.imshow("Select ROI", preview_img)
    
    elif event == cv2.EVENT_LBUTTONDOWN:
        roi = [x, y, x + n, y + n]
        rois.append(roi)
        # Confirmed ROI is displayed in blue
        cv2.rectangle(clone, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 1)
        cv2.imshow("Select ROI", clone)

# ROI selection interface
preview_img = None  # Global variable to store the preview image
clone = last_image.copy()
cv2.imshow("Select ROI", clone)
cv2.setMouseCallback("Select ROI", on_mouse, param=clone)
cv2.waitKey(0)
cv2.destroyAllWindows()

if not rois:
    raise ValueError("No ROI was selected.")

# Extract each ROI and create a 3D array for all files
roi_stacks = []
for roi in rois:
    x1, y1, x2, y2 = roi
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    
    roi_stack = []
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        roi_crop = image[y1:y2, x1:x2]
        roi_stack.append(roi_crop)
    
    roi_stacks.append(np.array(roi_stack))

# Save Numpy files for each ROI
for idx, roi_stack in enumerate(roi_stacks):
    output_file = f"ROI_3D_CF/{folder_name}_{cultivar_name}_{idx}.npy"
    np.save(output_file, roi_stack)
    print(f"3D ROI data saved to {output_file}. Size: {roi_stack.shape}")

# %%
