# Function: Visualize the CRTracker_In-domain inference results of CRMOT. Save pictures and videos.
# Author: Sijia Chen, Huazhong University of Science and Technology
# Time：20250304

import os
import shutil
import cv2
import colorsys


# Create a folder
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} create Success")
    else:
        print(f"Folder {folder_path} already exists")


# Clear the contents of the folder
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # If it is a file, delete the file
        if os.path.isfile(file_path):
            os.remove(file_path)
        # If it is a folder, delete it recursively
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


# ################################### Now: On the box, write the ID in the format "ID: ID number" #################################### #
def rectangle_bbox_score(image, x, y, w, h, color, thickness=2, label=None):
    """Draw a rectangle
    Parameters
    ----------
    label : Optional[str], id
    """
    
    pt1 = (int(x), int(y))
    pt2 = (int(x + w), int(y + h))
    cv2.rectangle(image, pt1, pt2, color, thickness)
    if label is not None:
        label = "ID:" + label # Change label
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)
        
        pt1 = int(x), int(y - 5 - text_size[0][1]) # Modify the starting point

        center = pt1[0] + 5, pt1[1] + text_size[0][1] - 2

        pt1 = int(x - 1), int(y - 3*text_size[0][1] - 10) # Modify the starting point

        pt2 = pt1[0] + 3*text_size[0][0] - 10, pt1[1] + 3*text_size[0][1] + 10 # Modify the end point 

        cv2.rectangle(image, pt1, pt2, color, -4)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    2.5, (255, 255, 255), thickness) # Font size, color, thickness
        
        # print("label postion:", center)



if __name__ == '__main__':

    # txt file path of the inference result
    sequence_dir_absolute_path = "/mnt/A/hust_csj/Code/Github/CRMOT/CRMOT_evaluation/data/In-domain/Inference_results/CRTracker_In-domain/Side_A man wearing a black coat and black trousers"
    
    for view in ["View1", "View2", "View3"]:

        sequence_absolute_path = os.path.join(sequence_dir_absolute_path, view + ".txt")
        
        # Read the txt file of the inference result
        result_txt = []
        with open(sequence_absolute_path, 'r', encoding='utf-8') as file:
            for line in file:
                values = line.strip().split(",")
                # Parsing the data
                img_name = int(values[0])
                id = int(values[1])
                x1 = int(float(values[2]))
                y1 = int(float(values[3]))
                x2 = int(float(values[4]))
                y2 = int(float(values[5]))

                # xyxy -> xywh
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1

                result_txt.append([img_name, id, x, y, w, h])

        # Get the language description corresponding to the txt
        description = sequence_dir_absolute_path.split("/")[-1] + "_" + view

        # The save path of the picture after drawing the frame
        drawed_img_dir_path = os.path.join("/mnt/A/hust_csj/Code/Github/CRMOT/CRMOT_evaluation/Visualization/results", description, "img")

        # Create a folder
        create_folder_if_not_exists(drawed_img_dir_path) 
        # If it already exists, clear the contents of the folder.
        clear_folder(drawed_img_dir_path) 

        # Get the image path collection corresponding to the txt
        imgs_txt_path = os.path.join("/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain", "images", "test", sequence_dir_absolute_path.split("/")[-1].split("_")[0] + "_" + view, "img1")
        frame_names = os.listdir(imgs_txt_path)
        frame_names.sort()

        # Traverse all pictures
        for frame_name in frame_names:
            frame_name_path = os.path.join(imgs_txt_path, frame_name)

            # Read pictures
            image = cv2.imread(frame_name_path)

            if image is None:
                print("Image loading failed")
            
            # Draw multiple rectangles on the image
            for result in result_txt:
                result_img_name = result[0]
                result_id = result[1]
                result_x = result[2]
                result_y = result[3]
                result_w = result[4]
                result_h = result[5]
                # Find the results corresponding to the pictures we want
                if (int(frame_name.split("_")[-1].split(".jpg")[0]) == result_img_name):
                    # Draw a rectangular frame
                    color = create_unique_color_uchar(result_id)
                    rectangle_bbox_score(image, result_x, result_y, result_w, result_h, color, label=str(result_id))

            # Save the modified image
            drawed_img_path = os.path.join(drawed_img_dir_path, frame_name.split("/")[-1])
            cv2.imwrite(drawed_img_path, image)
        
        # =================  Merge images into video  ====================
        # Image folder path
        folder_path = drawed_img_dir_path

        # The save path of the video after drawing the frame
        drawed_videos_dir_path = os.path.join("/mnt/A/hust_csj/Code/Github/CRMOT/CRMOT_evaluation/Visualization/results", description, "video")
        # Create a folder
        create_folder_if_not_exists(drawed_videos_dir_path) 
        # If it already exists, clear the contents of the folder.
        clear_folder(drawed_videos_dir_path) 

        # Get a list of image files in .jpg format
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        
        # Get the size of the first image
        image_path = os.path.join(folder_path,image_files[0])
        first_image = cv2.imread(image_path)
        height,width,channels = first_image.shape
        
        # Set the video output path and related parameters
        output_path = os.path.join(drawed_videos_dir_path, view + ".mp4")
        fps = 30.0 # Frame rate
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create a video writer object
        video_writer = cv2.VideoWriter(output_path,fourcc,fps,(width,height))
        
        # Write images to video frame by frame
        for image_file in image_files:
            image_path = os.path.join(folder_path,image_file)
            image = cv2.imread(image_path)
            video_writer.write(image)
        
        # Release resources
        video_writer.release()
        