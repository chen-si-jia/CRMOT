
# Function: Rename the inference result folder
# Note: If the folder path is too long, os.rename() will fail 
# and report an error: FileNotFoundError: [WinError 3] The system cannot find the specified path.


import numpy as np
import os,sys

def rename_subfolders(path):    
    old_names = os.listdir(path) 
    for old_name in old_names:     
        if old_name!= sys.argv[0]:   
            new_name = old_name.split("._")[0] 
            os.rename(os.path.join(path,old_name),os.path.join(path,new_name)) 
            print(old_name,"has been renamed successfully! New name is: ",new_name) 


if __name__ == '__main__': 
    # ############################## Method 1 ###################################### # 
    # Processing an inference result：
    # file_name_path = 'D:/桌面/CRMOT_evaluation/data/In-domain/Inference_results/CRTracker_In-domain'
    # rename_subfolders(file_name_path)

    # ############################## Method 2 ###################################### # 
    # # Processing multiple inference results：
    file_dir_name_path = 'D:/桌面/CRMOT_evaluation/data/In-domain/Inference_results'
    # file_dir_name_path = 'D:/桌面/CRMOT_evaluation/data/Cross-domain/Inference_results'
    files_name = os.listdir(file_dir_name_path)
    for file_name in files_name:
        file_name_path = os.path.join(file_dir_name_path, file_name)
        rename_subfolders(file_name_path)
