
# Function: Processing the GT folders, sorting and renaming

import numpy as np
import os,sys

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)           
	else:
		print("The folder already exists, no need to create it")

def rename_subfolders(path):     
    old_names = os.listdir(path) 
    for old_name in old_names:   
        if old_name!= sys.argv[0]:   
            new_name = old_name.split("._")[0] 
            os.rename(os.path.join(path,old_name),os.path.join(path,new_name)) 
            print(old_name,"has been renamed successfully! New name is: ",new_name) 


if __name__ == '__main__': 
    # CRTrack In-domain:
    file_name_path = 'D:/桌面/CRMOT_evaluation/data/In-domain/eval/sorted_gt_test_In-domain'

    # CRTrack Cross-domain:
    # file_name_path = 'D:/桌面/CRMOT_evaluation/data/In-domain/eval/sorted_gt_test_Cross-domain'

    dir_files = os.listdir(file_name_path)
    # Sort by frame number:
    for i, dir_file in enumerate(dir_files):
        views = ["View1", "View2", "View3", "View4"] # Maximum number of views
        # Read the corresponding gt from multiple views
        for j, view in enumerate(views):
            gt_text_path = file_name_path + "/" + dir_file + "/gt/" + str(view) + ".txt"
            # If the text file exists
            if(True == os.path.exists(gt_text_path)):
                # Read CSV file and save as 2D array
                sorted_data = np.genfromtxt(gt_text_path, delimiter=',')
                # If it is a two-dimensional array
                if 2 == sorted_data.ndim:
                    # Sort each row according to the specified order. 
                    # First press the first column, then the second column, sort from small to large
                    sorted_data = sorted(sorted_data, key=lambda x: (x[0], x[1]))

                    np.savetxt(gt_text_path, sorted_data, delimiter=',', fmt='%d')
                else:
                    # There is only one row of data, which becomes empty directly
                    with open(gt_text_path, 'w') as f:
                        f.write("")

    # Rename the folder:
    rename_subfolders(file_name_path)
