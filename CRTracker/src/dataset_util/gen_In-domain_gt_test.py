import os

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path)          
	else:
		print("The folder already exists, no need to create it")

if __name__ == '__main__':

    tackled_gt_text_path_list = [
        "/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/labels_with_ids_text/test/tackled_text/Circle.txt",
        "/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/labels_with_ids_text/test/tackled_text/Gate2.txt",
        "/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/labels_with_ids_text/test/tackled_text/Side.txt"
    ]

    for i, tackled_gt_text_path in enumerate(tackled_gt_text_path_list):
        txtFile = open(os.path.join(tackled_gt_text_path),'r')
        for line in txtFile.readlines():    
            temp = line.strip()
            data = temp.split(':') # Split by:
            full_name = temp
            sence = data[0].split('_')[0]
            ids = data[1].split(',')

            target_dir = "/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/labels_with_ids_text/test/gt_test/" + str(full_name) + "/" + "gt"
            mkdir(target_dir)

            # Set the maximum number of views to 3
            views = ["View1", "View2", "View3"]

            # Read the corresponding gt from 3 views
            for j, view in enumerate(views):
                with open(os.path.join(target_dir, str(view) + ".txt"), "w") as f: 
                    gt_path = "/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/images/test/" + str(sence) + "_" + str(view) + "/gt/gt.txt"
                    # Determine whether the view exists
                    if True == os.path.exists(gt_path): 
                        gt_File = open(os.path.join(gt_path),'r')
                        for line in gt_File.readlines():
                            temp = line.strip()
                            data = temp.split(',') # Split by,
                            gt_id = data[1]
                            if gt_id in ids:
                                f.write(line) 
                    else:
                        print("Path does not exist")
                        print("gt_path:", gt_path)
