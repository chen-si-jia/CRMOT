# Function: Integrate inference results and true value labels
# Note: gt_box_type、track_box_type need to be noted.

import os
import numpy as np


# Processing multiple folders under one folder:

# CRTrack In-domain:
file_dir_name_path = 'D:/桌面/CRMOT_evaluation/data/In-domain/eval/20241223'
save_dir_name_path = 'D:/桌面/CRMOT_evaluation/data/In-domain/eval/20241223_eval'
gt_dir = "D:/桌面/CRMOT_evaluation/data/In-domain/gt/sorted_gt_test_In-domain"

# # CRTrack Cross-domain:
# file_dir_name_path = 'D:/桌面/CRMOT_evaluation/data/Cross-domain/eval/20241223'
# save_dir_name_path = 'D:/桌面/CRMOT_evaluation/data/Cross-domain/eval/20241223_eval'
# gt_dir = "D:/桌面/CRMOT_evaluation/data/Cross-domain/gt/sorted_gt_test_Cross-domain"

files_name = os.listdir(file_dir_name_path)

for file_name in files_name:
	file_name_path = os.path.join(file_dir_name_path, file_name)
	track_dir = file_name_path
	save_dir  = os.path.join(save_dir_name_path, file_name + "_eval")

	gt_folder = "gt"

	gt_box_type = 'xywh' # coordinate format of ground truth
	delimiter = ',' # delimiter of inference results

	# track_box_type = 'xywh'
	track_box_type = 'xyxy' # coordinate format of inference results
	track_delimiter = ',' # delimiter of inference results

	scale = 1

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	save_gt_dir = save_dir+"/gt"
	if not os.path.exists(save_gt_dir):
		os.mkdir(save_gt_dir)

	save_gt_cvma_dir = save_dir+"/gt_cvma"
	if not os.path.exists(save_gt_cvma_dir):
		os.mkdir(save_gt_cvma_dir)

	save_track_dir = save_dir+"/track"
	if not os.path.exists(save_track_dir):
		os.mkdir(save_track_dir)

	save_track_cvma_dir = save_dir+"/track_cvma"
	if not os.path.exists(save_track_cvma_dir):
		os.mkdir(save_track_cvma_dir)

	seq_map_path = save_dir+"/seqs.txt"
	if os.path.exists(seq_map_path):
		os.remove(seq_map_path)
	seq_map_file = open(seq_map_path, 'a')
	seq_map_file.write('%s\n' % ('MOT16'))

	scene_list = os.listdir(gt_dir)
	for n in range(len(scene_list)):
		scene_id = scene_list[n]
		gt_vid_dir = gt_dir+"/"+scene_id+"/"+gt_folder
		track_vid_dir = track_dir+"/"+scene_id

		seq_map_file.write('%s\n' % (scene_id))
		
		save_gt_scene_path = save_gt_dir+"/"+scene_id+".txt"
		if os.path.exists(save_gt_scene_path):
			os.remove(save_gt_scene_path)
		save_gt_file = open(save_gt_scene_path, 'a')

		save_gt_cvma_scene_path = save_gt_cvma_dir+"/"+scene_id+".txt"
		if os.path.exists(save_gt_cvma_scene_path):
			os.remove(save_gt_cvma_scene_path)
		save_gt_cvma_file = open(save_gt_cvma_scene_path, 'a')
			
		save_track_scene_path = save_track_dir+"/"+scene_id+".txt"
		if os.path.exists(save_track_scene_path):
			os.remove(save_track_scene_path)
		save_track_file = open(save_track_scene_path, 'a')

		save_track_cvma_scene_path = save_track_cvma_dir+"/"+scene_id+".txt"
		if os.path.exists(save_track_cvma_scene_path):
			os.remove(save_track_cvma_scene_path)
		save_track_cvma_file = open(save_track_cvma_scene_path, 'a')

		vid_list = os.listdir(gt_vid_dir)
		print("vid_list:", vid_list)
		fr_cnt = 0
		for m in range(len(vid_list)):
			vid_name = vid_list[m]
			gt_path = gt_vid_dir+"/"+vid_name
			track_path = track_vid_dir+"/"+vid_name

			# convert format
			gt_data = np.loadtxt(gt_path, delimiter=delimiter, dtype=str)
			max_fr = 0
			for k in range(len(gt_data)):
				# Exception handling
				if 2 != gt_data.ndim:
					break

				max_fr = max(max_fr, int(gt_data[k, 0]))

				if gt_box_type=='xywh':
					save_gt_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
						% (int(gt_data[k, 0])+fr_cnt, int(gt_data[k, 1]), float(gt_data[k, 2]), float(gt_data[k, 3]),
						float(gt_data[k, 4]), float(gt_data[k, 5]), -1, -1, -1, -1))

					save_gt_cvma_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
						% (int(gt_data[k, 0])*len(vid_list)+m, int(gt_data[k, 1]), float(gt_data[k, 2]), float(gt_data[k, 3]),
						float(gt_data[k, 4]), float(gt_data[k, 5]), -1, -1, -1, -1))

				elif gt_box_type=='xyxy':
					save_gt_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
						% (int(gt_data[k, 0])+fr_cnt, int(gt_data[k, 1]), float(gt_data[k, 2]), float(gt_data[k, 3]),
						float(gt_data[k, 4])-float(gt_data[k, 2]), float(gt_data[k, 5])-float(gt_data[k, 3]), -1, -1, -1, -1))

					save_gt_cvma_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
						% (int(gt_data[k, 0])*len(vid_list)+m, int(gt_data[k, 1]), float(gt_data[k, 2]), float(gt_data[k, 3]),
						float(gt_data[k, 4])-float(gt_data[k, 2]), float(gt_data[k, 5])-float(gt_data[k, 3]), -1, -1, -1, -1))
		
			track_data = np.loadtxt(track_path, delimiter=track_delimiter, dtype=str)

			
			for k in range(len(track_data)):
				# Exception handling
				if 2 != track_data.ndim:
					break

				max_fr = max(max_fr, int(track_data[k, 0]))

				if track_box_type=='xywh':
					save_track_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
						% (int(track_data[k, 0])+fr_cnt, int(track_data[k, 1]), float(track_data[k, 2])*scale, float(track_data[k, 3])*scale,
						float(track_data[k, 4])*scale, float(track_data[k, 5])*scale, -1, -1, -1, -1))

					save_track_cvma_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
						% (int(track_data[k, 0])*len(vid_list)+m, int(track_data[k, 1]), float(track_data[k, 2])*scale, float(track_data[k, 3])*scale,
						float(track_data[k, 4])*scale, float(track_data[k, 5])*scale, -1, -1, -1, -1))
				elif track_box_type=='xyxy':
					save_track_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
						% (int(track_data[k, 0])+fr_cnt, int(track_data[k, 1]), float(track_data[k, 2])*scale, float(track_data[k, 3])*scale,
						(float(track_data[k, 4])-float(track_data[k, 2]))*scale, (float(track_data[k, 5])-float(track_data[k, 3]))*scale, -1, -1, -1, -1))

					save_track_cvma_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
						% (int(track_data[k, 0])*len(vid_list)+m, int(track_data[k, 1]), float(track_data[k, 2])*scale, float(track_data[k, 3])*scale,
						(float(track_data[k, 4])-float(track_data[k, 2]))*scale, (float(track_data[k, 5])-float(track_data[k, 3]))*scale, -1, -1, -1, -1))

			max_fr += 1
			fr_cnt += max_fr
		
		save_gt_file.close()
		save_track_file.close()
		save_gt_cvma_file.close()
		save_track_cvma_file.close()
	