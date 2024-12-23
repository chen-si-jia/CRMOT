import sys, os
sys.path.append(os.getcwd())
import argparse

import traceback
import time
import pickle
import pandas as pd
import glob
from os import path
import numpy as np



class Evaluator(object):
	""" The `Evaluator` class runs evaluation per sequence and computes the overall performance on the benchmark"""
	def __init__(self):
		pass

	def run(self, benchmark_name = None ,  gt_dir = None, res_dir = None, save_pkl = None, eval_mode = "train", seq_file = None, filedir_name = None):
		"""
		Params
		-----
		benchmark_name: Name of benchmark, e.g. MOT17
		gt_dir: directory of folders with gt data, including the c-files with sequences
		res_dir: directory with result files
			<seq1>.txt
			<seq2>.txt
			...
			<seq3>.txt
		eval_mode:
		seqmaps_dir:
		seq_file: File name of file containing sequences, e.g. 'c10-train.txt'
		save_pkl: path to output directory for final results
		"""

		start_time = time.time()

		self.benchmark_gt_dir = gt_dir
		self.seq_file = seq_file# "{}-{}.txt".format(benchmark_name, eval_mode)

		res_dir = res_dir
		self.benchmark_name = benchmark_name
		#self.seqmaps_dir = seqmaps_dir

		self.mode = eval_mode

		self.datadir = gt_dir #os.path.join(gt_dir, self.mode)

		# getting names of sequences to evaluate
		error_traceback = ""
		assert self.mode in ["train", "test", "all"], "mode: %s not valid " %s

		print("Evaluating Benchmark: %s" % self.benchmark_name)

		# ======================================================
		# Handle evaluation
		# ======================================================


		# load list of all sequences
		self.sequences = []
		txtFile = open(self.seq_file,'rb')
		for line in txtFile.readlines(): 
			temp = line.strip()
			temp = str(temp, 'utf-8') # Convert b'1' to '1'
			self.sequences.append(temp)
		self.sequences = self.sequences[1:] 


		self.gtfiles = []
		self.tsfiles = []
		for seq in self.sequences:
			#gtf = os.path.join(self.benchmark_gt_dir, self.mode ,seq, 'gt/gt.txt')
			gtf = gt_dir+"/"+seq+'.txt'
			if path.exists(gtf): self.gtfiles.append(gtf)
			else: raise Exception("Ground Truth %s missing" % gtf)
			tsf = os.path.join( res_dir, "%s.txt" % seq)
			if path.exists(gtf): self.tsfiles.append(tsf)
			else: raise Exception("Result file %s missing" % tsf)

		print('Found {} ground truth files and {} test files.'.format(len(self.gtfiles), len(self.tsfiles)))
		print( self.tsfiles)

		self.MULTIPROCESSING = True
		MAX_NR_CORES = 4 # Set the number of CPU threads. 
		# Note: The better the CPU performance, the larger the value can be. However, the larger the value, the faster it is.
		
		
		# set number of core for mutliprocessing
		if self.MULTIPROCESSING: self.NR_CORES = np.minimum( MAX_NR_CORES, len(self.tsfiles))
		try:

			""" run evaluation """
			results = self.eval()

			# calculate overall results
			results_attributes = self.Overall_Results.metrics.keys()

			for attr in results_attributes:
				""" accumulate evaluation values over all sequences """
				try:
					self.Overall_Results.__dict__[attr] = sum(obj.__dict__[attr] for obj in self.results)
				except:
					pass
			cache_attributes = self.Overall_Results.cache_dict.keys()
			for attr in cache_attributes:
				""" accumulate cache values over all sequences """
				try:
					self.Overall_Results.__dict__[attr] = self.Overall_Results.cache_dict[attr]['func']([obj.__dict__[attr] for obj in self.results])
				except:
					pass
			print("evaluation successful")


			# Compute clearmot metrics for overall and all sequences
			for res in self.results:
				res.compute_clearmot()
			self.Overall_Results.compute_clearmot()


			self.accumulate_df(type = "mail")
			self.failed = False
			error = None


		except Exception as e:
			print(str(traceback.format_exc()))
			print ("<br> Evaluation failed! <br>")

			error_traceback+= str(traceback.format_exc())
			self.failed = True
			self.summary = None

		end_time=time.time()

		self.duration = (end_time - start_time)/60.



		# ======================================================
		# Collect evaluation errors
		# ======================================================
		if self.failed:

			startExc = error_traceback.split("<exc>")
			error_traceback = [m.split("<!exc>")[0] for m in startExc[1:]]

			error = ""

			for err in error_traceback:
				error+="Error: %s" % err

			print("Error Message", error)
			self.error = error
			print("ERROR %s" % error)

		print ("Evaluation Finished")
		print("Your Results")
		print(self.render_summary())

		# save results
		if save_pkl:
			from openpyxl import Workbook
			# self.summary。DataFrame type written to excel
			excel_name = filedir_name + ".xlsx"
			self.summary.to_excel(os.path.join(save_pkl, excel_name))
			print("Successfully save results")

	def eval(self):
		raise NotImplementedError


	def accumulate_df(self, type = None):
		""" create accumulated dataframe with all sequences """
		single_scene_results = {}

		total_scene_results = {}
		total_scene_results["All scenes"] = {}
		total_scene_results["All scenes"]["MOTA"] = 0
		total_scene_results["All scenes"]["IDF1"] = 0
		total_scene_results["All scenes"]["seqNumber"] = 0

		for k, res in enumerate(self.results):
			# If CVRMA is negative, it is set to 0.
			if(res.MOTA < 0):
				res.MOTA = max(res.MOTA, 0)

			scene = res.seqName.split("_")[0]
			if scene not in single_scene_results:
				single_scene_results[scene] = {}
				single_scene_results[scene]["MOTA"] = res.MOTA
				single_scene_results[scene]["IDF1"] = res.IDF1
				single_scene_results[scene]["seqNumber"] = 1
			else:
				single_scene_results[scene]["MOTA"] += res.MOTA
				single_scene_results[scene]["IDF1"] += res.IDF1
				single_scene_results[scene]["seqNumber"] += 1

			total_scene_results["All scenes"]["MOTA"] += res.MOTA		
			total_scene_results["All scenes"]["IDF1"] += res.IDF1
			total_scene_results["All scenes"]["seqNumber"] += 1
			
			res.to_dataframe(display_name = True, type = type )
			if k == 0: summary = res.df
			else: summary = summary.append(res.df)
		summary = summary.sort_index()
			
		# Statistics of each scene and total
		for k, res in enumerate(self.results):
			# Only use the first one as a template for processing
			if 0 == k:
				# Iterate over the keys of a dictionary
				for scene in single_scene_results.keys():
					print(scene)
					print("single_scene_results[scene][seqNumber]:", single_scene_results[scene]["seqNumber"])
					res.seqName = scene
					res.MOTA = single_scene_results[scene]["MOTA"] / single_scene_results[scene]["seqNumber"]
					res.IDF1 = single_scene_results[scene]["IDF1"] / single_scene_results[scene]["seqNumber"]
					res.to_dataframe(display_name = True, type = type )
					summary = summary.append(res.df)
				# Iterate over the keys of a dictionary
				for scene in total_scene_results.keys():
					print(scene)
					print("total_scene_results[All scenes][seqNumber]:", total_scene_results["All scenes"]["seqNumber"])
					res.seqName = scene
					res.MOTA = total_scene_results[scene]["MOTA"] / total_scene_results["All scenes"]["seqNumber"]
					res.IDF1 = total_scene_results[scene]["IDF1"] / total_scene_results["All scenes"]["seqNumber"]
					res.to_dataframe(display_name = True, type = type )
					summary = summary.append(res.df)
			else:
				break
				
		self.summary = summary

	def render_summary( self, buf = None):
		"""Render metrics summary to console friendly tabular output.

		Params
		------
		summary : pd.DataFrame
		    Dataframe containing summaries in rows.

		Kwargs
		------
		buf : StringIO-like, optional
		    Buffer to write to
		formatters : dict, optional
		    Dicionary defining custom formatters for individual metrics.
		    I.e `{'mota': '{:.2%}'.format}`. You can get preset formatters
		    from MetricsHost.formatters
		namemap : dict, optional
		    Dictionary defining new metric names for display. I.e
		    `{'num_false_positives': 'FP'}`.

		Returns
		-------
		string
		    Formatted string
		"""
		output = self.summary.to_string(
			buf=buf,
			formatters=self.Overall_Results.formatters,
			justify = "left"
		)

		return output
def run_metrics( metricObject, args ):
	""" Runs metric for individual sequences
	Params:
	-----
	metricObject: metricObject that has computer_compute_metrics_per_sequence function
	args: dictionary with args for evaluation function
	"""
	metricObject.compute_metrics_per_sequence(**args)
	return metricObject


if __name__ == "__main__":
	Evaluator()