import multiprocessing
import subprocess
from format_argo import argo_to_formatted, create_data
import argparse
import numpy as np
import os
from model.import_data import merge
from collections import defaultdict
import pickle
from klepto.archives import dir_archive

RAW_DATA = "./resources/raw_data/ARGO"
DATA_DIR = './resources/data/ARGO'

THREAD = 30


def single(input_dir, data_dir, dtype):

	if dtype == 'test':
		input_dir = os.path.join(input_dir, 'test_obs')	
	else:
		input_dir = os.path.join(input_dir, dtype)
	files = [f.split('.')[0] for f in os.listdir(input_dir)]
	create_data(input_dir, files, data_dir, dtype, 0)

def multi(input_dir, files, output_dir, data_dir, dtype, i):

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	output_dir = output_dir + "set{}/".format(str(i))
	lst = argo_to_formatted(input_dir, files, output_dir, dtype)
	create_data(output_dir, lst, data_dir, dtype, i)

def run_merge(input_dir, data_dir, dtype, i):
	input_dir = input_dir + "set{}/".format(str(i))
	files = [f for f in os.listdir(input_dir) if '.npy' in f]
	lst = [os.path.join(input_dir, f) for f in files]
	sz = merge(lst, data_dir, dtype, i)
	return sz

def klepto_dump(merged_dict, loc):
    '''
    to dump the merged dictionary file
    :param merged_dict: the final merged dictionary obtained
    :return: None
    '''

    demo = dir_archive(loc, merged_dict, serialized =True)
    demo.dump()
    del demo
    ## https://stackoverflow.com/questions/3957765/loading-a-large-dictionary-using-python-pickle


# def merge_dicts(file_names):
#     '''
#     To read all the .npy files, and merge all the track dictionaries into one dictionary
#     :param file_names: consists all the .npy file names in test data
#     :return: Returns the final merged dictionary
#     '''

#     i = 0

#     merged_dict = {}

#     for file in file_names:

#         # if i < 3:

#         file_path = files_path + '/' + file
#         data = np.load(file_path, allow_pickle=True)

#         dict_data = data[1]
#         # print('dict data len : ',len(dict_data))
#         for key, value in dict_data.items():
#             merged_dict[key] = value

#         print('merged file ', file, '\n')
#         i += 1

#     return merged_dict


def generate_sgan(data_dir, dtype):
	input_dir = os.path.join(data_dir, dtype)
	files = [f for f in os.listdir(input_dir) if '.txt' in f]
	out_name = "{}Set.txt".format(dtype)
	with open(os.path.join(data_dir, out_name), 'w') as outfile:
		i = 0
		for f in files:
			print("Loading #{} file in {}...".format(i, dtype))
			i+= 1
			with open(os.path.join(input_dir, f)) as infile:
				for line in infile:
					outfile.write(line)
		print("Finish loading all files in {}.".format(dtype))
	

def generate_data(data_dir, dtype, shape):
	input_dir = os.path.join(data_dir, dtype)
	traj_files = [f for f in os.listdir(input_dir) if '-traj.npy' in f]
	track_files = [f for f in os.listdir(input_dir) if '-track.npy' in f]

	out_traj = "{}Set-traj.dat".format(dtype)
	mmp = np.memmap(os.path.join(data_dir, out_traj), dtype=np.float64, mode='w+', shape=shape)

	out_track = "{}Set-track.kpt".format(dtype)

	i = 0
	s = 0
	e = 0
	for f in traj_files:
		print("Loading #{}-traj file in {}...".format(i, dtype))
		i+= 1
		data = np.load(os.path.join(input_dir, f), allow_pickle=True)
		traj = data[0]
		e = s + len(traj)
		mmp[s:e, :] = traj[:]
		s = e

	i = 0
	track = defaultdict(dict)
	for f in track_files:
		print("Loading #{}-track file in {}...".format(i, dtype))
		i+= 1
		data = np.load(os.path.join(input_dir, f), allow_pickle=True)
		track.update(data[0])

	# with open(os.path.join(data_dir, out_track), 'wb') as f:
	# 	pickle.dump(track, f)
	klepto_dump(track, os.path.join(data_dir, out_track))
	print("Finish loading all files in {}.".format(dtype))

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="multiprocessing data")

	parser.add_argument('--set', '-s', required=True, type=str)

	args = parser.parse_args()

	# dtype = args.set
	# pool = multiprocessing.Pool(processes=THREAD)
	# cmds = []


	if dtype == 'train':
		print('preprocessing train set...')
		input_dir = RAW_DATA + '/train/data/'
		output_dir = RAW_DATA + '/train/formatted/'

	elif dtype == 'val':
		print('preprocessing val set...')
		input_dir = RAW_DATA + '/val/data/'
		output_dir = RAW_DATA + '/val/formatted/'
	elif dtype == 'test':
		print('preprocessing test set...')
		input_dir = RAW_DATA + '/test_obs/data/'
		output_dir = RAW_DATA + '/test_obs/formatted/'
	else:
		print('illegal set. Exiting...')
		exit(1)

	# single('./resources/raw_data/APOL', './resources/data/APOL', args.set)
	# single('./resources/raw_data/LYFT', './resources/data/LYFT', args.set)


# 	##### --- generate several npy files --- #####

	files = [f for f in os.listdir(input_dir) if '.csv' in f]

	bags = np.array_split(files, THREAD)


	pool = multiprocessing.Pool(processes = THREAD)

	cmds = []

	for i in range(THREAD):
		cmds.append((input_dir, bags[i], output_dir, DATA_DIR, dtype, i))

	pool.starmap(multi, cmds)

	sz = 0
	for i in range(THREAD):
		sz += run_merge(output_dir, DATA_DIR, dtype, i)



	# ##### --- generate txt for sgan model --- #####

	# generate_sgan(DATA_DIR, dtype)

	# ##### --- generate npy for traphic model --- #####

	# sz = 0
	# ddir = os.path.join(DATA_DIR, dtype)
	# files = [f for f in os.listdir(ddir) if '.npy' in f]
	# # for k in range(len(files)):
	# # 	print("Counting #{} file in {}...".format(k, dtype))
	# # 	dt = np.load(os.path.join(ddir, files[k]), allow_pickle=True)
	# # 	sz += len(dt[0])

	# generate_data(DATA_DIR, dtype, (sz, 47))

