import numpy as np
import argparse
import os
import re
import json
import pickle
import matplotlib.pyplot as plt



DIR = './resources/data'
META_DIR = './resources/meta'

def main(args):
	print('extracting {} {} in {} groups'.format(args.dset, args.set, args.num_files))
	data_folder = os.path.join(DIR, args.dset, args.set)
	files = [x for x in os.listdir(data_folder) if '.txt' in x]
	files.sort()
	files_bag = np.array_split(files, args.num_files)
	for i, fs in enumerate(files_bag):
		print("group {} :".format(i), [int(re.search(r'\d+', s).group()) for s in fs ] )
		meta = {}
		for f in fs:
			meta.update(extract_info(os.path.join(data_folder, f)))
		save_pickle(os.path.join(META_DIR, args.dset, '{}{}.pickle'.format(args.set, i)), meta)
#		writer = open(os.path.join(META_DIR, args.dset, '{}{}.json'.format(args.set, i)), 'w')
#		writer.write(json.dumps(meta))
#		writer.close()


def save_pickle(name, dic):
	with open(name, 'wb') as f:
		pickle.dump(dic, f)

def load_pickle(name):
	with open(name, 'rb') as f:
		return pickle.load(f)

def format_meta(dic, outdir, typ):
	dset_num = len(dic.keys())
	video_length = [x[0] for x in dic.values()]
	vehicle_num = [x[1] for x in dic.values()]
	traj_stats = [x[2] for x in dic.values()]
	traj = []
	for row in dic.values():
		traj.extend(row[3])

	Xmin = min([x[5][0] for x in dic.values()])
	Xmax = max([x[5][1] for x in dic.values()])
	Xmean = np.mean([x[5][2] for x in dic.values()])
	Xmedian = np.median([x[5][4] for x in dic.values()])
	Ymin = min([y[6][0] for y in dic.values()])
	Ymax = max([y[6][1] for y in dic.values()])
	Ymean = np.mean([x[6][2] for x in dic.values()])
	Ymedian = np.median([x[6][4] for x in dic.values()])

	with open(outdir + typ + '.txt', 'w') as f:
		f.write('Total number of dataset: {}\n'.format(dset_num))
		f.write('Video length: max--{} min--{} average--{} std--{} median--{}\n'.format(np.max(video_length), np.min(video_length), np.mean(video_length), np.std(video_length), np.median(video_length)))
		f.write('Vehicle number: max--{} min--{} average--{} std--{} median--{}\n'.format(np.max(vehicle_num), np.min(vehicle_num), np.mean(vehicle_num), np.std(vehicle_num), np.median(vehicle_num)))
		f.write('Trajectories length: max--{} min--{} average--{} std--{} median--{}\n'.format(np.max(traj), np.min(traj), np.mean(traj), np.std(traj), np.median(traj)))
		f.write('X coordinates: max--{} min--{} average--{} median--{}\n'.format(Xmax, Xmin, Xmean, Xmedian))
		f.write('Y coordinates: max--{} min--{} average--{} median--{}\n'.format(Ymax, Ymin, Ymean, Ymedian))


	X = []
	Y = []
	for x in np.unique(traj):
		X.append(x)
		Y.append(len([i for i in traj if i == x]))
	# plot( , , 'time', 'number of vehicles', 'density-{}'.format(typ), outdir)
	plot(X, Y, 'trajectories length', 'number of vehicles', 'trajectories_length-{}'.format(typ), outdir)

	densities = [x[4] for x in dic.values()]
	X = []
	Y = []
	xlen = 0
	ylen = 0
	for density in densities:
		xlen = max(xlen, len(density))


	X = [i + 1 for i in range(xlen)]
	Y = np.zeros(xlen)
	for density in densities:
		for item in density:
			Y[int(item[0]-1)] += item[1]
	Y = Y/len(densities)
	plot(X, Y, 'frame', 'number of vehicles', 'vehicle_density-{}'.format(typ), outdir)

# file: [row] frame_id vehicle_id tl-X tl-Y dset_id
# data: [row] frame_id vehicle_id dset_id
# stats: [dset_id]: tuple (length of video, number of vehicles, trajectories:(min, max, mean, std, median), list of trajectories,
#                          density over time, X:(min, max, mean, std, median), Y:(min, max, mean, std, median))
# trajectory: [length]
# density: [frame id, number of vehicles]
def extract_info(filename, delim='\t', detail=True):
	print('Processing {} ...'.format(int(re.search(r'\d+', filename).group())))
	data = [] 
	stats = {}

	with open(filename, 'r') as f:
		for line in f:
			line = line.strip(delim).split()
			line[0] = int(line[0])
			line[1] = int(line[1])
			temp = int(line[4])
			line[4] = float(line[3])
			line[3] = float(line[2])
			line[2] = temp
			# data.append(line[:3])
			data.append(line)
#	print('Finish loading and extracting...'.format(filename))
	data = np.array(data)
	dset = np.unique(data[:,2])

	for d in dset:
		sub = data[data[:, 2] == d]
		vids = np.unique(sub[:, 1])
		fids = np.unique(sub[:, 0])
		video_len = len(fids)
		vehicle_num = len(vids)
		X = (min(sub[:,3]), max(sub[:,3]), np.mean(sub[:,3]), np.std(sub[:,3]), np.median(sub[:,3]))
		Y = (min(sub[:,4]), max(sub[:,4]), np.mean(sub[:,4]), np.std(sub[:,4]), np.median(sub[:,4]))
		traj = []
		for i, v in enumerate(vids):
			traj.append(len(sub[sub[:,1] == v]))
		traj = np.array(traj)
		trajmeta = (min(traj), max(traj), np.mean(traj), np.std(traj), np.median(traj))

		density = []
		fids.sort()
		for f in fids:
			density.append([f, len(sub[sub[:,0] == f])])
		density = np.array(density)
		if detail:
			stats[int(d)] = (video_len, vehicle_num, trajmeta, traj, density, X, Y) 
		else:
			stats[int(d)] = (video_len, vehicle_num, trajmeta, density, X, Y) 
	
	del data
	return stats

def plot(x, y, x_label, y_label, title, save_loc, typ='line'):
	fig = plt.figure(title)
	if typ == 'line': 
		plt.plot(x, y)
	else:
		plt.hist(x, bins=50)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	fig.savefig(os.path.join(save_loc, title+'.png'))
	fig.clear()
	with open(os.path.join(save_loc, title+'.txt'), 'w') as f:
		if typ == 'line':
			for i in range(len(x)):
				f.write('{}, {}'.format(x[i], y[i]))
		else:
			for i in range(len(x)):
				f.write('{}'.format(x[i]))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--dset', '-d', required=True, type=str)
	# parser.add_argument('--set', '-s', required=True, type=str)
	parser.add_argument('--num_files', '-n', type=int, default=1)
	args = parser.parse_args()
	args.set = 'train'
	main(args)
	args.set = 'val'
	main(args)
	args.set = 'test'
	main(args)


	dic = load_pickle('./resources/meta/{}/train0.pickle'.format(args.dset))
	format_meta(dic, './resources/meta/{}/'.format(args.dset), 'train')
	dic = load_pickle('./resources/meta/{}/val0.pickle'.format(args.dset))
	format_meta(dic, './resources/meta/{}/'.format(args.dset), 'val')	
	dic = load_pickle('./resources/meta/{}/test0.pickle'.format(args.dset))
	format_meta(dic, './resources/meta/{}/'.format(args.dset), 'test')



