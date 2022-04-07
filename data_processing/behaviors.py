# import numpy as np
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import math
import pickle


def get_coordinates(scene_array_,fr,ag):
    '''
    To get x,y, cordinates from a dataset using frame_ID and agent_ID
    :param scene_array_: formatted data
    :param fr: frame_ID
    :param ag: agent_ID
    :return: x,y coordinates
    '''

    frame = scene_array_[np.where(scene_array_[:,0]==fr)]
    agent_data = frame[np.where(frame[:,1]==ag)]

    x = agent_data[0][2]
    y = agent_data[0][3]

    return x,y

def create_adjacent_matrix(scene_array_, scene_ID, total_num_of_agents):
    '''
    Creates adjacency matrix for each frame
    :param scene_array_: formatted data of one dataset
    :param scene_ID: equivalent to dataset_ID
    :return: returns a list of adjacency matrices for all the frame in that scene
    '''

    scene_array_n = np.copy(scene_array_)

    frame_ID_adj_mat_list = []

    lar = total_num_of_agents    # use 270 for LYFT

    for fr in range(1,int(np.amax(scene_array_n[:,0]) + 1)):
        frame_ID_adj_mat_dict = {}
        adj_matrix = np.zeros((lar, lar), dtype = np.float64)

        frame = scene_array_n[np.where(scene_array_n[:,0]==fr)]
        obj_set = set(np.unique(frame[:,1]).astype(int))

        for i in range(0 +1, adj_matrix.shape[0]+1):

            for j in range(1, adj_matrix.shape[1] +1):

                if i == j:
                    adj_matrix[i-1,j-1] = 0

                elif j in obj_set and i in obj_set:

                    x1, y1 = get_coordinates(scene_array_n,fr,i)
                    x2, y2 = get_coordinates(scene_array_n,fr,j)

                    a = np.array((x1 ,y1, 0))
                    b = np.array((x2, y2, 0))

                    eucl_dist = np.linalg.norm(a-b)

                    if eucl_dist < 10: ## Selected 10mts threshold according to US lane widths (3.7mts) rule.

                        val = math.exp(-eucl_dist)

                        adj_matrix[i-1,j-1] = val

                    else:
                        adj_matrix[i-1,j-1] = 0

                else:
                    adj_matrix[i-1,j-1] = 0

        frame_ID_adj_mat_dict['scene_ID'] = scene_ID
        frame_ID_adj_mat_dict['frame_ID'] = fr
        frame_ID_adj_mat_dict['adj_matrix'] = adj_matrix
        frame_ID_adj_mat_list.append(frame_ID_adj_mat_dict)

    return frame_ID_adj_mat_list

def generate_adjacency(dir, DATA_SET):
    '''
    Generates the adjacency matrices file in the specified folder
    :param dir: Directory where the formatted data is present (Format : frame_ID, agent_ID, X, Y, dataset_ID)
    :return: Saves the adjacency matrices as pickle files
    '''

    folder_path = dir
    val_filenames = []
    train_filenames = []
    frame_ID_adj_mat_list_train_scenes = []
    frame_ID_adj_mat_list_val_scenes = []
    total_num_of_agents = 170 # use 270 for LYFT,  use 170 for ARGO, use 905 for Apolloscape

    count = 1


    if DATA_SET == 'LYFT':

		file = 'trainSet0.txt'  # for val, file = 'valSet0.txt'
		data = np.loadtxt(folder_path + file)
		dataset_IDs = np.unique(data[:,4]).astype(int)
		total_num_of_agents= 270

		for d_id in dataset_IDs:
		    one_dataset = data[np.where(data[:,4]==d_id)]  #data_id
		    obj_IDs = np.unique(one_dataset[:,1]).astype(int)
		    d = dict([(y,x+1) for x,y in enumerate(sorted(set(obj_IDs)))])

		    for each_obj_ID in obj_IDs:
		        # print(each_obj_ID)
		        new_id = d[each_obj_ID]
		        one_obj_traj = one_dataset[np.where(one_dataset[:,1]==each_obj_ID)]
		        new_array = np.ones([one_obj_traj.shape[0],]) * new_id
		        one_obj_traj[:,1] = new_array
		        one_dataset[np.where(one_dataset[:,1]==each_obj_ID)]  = one_obj_traj

		    frame_ID_adj_mat_list = create_adjacent_matrix(one_dataset, d_id,total_num_of_agents)
		    frame_ID_adj_mat_list_train_scenes.append(frame_ID_adj_mat_list)



    elif DATA_SET == 'ARGO':

        file = 'trainSet0.txt'  # for val, file = 'valSet0.txt'
        data = np.loadtxt(folder_path + file)
        dataset_IDs = np.unique(data[:,4]).astype(int)
        d_IDs = dataset_IDs[:600]

        for d_id in d_IDs:
            one_dataset = data[np.where(data[:,4]==d_id)]  #data_id
            obj_IDs = np.unique(one_dataset[:,1]).astype(int)
            d = dict([(y,x+1) for x,y in enumerate(sorted(set(obj_IDs)))])

            for each_obj_ID in obj_IDs:
                # print(each_obj_ID)
                new_id = d[each_obj_ID]
                one_obj_traj = one_dataset[np.where(one_dataset[:,1]==each_obj_ID)]
                new_array = np.ones([one_obj_traj.shape[0],]) * new_id
                one_obj_traj[:,1] = new_array
                one_dataset[np.where(one_dataset[:,1]==each_obj_ID)]  = one_obj_traj

            frame_ID_adj_mat_list = create_adjacent_matrix(one_dataset, d_id,total_num_of_agents)
            frame_ID_adj_mat_list_train_scenes.append(frame_ID_adj_mat_list)

    elif DATA_SET == 'APOL':

        file = 'trainSet0.txt'  # for val, file = 'valSet0.txt'
        data = np.loadtxt(folder_path + file)
        # dataset_IDs = np.unique(data[:,4]).astype(int)
        dataset_IDs = [4] ##using only dataset 4, To use all datasets comment this line and uncomment above line

        for d_id in dataset_IDs:
            one_dataset = data[np.where(data[:,4]==d_id)]  #data_id
            obj_IDs = np.unique(one_dataset[:,1]).astype(int)
            d = dict([(y,x+1) for x,y in enumerate(sorted(set(obj_IDs)))])

            for each_obj_ID in obj_IDs:
                # print(each_obj_ID)
                new_id = d[each_obj_ID]
                one_obj_traj = one_dataset[np.where(one_dataset[:,1]==each_obj_ID)]
                new_array = np.ones([one_obj_traj.shape[0],]) * new_id
                one_obj_traj[:,1] = new_array
                one_dataset[np.where(one_dataset[:,1]==each_obj_ID)]  = one_obj_traj

            frame_ID_adj_mat_list = create_adjacent_matrix(one_dataset, d_id,total_num_of_agents)
            frame_ID_adj_mat_list_train_scenes.append(frame_ID_adj_mat_list)


    with open(folder_path + 'adjacency_mat_train_2.pkl', 'wb') as adg_t:
        pickle.dump(frame_ID_adj_mat_list_train_scenes, adg_t)

    with open(folder_path + 'adjacency_mat_val_2.pkl', 'wb') as adg_v:
        pickle.dump(frame_ID_adj_mat_list_val_scenes, adg_v)

def rates(list_train, DATA_SET):
    '''
    To obtain the theta_hat values for all the agents across all frames
    :param list_train: The pickled adjacency_list file
    :return: Returns a list of arrays. each colomn of this list is the list of theta_hat's of one agent
    '''

    all_rates_list = []

    if DATA_SET == 'ARGO' or 'LYFT':
        count = 1
        for list1 in list_train:
            print(count)

            diags_list = []

            for item in list1:
                adj = item['adj_matrix']

                d_vals = []
                for item in adj:
                    row_sum = sum(item)
                    d_vals.append(row_sum)
                diag_array = np.diag(d_vals)
                laplacian = diag_array - adj
                L_diag = np.diag(laplacian)
                diags_list.append(np.asarray(L_diag))


            all_rates_arr = np.empty([170,1])
            prev_ = diags_list[0]
            for items in range(1, len(diags_list)):
                next_ = diags_list[items]

                rate = next_ - prev_

                all_rates_arr = np.column_stack((all_rates_arr, rate))

                prev_ = next_

            all_rates_arr = np.delete(all_rates_arr , 0, 1)
            all_rates_list.append(all_rates_arr)
            count +=1

    elif DATA_SET == 'APOL':

        all_rates_list = []
        count = 1
        diags_list = []

        list1 = list_train

        for item in list1:
            print(count)
            adj = item['adj_matrix']

            d_vals = []
            for item in adj:
                row_sum = sum(item)
                d_vals.append(row_sum)
            diag_array = np.diag(d_vals)
            laplacian = diag_array - adj
            L_diag = np.diag(laplacian)
            diags_list.append(np.asarray(L_diag))
            count +=1

        all_rates_arr = np.empty([905,1])
        prev_ = diags_list[0]
        for items in range(1, len(diags_list)):
            next_ = diags_list[items]

            rate = next_ - prev_

            all_rates_arr = np.column_stack((all_rates_arr, rate))

            prev_ = next_

        all_rates_arr = np.delete(all_rates_arr , 0, 1)
        all_rates_list.append(all_rates_arr)

    return all_rates_list

def plot_behaviors(all_rates_list, num_of_agents):
    '''
    Plot the behaviors of all agents in the dataset
    :param all_rates_list: List of arrays with each agent's theta_hat's
    :param num_of_agents: number of agents in that dataset
    :return: None
    '''

    ## num_of_agents = 170 for ARGO, 270 for LYFT, 905 for APOL

    t = range(0,num_of_agents,1)
    all_mean = []

    all_rates_arr = all_rates_list[1]
    for item in range(0,num_of_agents):
        all_mean.append(np.mean(all_rates_arr[item]))

    overspeeding = []
    neutral = []
    braking = []
    for i,j in zip(all_mean, t):

        if i >= 0.01:
            overspeeding.append([i,j])

        elif i <= -0.01:
            braking.append([i,j])
        else:
            neutral.append([i,j])


    overspeeding = np.asarray(overspeeding)
    neutral = np.asarray(neutral)
    braking = np.asarray(braking)

    plt.scatter(overspeeding[:,1], overspeeding[:,0], marker='*', color='b',  label = r'overspeeding : $\theta^{\prime} >= 0.0025$')
    plt.scatter(neutral[:,1], neutral[:,0], marker='.', color='g', label = r'neutral : $-0.0025 < \theta^{\prime} < 0.0025 $')
    plt.scatter(braking[:,1], braking[:,0], marker='v', color='r', label = r'braking: $\theta^{\prime} <= -0.0025$')

    plt.ylabel(r'$\theta^{\prime}$')
    plt.xlabel('agents')
    plt.legend()
    # plt.ylim(top=0.01)
    # plt.ylim(bottom=-0.015)
    plt.rcParams.update({'font.size': 10})

def add_behaviors_stream2(adj_dir, obs_dir, pred_dir, DATA_SET):
    '''
    To add behavior labels to the strea2 observation sequences and prediction sequences
    :param adj_dir: location of the adjacency matrix list pickle file
    :param obs_dir: location of stream2 observation sequences list pickle file
    :param pred_dir: location of stream2 prediction sequences list pickle file
    :return: saves new pickle files for the stream2 observation, prediction sequences with behavior labels
    '''

    with open(adj_dir, 'rb') as adg_t:
        adj_mat = pickle.load(adg_t)

    with open(obs_dir, 'rb') as st2:
        loaded_train_st2 = pickle.load(st2)

    with open(pred_dir, 'rb') as st2_p:
        loaded_pred_st2 = pickle.load(st2_p)

    count = 0
    if DATA_SET == 'LYFT' or 'ARGO':

        for train,pred in zip(loaded_train_st2, loaded_pred_st2):
            dataset_ID = train['dataset_ID']
            agent_ID = train['agent_ID']
            all_frame_IDs = list(train.keys())[2:] + list(pred.keys())[2:]

            row_sum_list = []

            for adm in adj_mat[dataset_ID-1]:

                for fr_ID in all_frame_IDs:

                    if adm['frame_ID'] == fr_ID:
                        row_sum = sum(adm['adj_matrix'][agent_ID-1])
                        row_sum_list.append(row_sum)

            mean_theta = np.mean(row_sum_list)
            prev_ = row_sum_list[0]

            all_theta_hats = []
            for theta in range(1, len(row_sum_list)):

                next_ = row_sum_list[theta]
                theta_hat = next_ - prev_
                all_theta_hats.append(theta_hat)
                prev_ = next_

            theta_hat_mean = np.mean(all_theta_hats)

            train['mean_theta'] = mean_theta
            pred['mean_theta'] = mean_theta

            train['mean_theta_hat'] = theta_hat_mean
            pred['mean_theta_hat'] = theta_hat_mean
            print(count)
            count +=1

    elif DATA_SET == 'APOL':
        count = 0
        for train,pred in zip(loaded_train_st2, loaded_pred_st2):
            dataset_ID = train['dataset_ID']
            agent_ID = train['agent_ID']

            if agent_ID <= 905:
                all_frame_IDs = list(train.keys())[2:] + list(pred.keys())[2:]

                row_sum_list = []

                for adm in adj_mat:

                    for fr_ID in all_frame_IDs:

                        if adm['frame_ID'] == fr_ID:
                            row_sum = sum(adm['adj_matrix'][agent_ID-1])
                            row_sum_list.append(row_sum)

                mean_theta = np.mean(row_sum_list)
                prev_ = row_sum_list[0]

                all_theta_hats = []
                for theta in range(1, len(row_sum_list)):

                    next_ = row_sum_list[theta]
                    theta_hat = next_ - prev_
                    all_theta_hats.append(theta_hat)
                    prev_ = next_

                theta_hat_mean = np.mean(all_theta_hats)

                train['mean_theta'] = mean_theta
                pred['mean_theta'] = mean_theta

                train['mean_theta_hat'] = theta_hat_mean
                pred['mean_theta_hat'] = theta_hat_mean
                print(count)
                count +=1
            else:
                break

    with open( obs_dir + 'stream2_obs_data_train6310_behav.pkl', 'wb') as st2:
        pickle.dump(loaded_train_st2, st2)

    with open(pred_dir + 'stream2_pred_data_train6310_behav.pkl', 'wb') as st2_p:
        pickle.dump(loaded_pred_st2, st2_p)

## Directory where the formatted data is present
DATA_SET = 'ARGO'  # use 'LYFT' for lyft dataset, use 'APOL' for Apolloscape dataset and 'ARGO' for Argoverse dataset
DATA_DIR = 'directory/'


stream2_adjacency_mat_dir = 'directory/'
stream2_obs_dir = 'directory/'
stream2_pred_dir = 'directory/'
