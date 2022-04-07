import numpy as np
import pickle

def save_to_pkl(dir, file_sequence):
    with open(dir, 'wb') as f:
        pickle.dump(file_sequence, f)

def data_for_stream1(dir, train_seq_len = 3, pred_seq_len = 5, frame_lenth_cap = 10):
    '''
    This function returns train sequences and prediction sequences for LSTM and RNN
    :param dir: directory where the dataset is present
    :param train_seq_len: length of the train sequence
    :param pred_seq_len: length of the prediction sequence
    :param frame_lenth_cap: minimum length of time presence the agent must have
    :return: train sequences and prediction sequences
    '''

    data = np.loadtxt(dir)
    d_IDs = np.unique(data[:,4]).astype(int)

    train_sequence = []
    pred_sequence = []

    # traversing through all the dataset ID's
    for data_id in d_IDs:
        # print('data ID is',data_id)

        # dataset corresponding to that dataset_ID
        one_dataset = data[np.where(data[:,4]==data_id)]  #data_id
        obj_IDs = np.unique(one_dataset[:,1]).astype(int)

        # traversing through each agent from all agents corresponding to that dataset
        for each_obj_id in obj_IDs:

            # obtaining the trajectory of that individual agent
            one_obj_traj = one_dataset[np.where(one_dataset[:,1]==each_obj_id)] #each_obj_id

            # sleecting the object only if it is visible in at least 'frame_length_cap' number of frames
            if len(one_obj_traj[:,0]) >= frame_lenth_cap:

                # initializing an index range to obtain the train sequences and pred sequences
                index_range = len(one_obj_traj[:,0]) - train_seq_len - pred_seq_len

                # obtaining the sequences
                for idx in range(1, index_range+ 2):

                    train_sequence_dict = {}
                    pred_sequence_dict = {}

                    train_sequence_dict['dataset_ID'] = data_id
                    train_sequence_dict['agent_ID'] = each_obj_id
                    # train_sequence_dict['sequence'] = one_obj_traj[idx:idx+train_seq_len,2:4]
                    train_sequence_dict['sequence'] = one_obj_traj[idx-1:idx-1+train_seq_len,2:4]

                    pred_sequence_dict['dataset_ID'] = data_id
                    pred_sequence_dict['agent_ID'] = each_obj_id
                    # pred_sequence_dict['dataset_ID'] = one_obj_traj[idx+train_seq_len:idx+train_seq_len+pred_seq_len,2:4]
                    pred_sequence_dict['sequence'] = one_obj_traj[idx-1+train_seq_len:idx-1+train_seq_len+pred_seq_len,2:4]

                    train_sequence.append(train_sequence_dict)
                    pred_sequence.append(pred_sequence_dict)

    return train_sequence, pred_sequence

def data_for_stream2(dir, train_seq_len = 3, pred_seq_len = 5, frame_lenth_cap = 10):
    '''
    This function returns train sequences and prediction sequences for LSTM and RNN
    :param dir: directory where the dataset is present
    :param train_seq_len: length of the train sequence
    :param pred_seq_len: length of the prediction sequence
    :param frame_lenth_cap: minimum length of time presence the agent must have
    :return: train sequences and prediction sequences
    '''

    data = np.loadtxt(dir)

    train_sequence_stream2 = []
    pred_sequence_stream2 = []

    # total_objects = 220
    total_objects = int(np.amax(data[:,1]))

    d_IDs = np.unique(data[:,4]).astype(int)

    # traversing through all the dataset ID's
    for data_id in d_IDs:

        print('data ID is',data_id)
        # dataset corresponding to that dataset_ID
        one_dataset = data[np.where(data[:,4]==data_id)]  #data_id
        obj_IDs = np.unique(one_dataset[:,1]).astype(int)

        #     total_frames = 126
        total_frames = int(np.amax(data[:,0]))

        # traversing through each agent from all agents corresponding to that dataset
        for each_obj_id in obj_IDs:

            # obtaining the trajectory of that individual agent
            one_obj_traj = one_dataset[np.where(one_dataset[:,1]==each_obj_id)] #each_obj_id

            # sleecting the object only if it is visible in at least 'frame_length_cap' number of frames
            if len(one_obj_traj[:,0]) >= frame_lenth_cap:

                # initializing an index range to obtain the train sequences and pred sequences
                index_range = len(one_obj_traj[:,0]) - train_seq_len - pred_seq_len
                # print(index_range)

                # obtaining the sequences
                for idx in range(1,index_range+2):

                    # print(idx)
                    train_sequence_dict = {}

                    train_sequence_dict['dataset_ID'] = data_id
                    train_sequence_dict['agent_ID'] = each_obj_id

                    # obtaining all agents x and y values for the training sequences
                    for kdx in range(idx, idx+train_seq_len):

                        one_frame = one_dataset[np.where(one_dataset[:,0] == kdx)]
                        frame_objs = np.unique(one_frame[:,1]).astype(int)

                        frame_array = np.empty([2,total_objects])

                        for jdx in range(1,total_objects+1):

                            if jdx in frame_objs:
                                frame_array[0,jdx-1] = one_frame[np.where(one_frame[:,1]==jdx)][0][2]
                                frame_array[1,jdx-1] = one_frame[np.where(one_frame[:,1]==jdx)][0][3]

                            else:
                                frame_array[0,jdx-1] = 0
                                frame_array[1,jdx-1] = 0

                        train_sequence_dict[kdx] = frame_array

                    train_sequence_stream2.append(train_sequence_dict)

    return train_sequence_stream2

def load_batch(index, size, seq_ID, train_sequence_stream1, pred_sequence_stream_1, train_sequence_stream2,pred_sequence_stream_2):
    '''
    to load a batch of data
    :param index: index of the batch
    :param size: size of the batch of data
    :param seq_ID: either train sequence or a pred sequence, give as a str
    :param train_sequence: list of dicts of train sequences
    :param pred_sequence: list of dicts of pred sequences
    :return: Batch mof data
    '''

    i = index
    batch_size = size
    start_index = i * batch_size
    stop_index = (i+1) * batch_size

    if stop_index >= len(train_sequence_stream1):
        stop_index = len(train_sequence_stream1)

    if seq_ID == 'train':
        stream1_train_batch = train_sequence_stream1[start_index:stop_index]
        stream2_train_batch = train_sequence_stream2[start_index:stop_index]
        single_batch = [stream1_train_batch, stream2_train_batch]

    elif seq_ID == 'pred':
        stream1_pred_batch = pred_sequence_stream_1[start_index:stop_index]
        stream2_pred_batch = pred_sequence_stream_2[start_index:stop_index]
        single_batch = [stream1_pred_batch, stream2_pred_batch]

    else:
        single_batch = None
        print('please enter the sequence ID. enter train for train sequence or pred for pred sequence')

    return single_batch
