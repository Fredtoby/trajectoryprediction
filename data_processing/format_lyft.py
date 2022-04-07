import json
import numpy as np

def get_sample(filepath):
    '''
    to load the sample.json file
    :param filepath: filepath of the sample.json
    :return: sample
    '''
    with open(filepath) as json_file:
        sample = json.load(json_file)

    return sample


def get_sample_data(filepath):
    '''
    to load the sample_data.json file
    :param filepath: filepath of the sample_data.json
    :return: sample_data
    '''
    with open(filepath) as json_file:
        sample_data = json.load(json_file)

    return sample_data


def get_sample_annotation(filepath):
    '''
    to load the sample_annotation.json file
    :param filepath: filepath of the sample_annotation.json
    :return: sample_annotation file
    '''
    with open(filepath) as json_file:
        sample_annotation = json.load(json_file)
    return sample_annotation


def get_timestamp_list(sample, scene_token):
    '''
    to get the list with sample tokens corresponding to timestamp
    :param sample: sample from sample.json
    :param scene_token: scene_token from scene.json
    :return: sorted list of timestamp, tokens
    '''

    time_stamp_list = []
    visited_set = set()
    count = 0
    for items_sample in sample:
        if items_sample['scene_token'] == scene_token:
            time_stamp_list.append([items_sample['timestamp'],items_sample['token']])
            visited_set.add(items_sample['token'])
            count +=1

    sorted_timestamp_list = sorted(time_stamp_list)
    return sorted_timestamp_list

def zero_row_process(zero_row):
    '''
    to format the timestamp row
    :param zero_row: row to be formatter, timestamp_row
    :return: formatted row as an array
    '''

    first_ele = zero_row[0]
    frame_ID_list = []
    j = 1

    for i in zero_row:
        if i == first_ele:
            frame_ID_list.append(j)
        else:
            first_ele = i
            j+=1
            frame_ID_list.append(j)

    frame_ID_array = np.asarray(frame_ID_list, dtype= int).T

    return frame_ID_array


def create_ultimate_list(sorted_timestamp_list, sample_data, sample_annotation):
    '''
    to create a list consisting of all the data that we require to obtain the final file
    :param sorted_timestamp_list: sorted timestamp, token list
    :param sample_data: sample_data from sample_data.json
    :param sample_annotation: sample_annotation from from sample_annotation.json
    :return: ultimate list with all the required data sorted based on timestamp
    '''

    ultimate_list = []
    new_dict = {}
    for st_l in sorted_timestamp_list:
        new_dict['timestamp'] = st_l[0]
        new_dict['sample_token'] = st_l[1]
        new_dict['sample_data'] = {}
        ultimate_list.append(new_dict)
        new_dict = {}

    for st_list in ultimate_list:
        samp_data = []
        for items_sample_data in sample_data:
            if items_sample_data['sample_token'] == st_list['sample_token']:
                samp_data.append(items_sample_data)

        st_list['sample_data'] = samp_data

    for stl_list in ultimate_list:
        ann_data = []
        for items_sample_ann_data in sample_annotation:
            if items_sample_ann_data['sample_token'] == stl_list['sample_token']:
                ann_data.append(items_sample_ann_data)

        stl_list['annotation_data'] = ann_data

    return ultimate_list


def timestamp_annotations(ultimate_list):
    '''
    to obtain a list with [timestamp, instances, location]
    :param ultimate_list: the ultimate list obtained
    :return: [timestamp, instances, location] list and instance_tokens_list
    '''
    ts_ann_list = []
    instance_tokens_list = []

    for items5 in ultimate_list:
        ts = items5['timestamp']

        for items6 in items5['annotation_data']:
            ts_ann_list.append([ts, items6['instance_token'], items6['translation']])
            instance_tokens_list.append(items6['instance_token'])
    return ts_ann_list, instance_tokens_list

def instance_matching(instance_tokens_list):
    '''
    assigning each instance_token with a numerical value and obtained a dictionary
    :param instance_tokens_list: list of all instance_tokens
    :return: obtained dictionary
    '''

    k = 1
    instance_matching_dict = {}
    visited_tokens_set = set()
    for item7 in instance_tokens_list:

        if item7 not in visited_tokens_set:
            instance_matching_dict[item7] = k
            visited_tokens_set.add(item7)
            k += 1
        else:
            continue

    return instance_matching_dict


def timestamp_objectID_XYZ(ultimate_list, instance_matching_dict):
    '''
    final list with all the information [timestamp, object_ID, X, Y, Z]
    :param ultimate_list: ultimate list that was created
    :param instance_matching_dict: instance_matching_dict
    :return: list in the format [timestamp, object_ID, X, Y, Z]
    '''
    ts_obj_ID = []

    for items5 in ultimate_list:
        ts = items5['timestamp']

        for items6 in items5['annotation_data']:
            ts_obj_ID.append([ts, instance_matching_dict[items6['instance_token']], items6['translation'][0], items6['translation'][1], items6['translation'][2]])


    return ts_obj_ID


def format_frame_ID(ts_obj_ID):
    '''
    to transform timestamps into 1 to max range
    :param ts_obj_ID:  list in the format [timestamp, object_ID, X, Y, Z]
    :return: array of frame_ID's
    '''
    frame_ID_list = []
    for items7 in ts_obj_ID:
        ts = items7[0]
        frame_ID_list.append(ts)

    frame_ID_arr = np.asarray(frame_ID_list).T

    return frame_ID_arr


def lyft_to_formatted(dir):
    '''
    to format the lyft data
    :param dir: directory for lyft data
    :return: files in the LYFT folder of the supplied directory
    '''

    ## loading all the data
    sample_filepath = dir + 'sample.json'
    sample = get_sample(sample_filepath)

    sample_data_filepath = dir + 'sample_data.json'
    sample_data = get_sample_data(sample_data_filepath)

    sample_annotation_filepath = dir + 'sample_annotation.json'
    sample_annotation = get_sample_annotation(sample_annotation_filepath)

    scene_file_path = dir + 'scene.json'
    with open(scene_file_path) as json_file:
        scene = json.load(json_file)

    print('loaded all files')

    #iterating through all the scene data
    index = 1
    
    empty_array_train = np.empty((0,5))
    empty_array_test = np.empty((0,5))
    empty_array_val = np.empty((0,5))


    for items2 in scene:


        # using scene token to get the sample tokens
        scene_token = items2['token']
        sorted_timestamp_list = get_timestamp_list(sample, scene_token)
        print('got sorted_timestamp_list for scene ', index)
        ultimate_list = create_ultimate_list(sorted_timestamp_list, sample_data, sample_annotation)
        print('got ultimate_list for scene ', index)

        ts_ann_list, instance_tokens_list = timestamp_annotations(ultimate_list)
        instance_matching_dict = instance_matching(instance_tokens_list)
        print('got instance_matching_dict for scene ', index)

        ts_obj_ID = timestamp_objectID_XYZ(ultimate_list, instance_matching_dict)
        frame_ID_arr = format_frame_ID(ts_obj_ID)
        formatted_zero_row = zero_row_process(frame_ID_arr)
        ts_obj_ID_arr = np.asarray(ts_obj_ID)
        ts_obj_ID_arr[:,0] = formatted_zero_row
        print('got ts_obj_ID_arr for scene ', index)
        
        ts_obj_ID_arr[:,4] = np.ones((ts_obj_ID_arr.shape[0],)) * index
        
        if index <=126 and index >=0:

            empty_array_train =  np.concatenate((empty_array_train, ts_obj_ID_arr))  

        elif index <= 144 and index >126:
            empty_array_test =  np.concatenate((empty_array_test, ts_obj_ID_arr))  

        elif index <=180 and index> 144:
#             to_save_txt = dir + 'LYFT/test_obs/traj{:>04}.txt'.format(index)
            empty_array_val =  np.concatenate((empty_array_val, ts_obj_ID_arr))  

        index += 1

    return empty_array_train, empty_array_test, empty_array_val


DATA_DIR = 'directory/' ## provide the directory where the downloaded data is present

# train_dir = './resources/data/' + 'LYFT/train/*.txt'
files_path_to_sv_train = './resources/data/' + 'LYFT/train/trainSet0.npy'


# test_dir = './resources/data/' + 'LYFT/test/*.txt'
files_path_to_sv_test = './resources/data/' + 'LYFT/test/testSet0.npy'

# val_dir = './resources/data/' + 'LYFT/val/*.txt'
files_path_to_sv_val = './resources/data/' + 'LYFT/val/valSet0.npy'

print(files_path_to_sv_train)
print(files_path_to_sv_test)
print(files_path_to_sv_val)

tr, te, va  = lyft_to_formatted(DATA_DIR)

np.save(files_path_to_sv_train, tr)
np.save(files_path_to_sv_test, te)
np.save(files_path_to_sv_val, va)
