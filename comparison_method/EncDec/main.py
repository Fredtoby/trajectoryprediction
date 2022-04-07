from def_train_eval import *

from data_stream import *
import pickle              # import module first

DATA = 'NGSIM'
SUFIX = ''
DATA_DIR = '../../resources/data/{}/'.format(DATA)
epochs = 30



if __name__ == "__main__":
    f1 = open ( DATA_DIR +  'stream1_obs_data_{}.pkl'.format('train')  , 'rb')  # 'r' for reading; can be omitted
    g1 = open ( DATA_DIR + 'stream1_pred_data_{}.pkl'.format('train')   , 'rb')  # 'r' for reading; can be omitted
    tr_seq_1 = pickle.load ( f1 )  # load file content as mydict
    pred_seq_1 = pickle.load ( g1 )  # load file content as mydict
    

    f2 = open ( DATA_DIR +  'stream1_obs_data_{}.pkl'.format('test'), 'rb')  # 'r' for reading; can be omitted
    g2 = open ( DATA_DIR + 'stream1_pred_data_{}.pkl'.format('test'), 'rb')  # 'r' for reading; can be omitted
    test1 = pickle.load ( f2 )  # load file content as mydict
    test2 = pickle.load ( g2 )  # load file content as mydict


    trainIters(epochs, tr_seq_1 , pred_seq_1, test1, test2, DATA, SUFIX, print_every=1)
