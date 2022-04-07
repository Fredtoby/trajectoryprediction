import sys
import os
sys.path.append('..')
import time
import torch.utils.data as utils
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from models import *
from sklearn.cluster import SpectralClustering , KMeans
from data_stream import *

from scipy.sparse.linalg import eigs
from torch.autograd import Variable

device = torch.device("cuda:0")
# s1 = True
BATCH_SIZE=40
train_seq_len = 30
pred_seq_len = 50
FINAL_GRIP_OUTPUT_COORDINATE_SIZE = 256
FINAL_GRIP_OUTPUT_COORDINATE_SIZE_DECODER = 512
MODEL_LOC = '../../resources/trained_models/EncDec'


def trainIters(n_epochs, train_dataloader, valid_dataloader, test1, test2, data, sufix, print_every=1, plot_every=1000, learning_rate=1e-3):

    num_batches = int(len(train_dataloader)/BATCH_SIZE)
    # num_batches = 1


    train_raw = train_dataloader
    pred_raw = valid_dataloader

    # Initialize encoder, decoders for both streams

    batch = load_batch ( 0 , BATCH_SIZE , 'pred' , train_raw , pred_raw )
    batch_in_form = np.asarray ( [ batch[ i ][ 'sequence' ] for i in range ( BATCH_SIZE ) ] )
    batch_in_form = torch.Tensor ( batch_in_form )
    [ batch_size , step_size , fea_size ] = batch_in_form.size ()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size

    encoder_stream1 = Encoder ( input_dim , hidden_dim , output_dim ).to ( device )
    decoder_stream1 = Decoder ( 's1' , input_dim , hidden_dim , output_dim , batch_size , step_size ).to ( device )
    encoder_stream1_optimizer = optim.RMSprop ( encoder_stream1.parameters () , lr=learning_rate )
    decoder_stream1_optimizer = optim.RMSprop ( decoder_stream1.parameters () , lr=learning_rate )

    for epoch in range ( 0 , n_epochs ):
        #        print("epoch: ", epoch)
        print_loss_total_stream1 = 0  # Reset every print_every
        print_loss_total_stream2 = 0  # Reset every plot_every
        # Prepare train and test batch
        for bch in range ( num_batches ):
            # print ( '# {}/{} epoch {}/{} batch'.format ( epoch , n_epochs , bch , num_batches ) )
            trainbatch_both = load_batch ( bch , BATCH_SIZE , 'train' , train_raw , pred_raw )
            trainbatch = trainbatch_both
            trainbatch_in_form = np.asarray ( [ trainbatch[ i ][ 'sequence' ] for i in range ( BATCH_SIZE ) ] )
            trainbatch_in_form = torch.Tensor ( trainbatch_in_form ).to ( device )

            testbatch_both = load_batch ( bch , BATCH_SIZE , 'pred' , train_raw , pred_raw  )
            testbatch = testbatch_both
            testbatch_in_form = np.asarray ( [ testbatch[ i ][ 'sequence' ] for i in range ( BATCH_SIZE ) ] )
            testbatch_in_form = torch.Tensor ( testbatch_in_form ).to ( device )
            # for data in train_dataloader:

            input_stream1_tensor = trainbatch_in_form
            batch_agent_ids = [ trainbatch[ i ][ 'agent_ID' ] for i in range ( BATCH_SIZE ) ]
            target_stream1_tensor = testbatch_in_form


            loss_stream1 = train_stream ( input_stream1_tensor , target_stream1_tensor , encoder_stream1 , decoder_stream1 , encoder_stream1_optimizer , decoder_stream1_optimizer )
            print_loss_total_stream1 += loss_stream1

        print ( '{}/{} epochs: stream1 average loss:'.format(epoch, n_epochs) , print_loss_total_stream1 / num_batches )

    compute_accuracy_stream1(train_dataloader, 	valid_dataloader, encoder_stream1, decoder_stream1, n_epochs)
    save_model(encoder_stream1, decoder_stream1, data, sufix )
    compute_accuracy_stream1(test1, test2, encoder_stream1, decoder_stream1, n_epochs)

def save_model( encoder_stream, decoder_stream, data, sufix,  loc=MODEL_LOC):
    torch.save(encoder_stream.state_dict(), os.path.join(loc, 'encoder_stream_EncDec_{}{}.pt'.format(data, sufix)))
    torch.save(decoder_stream.state_dict(), os.path.join(loc, 'decoder_stream_EncDec_{}{}.pt'.format(data, sufix)))
    print('model saved at {}'.format(loc))

def eval(epochs, tr_seq_1, pred_seq_1, data, sufix, learning_rate=1e-3, loc=MODEL_LOC):
    
    
    encoder_stream1 = None
    decoder_stream1 = None

    encoder1loc = os.path.join(loc, 'encoder_stream_EncDec_{}{}.pt'.format(data, sufix))
    decoder1loc = os.path.join(loc, 'decoder_stream_EncDec_{}{}.pt'.format(data, sufix))

    train_raw = tr_seq_1
    pred_raw = pred_seq_1
    # Initialize encoder, decoders for both streams
    batch = load_batch ( 0 , BATCH_SIZE , 'pred' , train_raw , pred_raw , train2_raw , pred2_raw )
    batch , _ = batch
    batch_in_form = np.asarray ( [ batch[ i ][ 'sequence' ] for i in range ( BATCH_SIZE ) ] )
    batch_in_form = torch.Tensor ( batch_in_form )
    [ batch_size , step_size , fea_size ] = batch_in_form.size ()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size

    encoder_stream1 = Encoder ( input_dim , hidden_dim , output_dim ).to ( device )
    decoder_stream1 = Decoder ( 's1' , input_dim , hidden_dim , output_dim, batch_size, step_size ).to ( device )
    encoder_stream1_optimizer = optim.RMSprop(encoder_stream1.parameters(), lr=learning_rate)
    decoder_stream1_optimizer = optim.RMSprop(decoder_stream1.parameters(), lr=learning_rate)
    encoder_stream1.load_state_dict(torch.load(encoder1loc))
    encoder_stream1.eval()  
    decoder_stream1.load_state_dict(torch.load(decoder1loc))
    decoder_stream1.eval()



    compute_accuracy_stream1(tr_seq_1, pred_seq_1, encoder_stream1, decoder_stream1, epochs)






       
def train_stream(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer):

    Hidden_State , _ = encoder.loop(input_tensor)
    stream2_out,_ = decoder.loop(Hidden_State)

    l = nn.MSELoss()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = l(stream2_out, target_tensor)
    loss.backward()

    # loss = -log_likelihood(mu_1, mu_2, log_sigma_1, log_sigma_2, rho, target_tensor)
    # print(loss)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def compute_accuracy_stream1 ( traindataloader , labeldataloader , encoder , decoder , n_epochs ):
    ade = 0
    fde = 0
    count = 0

    train_raw = traindataloader
    pred_raw = labeldataloader
    train2_raw = [ ]
    pred2_raw = [ ]

    batch = load_batch ( 0 , BATCH_SIZE , 'pred' , train_raw , pred_raw  )
    batch_in_form = np.asarray ( [ batch[ i ][ 'sequence' ] for i in range ( BATCH_SIZE ) ] )
    batch_in_form = torch.Tensor ( batch_in_form )
    [ batch_size , step_size , fea_size ] = batch_in_form.size ()

    for epoch in range ( 0 , n_epochs ):
        # Prepare train and test batch

        trainbatch_both = load_batch ( epoch , BATCH_SIZE , 'train' , train_raw , pred_raw )
        trainbatch  = trainbatch_both
        trainbatch_in_form = np.asarray ( [ trainbatch[ i ][ 'sequence' ] for i in range ( BATCH_SIZE ) ] )
        trainbatch_in_form = torch.Tensor ( trainbatch_in_form )

        testbatch_both = load_batch ( epoch , BATCH_SIZE , 'pred' , train_raw , pred_raw  )
        testbatch  = testbatch_both
        testbatch_in_form = np.asarray ( [ testbatch[ i ][ 'sequence' ] for i in range ( BATCH_SIZE ) ] )
        testbatch_in_form = torch.Tensor ( testbatch_in_form )
        # for data in train_dataloader:

        train = trainbatch_in_form.to ( device )
        label = testbatch_in_form.to ( device )

        Hidden_State , _ = encoder.loop ( train )
        stream2_out , _  = decoder.loop ( Hidden_State )
        scaled_train = scale_train ( stream2_out , label )
        mse = MSE ( scaled_train/torch.max(scaled_train) , label/torch.max(label) ) * (torch.max(label)).cpu().detach().numpy()
        # mse = MSE ( scaled_train , label )
        mse = np.sqrt ( mse )
        # print(mse)
        ade += mse
        fde += mse[ -1 ]
        # count += testbatch_in_form.size ()[ 0 ]
        count += 1
    ade = ade / count
    fde = fde / count
    print ( "ADE: {} FDE: {}".format ( ade , fde ) )
    print ( "average: ADE: {} FDE: {}".format ( np.mean(ade), np.mean(fde) ) )




def scale_train(train_tensor, target_tensor):
    train_tensor_x = train_tensor[:,0,:].clone()
    train_tensor_y= train_tensor[:,1,:].clone()
    target_tensor_x= target_tensor[:,0,:].clone()
    target_tensor_y= target_tensor[:,1,:].clone()

    train_tensor[:,0,:] = torch.mean(target_tensor_x) + (train_tensor_x - torch.mean(train_tensor_x))*( (torch.std(target_tensor_x)/ torch.std(train_tensor_x)))
    train_tensor[:,1,:] = torch.mean(target_tensor_y) + (train_tensor_y- torch.mean(train_tensor_y))*( (torch.std(target_tensor_y)/ torch.std(train_tensor_y)))

    return train_tensor


def MSE ( y_pred , y_gt , device=device ):
    # y_pred = y_pred.numpy()
    y_pred = y_pred.cpu ().detach ().numpy ()
    y_gt = y_gt.cpu ().detach ().numpy ()
    acc = np.zeros ( np.shape ( y_pred )[ :-1 ] )
    muX = y_pred[ : , : , 0 ]
    muY = y_pred[ : , : , 1 ]
    x = np.array ( y_gt[ : , : , 0 ] )
    y = np.array ( y_gt[ : , : , 1 ] )
    #print ( muX , x , muY , y )
    acc = np.power ( x - muX , 2 ) + np.power ( y - muY , 2 )
    lossVal = np.sum ( acc , axis=0 ) / len ( acc )
    return lossVal
