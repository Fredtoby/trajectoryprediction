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
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from torch.autograd import Variable

device = torch.device("cuda")
BATCH_SIZE=128
MU = 5
MODEL_LOC = 'resources/trained_models/Ours/{}'


def load_batch(index, size, seq_ID, train_sequence_stream1, pred_sequence_stream_1, train_sequence_stream2, pred_sequence_stream2, train_eig_seq, pred_eig_seq):
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
        start_index = stop_index - batch_size
    if seq_ID == 'train':
        stream1_train_batch = train_sequence_stream1[start_index:stop_index]
        stream2_train_batch = train_sequence_stream2[start_index:stop_index]
        eigs = train_eig_seq[start_index:stop_index]
        single_batch = [stream1_train_batch, stream2_train_batch, eigs]

    elif seq_ID == 'pred':
        stream1_pred_batch = pred_sequence_stream_1[start_index:stop_index]
        stream2_pred_batch = pred_sequence_stream2[start_index:stop_index]
        eigs = pred_eig_seq[start_index:stop_index]
        single_batch = [stream1_pred_batch, stream2_pred_batch, eigs]
    else:
        single_batch = None
        print('please enter the sequence ID. enter train for train sequence or pred for pred sequence')
    return single_batch


def trainIters(n_epochs, train_dataloader, valid_dataloader, train2_dataloader,valid2_dataloader, train_eig, val_eig, data, sufix, s2, print_every=1, plot_every=1000, learning_rate=1e-3, save_every=5):
    start = time.time()
    plot_losses_stream1 = []
    plot_losses_stream2 = []

    output_stream2_decoder = None
    num_batches = int(len(train_dataloader)/BATCH_SIZE)
    # Stream1 Data
    # inputs , labels = next ( iter ( train_dataloader ) )
    # [ batch_size , step_size , fea_size ] = inputs.size ()
    # input_dim = fea_size
    # hidden_dim = fea_size
    # output_dim = fea_size
    encoder_stream1 = None 
    decoder_stream1 = None
    encoder_stream2 = None
    decoder_stream2 = None
    encoder1loc = os.path.join(MODEL_LOC.format(data), 'encoder_stream1_{}{}.pt'.format(data, sufix))
    decoder1loc = os.path.join(MODEL_LOC.format(data), 'decoder_stream1_{}{}.pt'.format(data, sufix)) 
    
    train_raw = train_dataloader
    pred_raw = valid_dataloader
    train2_raw = train2_dataloader
    pred2_raw = valid2_dataloader
    train_eig_raw = train_eig
    pred_eig_raw = val_eig
    # Initialize encoder, decoders for both streams
    batch = load_batch ( 0 , BATCH_SIZE , 'pred' , train_raw , pred_raw , train2_raw , pred2_raw, train_eig_raw, pred_eig_raw)
    batch , _, _= batch
    batch_in_form = np.asarray ( [ batch[ i ][ 'sequence' ] for i in range ( BATCH_SIZE ) ] )
    batch_in_form = torch.Tensor ( batch_in_form )
    [ batch_size , step_size , fea_size ] = np.shape(batch_in_form)
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size

    encoder_stream1 = Encoder ( input_dim , hidden_dim , output_dim ).to ( device )
    decoder_stream1 = Decoder ( 's1' , input_dim , hidden_dim , output_dim, batch_size, step_size ).to ( device )
    encoder_stream1_optimizer = optim.RMSprop(encoder_stream1.parameters(), lr=learning_rate)
    decoder_stream1_optimizer = optim.RMSprop(decoder_stream1.parameters(), lr=learning_rate)
    print("loading {}...".format(encoder1loc))
    #encoder_stream1.load_state_dict(torch.load(encoder1loc))
    #encoder_stream1.eval()       
    #decoder_stream1.load_state_dict(torch.load(decoder1loc))        
    #decoder_stream1.eval()
    if s2 is True:
        batch = load_batch ( 0 , BATCH_SIZE , 'pred' , train_raw , pred_raw , train2_raw , pred2_raw, train_eig_raw, pred_eig_raw )
        _ , _, batch = batch
        batch = np.asarray([batch[i] for i in range(len(batch))])
        batch_in_form = torch.Tensor(batch)
        [ batch_size , step_size , fea_size ] = batch_in_form.size()
        input_dim = fea_size
        hidden_dim = fea_size
        output_dim = fea_size

        encoder_stream2 = Encoder ( input_dim , hidden_dim , output_dim ).to ( device )
        decoder_stream2 = Decoder ( 's2', input_dim , hidden_dim , output_dim, batch_size, step_size ).to ( device )
        encoder_stream2_optimizer = optim.RMSprop(encoder_stream2.parameters(), lr=learning_rate)
        decoder_stream2_optimizer = optim.RMSprop(decoder_stream2.parameters(), lr=learning_rate)


    for epoch in range(0, n_epochs):
#        print("epoch: ", epoch)
        print_loss_total_stream1 = 0  # Reset every print_every
        print_loss_total_stream2 = 0  # Reset every plot_every
        # Prepare train and test batch
        for bch in range(num_batches):
            print('# {}/{} epoch {}/{} batch'.format(epoch, n_epochs, bch, num_batches))
            trainbatch_both = load_batch ( bch , BATCH_SIZE , 'train' , train_raw , pred_raw, train2_raw, pred2_raw, train_eig_raw, pred_eig_raw )
            trainbatch, train_middle, trainbatch2 = trainbatch_both
            trainbatch_in_form = np.asarray([trainbatch[i]['sequence'] for i in range(BATCH_SIZE)])
            trainbatch_in_form = torch.Tensor( trainbatch_in_form ).to(device)

            testbatch_both = load_batch ( bch , BATCH_SIZE , 'pred' , train_raw , pred_raw, train2_raw, pred2_raw, train_eig_raw, pred_eig_raw )
            testbatch, test_middle, testbatch2 = testbatch_both
            testbatch_in_form = np.asarray([testbatch[i]['sequence'] for i in range(BATCH_SIZE)])
            testbatch_in_form =  torch.Tensor(testbatch_in_form ).to(device)
            # for data in train_dataloader:

            input_stream1_tensor = trainbatch_in_form
            batch_agent_ids = [trainbatch[i]['agent_ID'] for i in range(BATCH_SIZE)]
            target_stream1_tensor = testbatch_in_form
            if s2 is True:
                trainbatch2 = torch.Tensor(np.asarray([trainbatch2[i] for i in range(len(trainbatch2))]))
                input_stream2_tensor = trainbatch2
                input_stream2_tensor = Variable(input_stream2_tensor.to(device))
                testbatch2 =  torch.Tensor(np.asarray([testbatch2[i] for i in range(len(testbatch2))]))
                target_stream2_tensor = testbatch2
                target_stream2_tensor = Variable(target_stream2_tensor.to(device))
            if s2 is True:
                loss_stream2, output_stream2_decoder = train_stream2(input_stream2_tensor, target_stream2_tensor, encoder_stream2, decoder_stream2, encoder_stream2_optimizer, decoder_stream2_optimizer)
                print_loss_total_stream2 += loss_stream2

            loss_stream1 = train_stream1(input_stream1_tensor, target_stream1_tensor, encoder_stream1, decoder_stream1, encoder_stream1_optimizer, decoder_stream1_optimizer, output_stream2_decoder, batch_agent_ids, test_middle, s2)
            print(loss_stream1)
            print_loss_total_stream1 += loss_stream1
                # print(loss_stream1)

            # print_loss_avg_stream1 = print_loss_total_stream1 / print_every
            # print_loss_total_stream1 = 0
        print( 'stream1 average loss:', print_loss_total_stream1/num_batches)
            # print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),epoch, epoch / n_epochs * 100, print_loss_avg_stream1))
        if s2 is True:
            # print_loss_avg_stream2 = print_loss_total_stream2 / print_every
            # print_loss_total_stream2 = 0
            print( 'stream2 average loss:', print_loss_total_stream2/num_batches)

            # print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),epoch, epoch / n_epochs * 100, print_loss_avg_stream2))

            # if epoch % plot_every == 0:
            #     if s1 is True:
            #         plot_loss_avg_stream1 = plot_loss_total_stream1 / plot_every
            #         plot_losses_stream1.append(plot_loss_avg_stream1)
            #         plot_loss_total_stream1 = 0
            #     if s2 is True:
            #         plot_loss_avg_stream2 = plot_loss_total_stream2 / plot_every
            #         plot_losses_stream2.append(plot_loss_avg_stream2)
            #         plot_loss_total_stream2 = 0
        if epoch % save_every == 0:
            save_model(encoder_stream1, decoder_stream1, encoder_stream2, decoder_stream2, data, sufix, s2)

    compute_accuracy_stream1(train_dataloader, valid_dataloader, encoder_stream1, decoder_stream1, n_epochs)
    # showPlot(plot_losses)
    save_model(encoder_stream1, decoder_stream1, encoder_stream2, decoder_stream2 , data, sufix, s2)
    return encoder_stream1, decoder_stream1

def eval(epochs, tr_seq_1, pred_seq_1, data, sufix, learning_rate=1e-3, loc=MODEL_LOC):
    
    encoder_stream1 = None
    decoder_stream1 = None
    encoder_stream2 = None
    decoder_stream2 = None

    encoder1loc = os.path.join(loc.format(data), 'encoder_stream1_{}{}.pt'.format(data, sufix))
    decoder1loc = os.path.join(loc.format(data), 'decoder_stream1_{}{}.pt'.format(data, sufix))
    encoder2loc = os.path.join(loc.format(data), 'encoder_stream2_{}{}.pt'.format(data, sufix))
    decoder2loc = os.path.join(loc.format(data), 'decoder_stream2_{}{}.pt'.format(data, sufix))

    train_raw = tr_seq_1
    pred_raw = pred_seq_1
#    train2_raw = tr_seq_2
#    pred2_raw = pred_seq_2
    # Initialize encoder, decoders for both streams
    batch = load_batch ( 0 , BATCH_SIZE , 'pred' , train_raw , pred_raw , [], [], [], [] )
    batch , _, _ = batch
    batch_in_form = np.asarray ( [ batch[ i ][ 'sequence' ] for i in range ( BATCH_SIZE ) ] )
    batch_in_form = torch.Tensor ( batch_in_form )
    [ batch_size , step_size , fea_size ] = np.shape(batch_in_form)
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



def train_stream1(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, stream2_output,batch_agent_ids, testbatch2, s2):

    target_length = target_tensor.size(0)

    Hidden_State , _ = encoder.loop(input_tensor)
    _, _, mu_1, mu_2, log_sigma_1, log_sigma_2, rho = decoder.loop(Hidden_State)


    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    if s2 == True:
        cluster_centers = calculate_cluster_centers(batch_agent_ids, stream2_output, testbatch2)
        cluster_centers = Variable(torch.Tensor(cluster_centers).to(device))
        loss = -log_likelihood(mu_1, mu_2, log_sigma_1, log_sigma_2, rho, target_tensor, cluster_centers)
    else:
        loss = -log_likelihood(mu_1, mu_2, log_sigma_1, log_sigma_2, rho, target_tensor, None)



    loss = loss if loss >0 else -1*loss
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

def save_model(encoder_stream1, decoder_stream1, encoder_stream2, decoder_stream2, data, sufix, s2, loc=MODEL_LOC):
    torch.save(encoder_stream1.state_dict(), os.path.join(loc.format(data), 'encoder_stream1_{}{}.pt'.format(data, sufix)))
    torch.save(decoder_stream1.state_dict(), os.path.join(loc.format(data), 'decoder_stream1_{}{}.pt'.format(data, sufix)))
    if s2:
        torch.save(encoder_stream2.state_dict(), os.path.join(loc.format(data), 'encoder_stream2_{}{}.pt'.format(data, sufix)))
        torch.save(decoder_stream2.state_dict(), os.path.join(loc.format(data), 'decoder_stream2_{}{}.pt'.format(data, sufix)))
    print('model saved at {}'.format(loc.format(data)))


def train_stream2(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer):

    Hidden_State , _ = encoder.loop(input_tensor)
    stream2_out,_, _, _, _, _, _, _ = decoder.loop(Hidden_State)

    l = nn.MSELoss()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = l(stream2_out, target_tensor)
    loss.backward()

    # loss = -log_likelihood(mu_1, mu_2, log_sigma_1, log_sigma_2, rho, target_tensor)
    # print(loss)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), stream2_out

def generate(inputs, encoder, decoder):
    with torch.no_grad():
        Hidden_State , Cell_State = encoder.loop(inputs)
        decoder_hidden , decoder_cell , mu_1 , mu_2 , log_sigma_1 , log_sigma_2 , rho = decoder.loop(Hidden_State)
        [ batch_size , step_size , fea_size ] = mu_1.size ()
        out = []
        for i in range(batch_size):
            mu1_current = mu_1[ i , : , : ]
            mu2_current = mu_2[ i , : , : ]
            sigma1_current = log_sigma_1[ i , : , : ]
            sigma2_current = log_sigma_2[ i , : , : ]
            rho_current = rho[ i , : , : ]
            out.append(sample(mu1_current , mu2_current , sigma1_current , sigma2_current , rho_current))

        return np.array(out)


def log_likelihood(mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, cluster_centers):
    [batch_size, step_size, fea_size] = y.size()

    epoch_loss = 0
    for i in range(step_size):
        mu1_current = mu_1[:,i,:]
        mu2_current = mu_2[:,i,:]
        if cluster_centers is not None:
            muc_x = cluster_centers[:,i,0]
            muc_y = cluster_centers[:,i,1]
        sigma1_current = log_sigma_1[:,i,:]
        sigma2_current = log_sigma_2[:,i,:]
        rho_current = rho[:,i,:]
        y_current = y[:,i,:]
        if cluster_centers is None:
            batch_loss = compute_sample_loss(mu1_current, mu2_current, sigma1_current, sigma2_current, rho_current, y_current).sum()
        else:
            batch_loss = compute_sample_loss(mu1_current, mu2_current, sigma1_current, sigma2_current, rho_current, y_current).sum() + torch.sqrt(torch.sum((mu1_current-muc_x)**2) + torch.sum((mu2_current-muc_y)**2))
        batch_loss = batch_loss/batch_size
        epoch_loss += batch_loss
    return epoch_loss

def compute_sample_loss(mu_1, mu_2,log_sigma_1, log_sigma_2, rho, y):
    const = 1E-20 # to prevent numerical error
    pi_term = torch.Tensor([2*np.pi]).to(device)

    y_1 = y[:,0]
    # y_1 = (y_1-torch.mean(y_1))/y_1.max()
    y_2 = y[:,1]
    # y_2 = (y_2 - torch.mean(y_2))/y_2.max()
    mu_1 = torch.mean(y_1) + (y_1 -torch.mean(mu_1))
    # mu_1 = torch.mean(y_1) + (y_1 -torch.mean(mu_1)) * (torch.std(y_1)/torch.std(mu_1))
    mu_2 = torch.mean(y_2) + (y_2 -torch.mean(mu_2))
    # mu_2 = torch.mean(y_2) + (y_2 -torch.mean(mu_2)) * ((torch.std(y_2))/(torch.std(mu_2)))
    z = ( (y_1 - mu_1)**2/(log_sigma_1**2) + ((y_2 - mu_2)**2/(log_sigma_2**2)) - 2*rho*(y_1-mu_1)*(y_2-mu_2)/((log_sigma_1 *log_sigma_2)) )
    mog_lik2 = torch.exp( (-1*z)/(2*(1-rho**2)) )
    mog_lik1 =  1/(pi_term * log_sigma_1 * log_sigma_2 * (1-rho**2).sqrt() )
    mog_lik = (mog_lik1*(mog_lik2+1e-8)).log()
    return mog_lik

def calculate_cluster_centers(agent_IDs, stream2_output, testbatch2):
    stream2_output = stream2_output.cpu().detach().numpy()
    num_batches = len(testbatch2)
    num_steps = stream2_output.shape[1]
    cluster_centers = np.zeros((num_batches, num_steps, 2))
    km = KMeans(init='k-means++', n_clusters=3)
    for t in range(num_steps):
        for b in range(len(testbatch2) - 1):
            clusters = km.fit(stream2_output[b, t, :].reshape(-1, 1))
            cluster_indcs = np.where(clusters.labels_ == clusters.labels_[agent_IDs[b]])[0]
            testbatch2_item = testbatch2[b]
            keys = list(testbatch2_item.keys())
            coords = testbatch2_item[keys[2 + t]][:, cluster_indcs]
            cluster_centers[b, t, :] = np.mean(coords, axis=1)
    return cluster_centers

# ====================================== HELPER FUNCTIONS =========================================

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def sample(mu_1 , mu_2 , log_sigma_1 , log_sigma_2, rho):

    sample = []
    for i in range(len(mu_1)):
        mu = np.array([mu_1[i][0].item(), mu_2[i][0].item()])
        sigma_1 = log_sigma_1[i][0].item()
        sigma_2 = log_sigma_2[i][0].item()
        c = rho[i][0].item() * sigma_1 * sigma_2
        cov = np.array([[sigma_1**2, c],[c, sigma_2**2]])
        sample.append(np.random.multivariate_normal(mu, cov))

    return sample

def computeDist ( x1 , y1, x2, y2 ):
    return np.sqrt( pow ( x1 - x2 , 2 ) + pow ( y1 - y2 , 2 ) )

def computeKNN ( curr_dict , ID , k ):
    import heapq
    from operator import itemgetter

    ID_x = curr_dict[ ID ][0]
    ID_y = curr_dict[ ID ][1]
    dists = {}
    for j in range ( len ( curr_dict ) ):
        if j != ID:
            dists[ j ] = computeDist ( ID_x , ID_y, curr_dict[ j ][0],curr_dict[ j ][1] )
    KNN_IDs = dict ( heapq.nsmallest ( k , dists.items () , key=itemgetter ( 1 ) ) )
    neighbors = list ( KNN_IDs.keys () )

    return neighbors

def compute_A ( frame ):
    A = np.zeros ( [ frame.shape[ 0 ] , frame.shape[ 0 ] ] )
    for i in range ( len ( frame ) ):
        if frame[ i ] is not None:
            neighbors = computeKNN ( frame , i , 4 )
        for neighbor in neighbors:
            A[ i ][ neighbor ] = 1
    return A


def compute_accuracy_stream1(traindataloader, labeldataloader, encoder, decoder, n_epochs):
    ade = 0
    fde = 0
    count = 0


    train_raw = traindataloader
    pred_raw = labeldataloader
    train2_raw = []
    pred2_raw = []

    batch = load_batch(0, BATCH_SIZE, 'pred', train_raw, pred_raw, train2_raw, pred2_raw, [], [])
    batch, _, _ = batch
    batch_in_form = np.asarray([batch[i]['sequence'] for i in range(BATCH_SIZE)])
    batch_in_form = torch.Tensor(batch_in_form)
    [ batch_size , step_size , fea_size ] = np.shape(batch_in_form)

    print('computing accuracy...')
    for epoch in range(0, n_epochs):
        # Prepare train and test batch
        if epoch % (int(n_epochs/10) + 1) == 0:
            print("{}/{} in computing accuracy...".format(epoch, n_epochs))
        trainbatch_both = load_batch ( epoch , BATCH_SIZE , 'train' , train_raw , pred_raw, train2_raw, pred2_raw, [], [] )
        trainbatch, trainbatch2, _ = trainbatch_both
        trainbatch_in_form = np.asarray([trainbatch[i]['sequence'] for i in range(len(trainbatch))])
        trainbatch_in_form = torch.Tensor ( trainbatch_in_form )

        testbatch_both = load_batch ( epoch , BATCH_SIZE , 'pred' , train_raw , pred_raw, train2_raw, pred2_raw, [], [] )
        testbatch, testbatch2, _  = testbatch_both
        testbatch_in_form = np.asarray([testbatch[i]['sequence'] for i in range(len(trainbatch))])
        testbatch_in_form = torch.Tensor ( testbatch_in_form )

        train = trainbatch_in_form.to(device)
        label = testbatch_in_form.to(device)

        pred = generate(train, encoder, decoder)
        mse = MSE(pred, label)
        # print(mse)
        mse = np.sqrt(mse)
        ade += mse
        fde += mse[-1]
        # count += testbatch_in_form.size()[0]
        count +=1  
    
    ade = ade/count
    fde = fde/count
    print('RMSE: {}'.format(ade))
    print("ADE: {} FDE: {}".format(np.mean(ade), fde))

def makeplot(x, y, x_label, y_label, title, save_loc):
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    fig.savefig(os.path.join(save_loc, title+'.png'))
    fig.clear()

def MSE(y_pred, y_gt, device=device):
    # y_pred = y_pred.numpy()
    y_gt = y_gt.cpu().detach().numpy()
    acc = np.zeros(np.shape(y_pred)[:-1])
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = np.array(y_gt[:,:, 0])
    x = (x-np.mean(x))/x.std()
    y = np.array(y_gt[:,:, 1])
    # muX = np.mean(x) + (x - np.mean(muX)) * (np.std(x))/(np.std(muX))
    # muX = np.mean(x) + (x - np.mean(muX))
    y = (y-np.mean(y))/y.std()
    # muY = np.mean(y) + (y -np.mean(muY)) * (np.std(y))/(np.std(muY))
    # muY = np.mean(y) + (y -np.mean(muY))
    acc = np.power(x-muX, 2) + np.power(y-muY, 2)
    lossVal = np.sum(acc, axis=0)/len(acc)
    return lossVal

def compute_eigs ( train_stream2):
    N = train_stream2[0][list ( train_stream2[ 0 ].keys())[ -1]].shape[ 1 ]
    A = np.zeros ( [ N, N ] )
    frame = {}
    eig_batch = []
    for batch_idx in range(len(train_stream2)):
        eig_frame = []
        for which_frame in list ( train_stream2[ batch_idx ].keys())[ 2: ]:
            for j in range(N):
                frame[j] = [train_stream2[batch_idx][which_frame][0,j],train_stream2[batch_idx][which_frame][1,j]]
            for l in range (N):
                if frame[ l ] is not None:
                    neighbors = computeKNN ( frame , l , 4 )
                for neighbor in neighbors:
                    # if neighbor in labels:
                    # if idx < labels.index ( neighbor ):
                    dist_of_neighbor = computeDist(frame[l][0],frame[l][1], frame[neighbor][0],frame[neighbor][1])
                    if dist_of_neighbor <= MU:
                        A[ l ][ neighbor ] = np.exp(-1*computeDist(frame[l][0],frame[l][1], frame[neighbor][0],frame[neighbor][1]))
            d = [ np.sum ( A[ row , : ] ) for row in range ( A.shape[ 0 ] ) ]
            D = np.diag ( d )
            L = D - A
            _, vecs = eigs(L, k=2 )
            eig_frame.append(np.real(vecs[:,1]))
        eig_batch.append ( np.array(eig_frame) )
    return torch.Tensor(np.array(eig_batch))
