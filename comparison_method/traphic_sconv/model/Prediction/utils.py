from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
import pickle
import os
from klepto.archives import dir_archive

#___________________________________________________________________________________________________________________________

def lstToCuda(lst):
    for item in lst:
        item.cuda()
    return lst

def klepto_load(loc):
    '''
    for loading the dumped dictionarys
    :return: loaded dictionary
    '''

    dic = dir_archive(loc, {}, serialized=True)
    dic.load()
    print('dictionary loaded')
    return dic

### Dataset class for the NGSIM dataset
class ngsimDataset(Dataset):


    def __init__(self, file_name, data_dir, track_dir, dtype, dsId, device='cuda:0', set=30, t_h=24, t_f=36, d_s=1 , enc_size = 64, grid_size = (13,3), upp_grid_size = (7,3)):
        
        # d = np.memmap(file_name + '-traj.dat', dtype=np.float64, mode='r')
        # self.D = np.reshape(d, (-1, 47))
        # self.T = klepto_load(file_name + '-track.kpt')

        # tracks = data[1]
        # # transform self.T to desired sparse matrix
        # # maxdid = int(max(tracks.keys()))
        # maxvehicleset = []
        # for v in tracks.values():
        #     maxvehicleset.append(max(v.keys()))
        # maxvehicle = int(max(maxvehicleset))

        # temp = {}
        

        # for k in tracks.keys():
        #     temp[k] = np.full(maxvehicle, 0, dtype=object)

        #     for i in range(maxvehicle):
        #         if (i + 1) in tracks[k].keys():
        #             temp[k][i] = tracks[k][i + 1]
        #         else:
        #             temp[k][i] = np.empty((1,0))



        # if dtype == 'test':
        #     track_loc = os.path.join(track_dir, dtype+'_obs', 'formatted')
        # else:
        #     track_loc = os.path.join(track_dir, dtype, 'formatted')
        # self.t_map = []
        # for i in range(set):
        #     self.t_map.append([int(f.split('.')[0]) for f in os.listdir(track_loc + '/set{}'.format(i)) if '.npy' in f])
        self.dtype = dtype
        self.data_dir = data_dir
        self.dset = dsId
        # t = np.load(os.path.join(self.data_dir, self.dtype, "{}Set{}-track.npy".format(self.dtype, self.dset)), allow_pickle=True)
        # self.T = t[0]

        d= np.load(os.path.join(self.data_dir, self.dtype, "{}Set{}-traj.npy".format(self.dtype, self.dset)), allow_pickle=True)
        self.D = d[0]
        t = np.load(os.path.join(self.data_dir, self.dtype, "{}Set{}-track.npy".format(self.dtype, self.dset)), allow_pickle=True)
        self.T = t[0]

        # for ids in self.D[:,1]:
        #     if ids not in self.T[:,1]
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size # size of encoder LSTM
        self.grid_size = grid_size # size of social context grid
        self.upp_grid_size = upp_grid_size #                                           #Behavioral Modification 2: Adding Kinetic Flow layer
        self.inds = [14,15,16,17,18,19,20,27,28,29,30,31,32,33, 40,41,42,43,44,45, 46]
        # self.inds = [32,33,34,35]
        self.dwn_inds = [8,9,10,11,12,13,21,22,23,24,25,26,34,35,36,37,38, 39]
        # self.dwn_inds = [35,36,37,38, 39]
        self.device = device
        self.ddd = [128, 128, 128, 129, 133, 133, 128]
        self.vvv = [8, 11, 53, 21, 13, 22, 52]
        self.fff = [2, 3, 2, 48, 2, 2, 16]
        


    def __len__(self):
        return len(self.D)



    def __getitem__(self, idx):
        dsId = self.D[idx, 0].astype(int)# Dataset ID
        vehId = self.D[idx, 1].astype(int)# Vehicle ID
        t = self.D[idx, 2] # Frame
        grid = self.D[idx,8:47] # Nhbr Info
        upp_grid = self.D[idx,self.inds]
        neighbors = []
        upper_neighbors = []
        # print(dsId)
        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        # if dsId not in self.t_map[self.dset]:
        #     self.dset = 0
        #     while dsId not in self.t_map[self.dset]:
        #         self.dset += 1
        # dt = np.load(os.path.join(self.data_dir, self.dtype, "{}Set{}-track.npy".format(self.dtype, self.dset)), allow_pickle=True)
        # self.T = dt[0]
        current = np.array([self.D[idx, 3:5]])
        hist = self.getHistory(vehId,t,vehId,dsId,current)
        fut = self.getFuture(vehId,t,dsId)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t,vehId,dsId, current))

                                                                                        #Behavioral Modification 2: Adding Kinetic Flow layer
        for i in upp_grid:
            upper_neighbors.append(self.getHistory(i.astype(int), t,vehId,dsId, current))

        upp_count = np.count_nonzero(upp_grid)
        dwn_count = np.count_nonzero(self.D[idx,self.dwn_inds])
        hist = np.concatenate((hist, np.array([[upp_count, dwn_count]])), axis=0)

        # Maneuvers 'lon_enc' = one-hot vector, 'lat_enc = one-hot vector
        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 7] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 6] - 1)] = 1

        # print(type(hist))
        # print(type(fut))
        # print(type(upper_neighbors))
        # print(type(neighbors))
        # print(type(lat_enc))
        # print(type(lon_enc))

        # print(np.shape(hist)) # 1, 2
        # print(np.shape(fut)) # x, 2
        # print(np.shape(upper_neighbors)) # 21, 0, 2
        # print(np.shape(neighbors)) # 39, 0, 2
        # print(np.shape(lat_enc)) # 3, 
        # print(np.shape(lon_enc)) # 2, 
         
        if dsId in self.ddd:
            idx = 0
            while self.ddd[idx] != dsId:
                idx += 1
            if vehId == self.vvv[idx] and t == self.fff[idx]:
                bb = True
            else:
                bb = False
        else:
            bb = False

        return hist,fut,upper_neighbors, neighbors,lat_enc,lon_enc, bb, dsId, vehId, t



    ## Helper function to get track history
    def getHistory(self,vehId,t,refVehId,dsId, current):
        if vehId == 0:
            return current
        else:
            # if self.T[dsId].size<=vehId-1:
            # if self.T[dsId].size<=vehId-1 or self.T[dsId][vehId-1].size==0:
            if not vehId in self.T[dsId].keys():
                return current
            # refTrack = (self.T[dsId][refVehId-1].transpose()).astype(float)
            # vehTrack = (self.T[dsId][vehId-1].transpose()).astype(float)
            refTrack = (self.T[dsId][refVehId].transpose()).astype(float)
            vehTrack = (self.T[dsId][vehId].transpose()).astype(float)
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]

            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return current
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item(0) - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item(0) + 1
                hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos

            if len(hist) < self.t_h//self.d_s + 1:
                return current

                                                                                                            # Behavioral Modification 3: Change inputs
            m1 = int(self.t_h/3)
            m2 = 2 * m1
            vel0 = np.array([[hist[m1][0] - hist[0][0], hist[m1][1] -hist[0][1]]])
            vel5 = np.array([[hist[m2][0] - hist[m1][0], hist[m2][1] -hist[m1][1]]])
            vel10 = np.array([[hist[self.t_h][0] - hist[m2][0], hist[self.t_h-1][1] -hist[m2][1]]])
            hist = np.concatenate((hist, np.concatenate((vel0,vel5,vel10), axis=0)), axis=0)
            return hist



    ## Helper function to get track future
    def getFuture(self, vehId, t,dsId):
        # vehTrack = (self.T[dsId][vehId-1].transpose()).astype(float)
        vehTrack = (self.T[dsId][vehId].transpose()).astype(float)
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item(0) + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item(0) + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
        # print(dsId, vehId, vehTrack[stpt:enpt:self.d_s,1:3])
        return fut



    ## Collate function for dataloader
    def collate_fn(self, samples):

        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _,_,upp_nbrs, nbrs,_,_, _, _, _, _ in samples:
            nbr_batch_size += sum([len(nbrs[i])!=0 for i in range(len(nbrs))])
        maxlen = self.t_h//self.d_s + 1+4                                                           # Behavioral Modification 3: Change inputs/ change max len to +3
        nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2)

                                                                                                    # Behavioral Modification 2: Adding Kinetic Flow layer

        upp_nbr_batch_size = 0
        for _,_,upp_nbrs, nbrs,_,_, _, _, _, _ in samples:
            upp_nbr_batch_size += sum([len(upp_nbrs[i])!=0 for i in range(len(upp_nbrs))])
        upp_maxlen = self.t_h//self.d_s + 1+3                                                               # Behavioral Modification 3: Change inputs/ change max len to +3
        upp_nbrs_batch = torch.zeros(upp_maxlen,upp_nbr_batch_size,2)

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.enc_size)
        mask_batch = mask_batch.byte()

        upp_pos = [0,0]                                                                                 # Behavioral Modification 2: Adding Kinetic Flow layer
        upp_mask_batch = torch.zeros(len(samples), self.upp_grid_size[1],self.upp_grid_size[0],self.enc_size)
        upp_mask_batch = upp_mask_batch.byte()


        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen,len(samples),2)
        fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        lat_enc_batch = torch.zeros(len(samples),3)
        lon_enc_batch = torch.zeros(len(samples), 2)
        bb_batch = np.zeros(len(samples))
        dd_batch = np.zeros(len(samples))
        vv_batch = np.zeros(len(samples))
        ff_batch = np.zeros(len(samples))


        count = 0
        upp_count = 0
        for sampleId,(hist, fut, upp_nbrs, nbrs, lat_enc, lon_enc, bb, dd, vv, ff) in enumerate(samples):
            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut),sampleId,:] = 1
            lat_enc_batch[sampleId,:] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            bb_batch[sampleId] = bb
            dd_batch[sampleId] = dd
            vv_batch[sampleId] = vv
            ff_batch[sampleId] = ff

            # Set up neighbor, neighbor sequence length, and mask batches:
            for id,nbr in enumerate(nbrs):
                if len(nbr)!=0:
                    nbrs_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId,pos[1],pos[0],:] = torch.ones(self.enc_size).byte()
                    count+=1

                                                                                                   # Behavioral Modification 2: Adding Kinetic Flow layer
            for id, upp_nbr in enumerate(upp_nbrs):
                if len(upp_nbr) != 0:
                    upp_nbrs_batch[0:len(upp_nbr), upp_count, 0] = torch.from_numpy(upp_nbr[:, 0])
                    upp_nbrs_batch[0:len(upp_nbr), upp_count, 1] = torch.from_numpy(upp_nbr[:, 1])
                    upp_pos[0] = id % self.upp_grid_size[0]
                    upp_pos[1] = id // self.upp_grid_size[0]
                    upp_mask_batch[sampleId, upp_pos[1], upp_pos[0], :] = torch.ones(self.enc_size).byte()
                    upp_count += 1


        return hist_batch, upp_nbrs_batch, nbrs_batch, upp_mask_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch, bb_batch, dd_batch, vv_batch, ff_batch

#________________________________________________________________________________________________________________________________________





## Custom activation for output layer (Graves, 2015)
def outputActivation(x):
    muX = x[:,:,0:1]
    muY = x[:,:,1:2]
    sigX = x[:,:,2:3]
    sigY = x[:,:,3:4]
    rho = x[:,:,4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
    return out

## Batchwise NLL loss, uses mask for variable output lengths
def maskedNLL(y_pred, y_gt, mask, device='cpu'):
    acc = torch.zeros_like(mask, device=device)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    sigX = y_pred[:,:,2]
    sigY = y_pred[:,:,3]
    rho = y_pred[:,:,4]
    ohr = torch.pow(1-torch.pow(rho,2),-0.5)
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr)
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
#    print(acc[0])
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, device='cpu',num_lat_classes=3, num_lon_classes = 2,use_maneuvers = True, avg_along_time = False, cuda=True):
    if use_maneuvers:
        if cuda:
            acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lon_classes*num_lat_classes).cuda(device)
        else:
            acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lon_classes*num_lat_classes)
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:,l]*lon_pred[:,k]
                wts = wts.repeat(len(fut_pred[0]),1)
                y_pred = fut_pred[k*num_lat_classes + l]
                y_gt = fut
                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = y_pred[:, :, 2]
                sigY = y_pred[:, :, 3]
                rho = y_pred[:, :, 4]
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
                out = -(torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY,2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr))
                acc[:, :, count] =  out + torch.log(wts)
                count+=1
        acc = -logsumexp(acc,dim = 2)
        acc = acc * op_mask[:,:,0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc,dim=1)
            counts = torch.sum(op_mask[:,:,0],dim=1)
            return lossVal,counts
    else:
        if cuda:
            acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda(device)
        else:
            acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1)
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = torch.pow(ohr, 2) * (
        torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(
            sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr)
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:,:,0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal,counts

## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask, device='cpu'):
    acc = torch.zeros_like(mask, device=device)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
#    print(acc)
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMSETest(y_pred, y_gt, mask, device='cpu'):
    acc = torch.zeros_like(mask, device=device)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
