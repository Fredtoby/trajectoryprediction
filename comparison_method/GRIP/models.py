import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda:0")

class GRIPModel(nn.Module):
    def __init__(self, coordinates, timesteps):
        super(GRIPModel, self).__init__()
        self.coordinates = coordinates
        self.Conv1 = nn.Conv2d(coordinates, 64, kernel_size=(1, 3), padding=(0, 1))
        self.Conv2 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))

        self.Conv3 = nn.Conv2d(64, 128, padding=(0, 1),kernel_size=(1, 3), stride=(1, 2 if timesteps >= 2 else 1))
        self.Conv4 = nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1))

        self.Conv5 = nn.Conv2d(128, 256, kernel_size=(1, 3), padding=(0, 0 if timesteps >= 4 else 1), stride=(1, 2 if timesteps >= 4 else 1))

        self.Dropout = nn.Dropout(0.5)

    def graph_operation(self, x, adj, alpha=0.001):
        x_in = x.permute(0, 2, 1, 3)
        lambda_mat = torch.zeros_like(adj)
        diags = torch.sum(adj, dim=1) + alpha
        diag_indcs = np.diag_indices(lambda_mat.shape[1])
        lambda_mat[:, diag_indcs[0], diag_indcs[1], :] = diags
        lambda_inv = 1 / lambda_mat
        lambda_inv[lambda_inv == float("inf")] = 0
        lambda_inv_sqrt = torch.sqrt(lambda_inv)
        x_out = torch.zeros_like(x_in)
        for t in range(x_in.shape[3]):
            x_out[:, :, :, t] = torch.matmul(lambda_inv[:, :, :, t] +
                                             torch.matmul(lambda_inv_sqrt[:, :, :, t],
                                                          torch.matmul(adj[:, :, :, t], lambda_inv_sqrt[:, :, :, t])),
                                             x_in[:, :, :, t])
        return x_out.permute(0, 2, 1, 3)

    def computeDist (self, x1 , y1 , x2 , y2 ):
        # return np.abs ( x1 - y1 )
        return np.sqrt ( pow ( x1 - x2 , 2 ) + pow ( y1 - y2 , 2 ) )

    def computeKNN (self, curr_dict , ID , k ):
        import heapq
        from operator import itemgetter

        ID_x = curr_dict[ ID ][ 0 ]
        ID_y = curr_dict[ ID ][ 1 ]
        dists = {}
        for j in range ( len ( curr_dict ) ):
            if j != ID:
                dists[ j ] = self.computeDist ( ID_x , ID_y , curr_dict[ j ][ 0 ] , curr_dict[ j ][ 1 ] )
        KNN_IDs = dict ( heapq.nsmallest ( k , dists.items () , key=itemgetter ( 1 ) ) )
        neighbors = list ( KNN_IDs.keys () )

        return neighbors

    def compute_A ( self , xt ):
        # return Variable(torch.Tensor(np.ones([xt.shape[0],xt.shape[0]])).cuda())
        xt = xt.cpu ().detach ().numpy ()
        A = np.zeros ( [ xt.shape[ 0 ] , xt.shape[ 0 ] ] )
        for i in range ( len ( xt ) ):
            if xt[i][0] and xt[i][1]:
                neighbors = self.computeKNN ( xt , i , 4 )
                for neighbor in neighbors:
                    # if neighbor in labels:
                    # if idx < labels.index ( neighbor ):
                    A[ i ][ neighbor ] = 1
        return Variable ( torch.Tensor ( A ).to (device) )

    def forward(self, x):
        Ats = torch.zeros((x.shape[0], x.shape[2], x.shape[2], x.shape[3])).to(device)
        for b in range(x.shape[0]):
            if b % 5 == 0:
                print('{}/{}'.format(b, x.shape[0]))
            for t in range(x.shape[3]):

                Ats[b, :, :, t] = self.compute_A(x[b, :, :, t].permute(1, 0))
        # print(1)
        x = self.Conv1(x)
        x = self.Dropout(self.graph_operation(x, Ats))
        x = self.Conv2(x)
        x = self.Dropout(self.graph_operation(x, Ats))
        # print(2)
        x = self.Conv3(x)
        x = self.Dropout(self.graph_operation(x, Ats))
        x = self.Conv4(x)
        x = self.Dropout(self.graph_operation(x, Ats))

        x = self.Conv5(x)
        x = self.Dropout(self.graph_operation(x, Ats))

        return x


class Encoder ( nn.Module ):
    def __init__ ( self , final_coordinate_size , n_agents  ):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super ( Encoder , self ).__init__ ()

        self.n_agents = n_agents
        self.coordinates = final_coordinate_size
        self.fl = nn.Linear ( n_agents , n_agents)
        self.il = nn.Linear ( n_agents , n_agents)
        self.ol = nn.Linear ( n_agents, n_agents)
        self.Cl = nn.Linear ( n_agents , n_agents)


    def forward ( self , input , Hidden_State , Cell_State ):
        # print(input.shape[1])
        print(Hidden_State.shape[1])
        combined = Hidden_State
        f = torch.sigmoid ( self.fl ( combined ) )
        i = torch.sigmoid ( self.il ( combined ) )
        o = torch.sigmoid ( self.ol ( combined ) )
        C = torch.tanh ( self.Cl ( combined ) )
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * torch.tanh ( Cell_State )

        return Hidden_State , Cell_State

    def loop ( self , inputs ):
        batch_size = inputs.size ( 0 )
        time_step = inputs.size ( 3 )
        Hidden_State , Cell_State = self.initHidden ( batch_size )
        for i in range ( time_step ):
            if i==0:
                Hidden_State = inputs[:,:,:,i]
            Hidden_State , Cell_State = self.forward(inputs[:,:,: ,i], Hidden_State, Cell_State)
        return Hidden_State , Cell_State

    def initHidden ( self , batch_size ):
        Hidden_State = Variable ( torch.zeros ( batch_size , self.coordinates, self.n_agents ).to ( device ) )
        Cell_State = Variable ( torch.zeros ( batch_size , self.coordinates, self.n_agents).to ( device ) )
        return Hidden_State , Cell_State


class Decoder(nn.Module):
    def __init__(self, final_coordinate_size, batchsize, n_agents, timestep):
        super(Decoder, self).__init__()
        self.coordinates = final_coordinate_size
        self.batch_size = batchsize
        self.time_step = timestep
        self.n_agents = n_agents
        self.fl = nn.Linear ( n_agents , n_agents)
        self.il = nn.Linear ( n_agents , n_agents)
        self.ol = nn.Linear ( n_agents, n_agents)
        self.Cl = nn.Linear ( n_agents , n_agents)
        self.linear = nn.Linear ( final_coordinate_size ,  2)


    def forward(self , Hidden_State , Cell_State):

        # combined = torch.cat ( (input , Hidden_State) , 1 )
        combined = Hidden_State
        f = torch.sigmoid ( self.fl ( combined ) )
        i = torch.sigmoid ( self.il ( combined ) )
        o = torch.sigmoid ( self.ol ( combined ) )
        C = torch.tanh ( self.Cl ( combined ) )
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * torch.tanh ( Cell_State )

        return Hidden_State , Cell_State

    def loop ( self, hidden_vec_from_encoder ):
        time_step = self.time_step
        Cell_State, stream2_output = self.initHidden()
        for i in range ( time_step ):
            if i == 0:
                Hidden_State = hidden_vec_from_encoder
            Hidden_State , Cell_State = self.forward( Hidden_State , Cell_State )
            stream2_output[:,:,:,i] = self.linear(Hidden_State.permute(0,2,1)).permute(0,2,1)
        return stream2_output, Hidden_State , Cell_State

    def initHidden(self):
        # out = torch.zeros(self.batch_size , self.coordinates, self.n_agents , device=device)
        output =  torch.zeros(self.batch_size , 2, self.n_agents , self.time_step, device=device)
        return torch.zeros(self.batch_size , self.coordinates, self.n_agents , device=device), output
