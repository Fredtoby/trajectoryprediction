import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda")

class GraphConvolution ( nn.Module ):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__ ( self , in_features , out_features , bias=True ):
        super ( GraphConvolution , self ).__init__ ()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter ( torch.FloatTensor ( in_features , out_features ).cuda () )
        if bias:
            self.bias = Parameter ( torch.FloatTensor ( out_features ).cuda () )
        else:
            self.register_parameter ( 'bias' , None )
        self.reset_parameters ()

    def reset_parameters ( self ):
        stdv = 1. / math.sqrt ( self.weight.size ( 1 ) )
        self.weight.data.uniform_ ( -stdv , stdv )
        if self.bias is not None:
            self.bias.data.uniform_ ( -stdv , stdv )

    def forward ( self , input , adj ):
        support = torch.mm ( input , self.weight )
        output = torch.spmm ( adj , support )
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__ ( self ):
        return self.__class__.__name__ + ' (' \
               + str ( self.in_features ) + ' -> ' \
               + str ( self.out_features ) + ')'


class GCN ( nn.Module ):
    def __init__ ( self , nfeat , nhid , nclass , dropout ):
        super ( GCN , self ).__init__ ()

        self.gc1 = GraphConvolution ( nfeat , nhid )
        self.gc2 = GraphConvolution ( nhid , nclass )
        self.dropout = dropout

    def forward ( self , x , adj ):
        x = F.relu ( self.gc1 ( x , adj ) )
        x = F.dropout ( x , self.dropout , training=self.training )
        x = self.gc2 ( x , adj )
        # return F.log_softmax(x, dim=1)
        return x


class Encoder ( nn.Module ):
    def __init__ ( self , input_size , cell_size , hidden_size ):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super ( Encoder , self ).__init__ ()

        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear ( input_size + hidden_size , hidden_size )
        self.il = nn.Linear ( input_size + hidden_size , hidden_size )
        self.ol = nn.Linear ( input_size + hidden_size , hidden_size )
        self.Cl = nn.Linear ( input_size + hidden_size , hidden_size )

    def computeDist ( self , x1 , y1 ):
        return np.abs ( x1 - y1 )
        # return sqrt ( pow ( x1 - x2 , 2 ) + pow ( y1 - y2 , 2 ) )

    def computeKNN ( self , curr_dict , ID , k ):
        import heapq
        from operator import itemgetter

        ID_x = curr_dict[ ID ]
        dists = {}
        for j in range ( len ( curr_dict ) ):
            if j != ID:
                dists[ j ] = self.computeDist ( ID_x , curr_dict[ j ] )
        KNN_IDs = dict ( heapq.nsmallest ( k , dists.items () , key=itemgetter ( 1 ) ) )
        neighbors = list ( KNN_IDs.keys () )

        return neighbors
        # return [1,2,3]

    def compute_A ( self , xt ):
        # return Variable(torch.Tensor(np.ones([xt.shape[0],xt.shape[0]])).cuda())
        xt = xt.cpu ().detach ().numpy ()
        A = np.zeros ( [ xt.shape[ 0 ] , xt.shape[ 0 ] ] )
        for i in range ( len ( xt ) ):
            #if xt[ i ] is not None:
            if xt[i][0] and xt[i][1] :
                neighbors = self.computeKNN ( xt , i , 4 )
                for neighbor in neighbors:
                    # if neighbor in labels:
                    # if idx < labels.index ( neighbor ):
                    A[ i ][ neighbor ] = 1
        return Variable ( torch.Tensor ( A ).cuda () )

    def forward ( self , input , Hidden_State , Cell_State ):
        graph = False

        if graph is True:
            gcn_feat = [ ]
            gcn_model = GCN ( nfeat=1 , nhid=16 , nclass=1 , dropout=0.5 )
            for j in range ( input.shape[ 0 ] ):
                features = input[ j , : ]
                gcn_feat.append ( gcn_model ( torch.unsqueeze ( features , dim=1 ) ,
                                              self.compute_A ( features ) ).cpu ().detach ().numpy () )

            input = Parameter ( torch.FloatTensor ( np.asarray ( gcn_feat ) ).cuda () )
            input = torch.squeeze ( input )
        # print(input)
        combined = torch.cat ( (input , Hidden_State) , 1 )
        f = torch.sigmoid ( self.fl ( combined ) )
        i = torch.sigmoid ( self.il ( combined ) )
        o = torch.sigmoid ( self.ol ( combined ) )
        C = torch.tanh ( self.Cl ( combined ) )
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * torch.tanh ( Cell_State )

        return Hidden_State , Cell_State

    def loop ( self , inputs ):
        batch_size = inputs.size ( 0 )
        time_step = inputs.size ( 1 )
        Hidden_State , Cell_State = self.initHidden ( batch_size )
        for i in range ( time_step ):
            Hidden_State , Cell_State = self.forward(torch.squeeze(inputs[:, i:i+1,:]), Hidden_State, Cell_State)
        return Hidden_State , Cell_State

    def initHidden ( self , batch_size ):
        Hidden_State = Variable ( torch.zeros ( batch_size , self.hidden_size ).to ( device ) )
        Cell_State = Variable ( torch.zeros ( batch_size , self.hidden_size ).to ( device ) )
        return Hidden_State , Cell_State


class Decoder(nn.Module):
    def __init__(self, stream, input_size , cell_size , hidden_size, batchsize, timestep):
        super(Decoder, self).__init__()
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.batch_size = batchsize
        self.time_step = timestep
        self.num_mog_params = 5
        self.sampled_point_size = 2
        self.stream = stream
        self.stream_specific_param = self.num_mog_params
        self.stream_specific_param = input_size if self.stream=='s2' else self.num_mog_params
        self.fl = nn.Linear ( self.stream_specific_param + hidden_size , hidden_size )
        self.il = nn.Linear ( self.stream_specific_param + hidden_size , hidden_size )
        self.ol = nn.Linear ( self.stream_specific_param + hidden_size , hidden_size )
        self.Cl = nn.Linear ( self.stream_specific_param + hidden_size , hidden_size )
        self.linear1 = nn.Linear ( cell_size ,  self.stream_specific_param )
        # self.one_lstm = nn.LSTMCell
        # self.linear2 = nn.Linear ( self.sampled_point_size ,  hidden_size )


    def forward(self, input , Hidden_State , Cell_State):
        graph = False

        if graph is True:
            gcn_feat = [ ]
            gcn_model = GCN ( nfeat=1 , nhid=16 , nclass=1 , dropout=0.5 )
            for j in range ( input.shape[ 0 ] ):
                features = input[ j , : ]
                gcn_feat.append ( gcn_model ( torch.unsqueeze ( features , dim=1 ) ,
                                              self.compute_A ( features ) ).cpu ().detach ().numpy () )

            input = Parameter ( torch.FloatTensor ( np.asarray ( gcn_feat ) ).cuda () )
            input = torch.squeeze ( input )
        # print(input)
        combined = torch.cat ( (input , Hidden_State) , 1 )
        f = torch.sigmoid ( self.fl ( combined ) )
        i = torch.sigmoid ( self.il ( combined ) )
        o = torch.sigmoid ( self.ol ( combined ) )
        C = torch.tanh ( self.Cl ( combined ) )
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * torch.tanh ( Cell_State )

        return Hidden_State , Cell_State

    def loop ( self, hidden_vec_from_encoder ):
        batch_size = self.batch_size
        time_step = self.time_step
        if self.stream =='s2':
            Cell_State, out, stream2_output = self.initHidden()
        else:
            Cell_State , out  = self.initHidden ()
        mu1_all , mu2_all , sigma1_all , sigma2_all , rho_all = self.initMogParams()
        for i in range ( time_step ):
            if i == 0:
                Hidden_State = hidden_vec_from_encoder
            Hidden_State , Cell_State = self.forward( out , Hidden_State , Cell_State )
            # print(Hidden_State.data)
            mog_params = self.linear1(Hidden_State)
            # mog_params = params.narrow ( -1 , 0 , params.size ()[ -1 ] - 1 )
            out = mog_params
            if self.stream == 's2':
                stream2_output[:,i,:] = out
            if self.stream == 's1':
                mu_1 , mu_2 , log_sigma_1 , log_sigma_2 , pre_rho = mog_params.chunk ( 6 , dim=-1 )
                rho = torch.tanh ( pre_rho )
                log_sigma_1 = torch.exp(log_sigma_1)
                log_sigma_2 = torch.exp(log_sigma_2)
                mu1_all[:,i,:] = mu_1
                mu2_all[:,i,:] = mu_2
                sigma1_all[:,i,:] = log_sigma_1
                sigma2_all[:,i,:] = log_sigma_2
                rho_all[:,i,:] = rho
            # print(mu1_all.grad_fn)
            # out = self.sample(mu_1 , mu_2 , log_sigma_1 , log_sigma_2, rho)
        if self.stream == 's1':
            return Hidden_State , Cell_State, mu1_all , mu2_all , sigma1_all , sigma2_all , rho_all
        else:
            return stream2_output, Hidden_State , Cell_State , mu1_all , mu2_all , sigma1_all , sigma2_all , rho_all

    def initHidden(self):
        out = torch.randn(self.batch_size, self.num_mog_params, device=device) if self.stream == 's1' else torch.randn(self.batch_size, self.hidden_size, device=device)
        if self.stream == 's2':
            output =  torch.randn(self.batch_size, self.time_step, self.hidden_size, device=device)
            return torch.randn(self.batch_size, self.hidden_size, device=device), out, output
        else:
            return torch.randn ( self.batch_size , self.hidden_size , device=device ) , out

    def initMogParams(self):
        mu1_all = torch.rand(self.batch_size, self.time_step, 1, device=device) * 1000
        mu2_all = torch.rand(self.batch_size, self.time_step, 1, device=device) * 1000
        sigma1_all = torch.rand(self.batch_size, self.time_step, 1, device=device) * 100
        sigma2_all = torch.rand(self.batch_size, self.time_step, 1, device=device) * 100
        rho_all = torch.randn(self.batch_size, self.time_step, 1, device=device)
        
        return mu1_all, mu2_all, sigma1_all, sigma2_all, rho_all
