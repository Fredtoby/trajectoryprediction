import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda:0")

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

    def forward ( self , input , Hidden_State , Cell_State ):
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
        Stream_output  = torch.cat((mu1_all,mu2_all), dim=2)
        return Stream_output , Cell_State

    def initHidden(self):
        out = torch.randn(self.batch_size, self.num_mog_params, device=device) if self.stream == 's1' else torch.randn(self.batch_size, self.hidden_size, device=device)
        if self.stream == 's2':
            output =  torch.randn(self.batch_size, self.time_step, self.hidden_size, device=device)
            return torch.randn(self.batch_size, self.hidden_size, device=device), out, output
        else:
            return torch.randn ( self.batch_size , self.hidden_size , device=device ) , out

    def initMogParams(self):
        mu1_all = torch.randn(self.batch_size, self.time_step, 1, device=device)
        mu2_all = torch.randn(self.batch_size, self.time_step, 1, device=device)
        sigma1_all = torch.randn(self.batch_size, self.time_step, 1, device=device)
        sigma2_all = torch.randn(self.batch_size, self.time_step, 1, device=device)
        rho_all = torch.randn(self.batch_size, self.time_step, 1, device=device)

        return mu1_all, mu2_all, sigma1_all, sigma2_all, rho_all
