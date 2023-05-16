'''
File to store implementation of various network heads and models.
This stores all needed architectures.

@Author: George Witt
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    '''
    The multi head attention is a necessary block to work with self-attention.
    See Ho et al and Vaswani et all from 'Attention is all you need', the famous paper.

    We implemented these on the last homework.

    Further deatils are given in the forward call.
    No backward call is provided, gradients are tracked.
    '''

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class SelfAttention(nn.Module):
    '''
    The self attention head is a standard self-attention head as described in
    see Vaswani et al in 'Attention is all you need', the famous paper. 

    We implemented these on the last homework.

    Further details are given in the forward call.
    No backward call is provided, gradients are tracked.
    '''

class BottomConv(nn.Module):
    '''
    The Bottom convolutional layers are the same as in the Up/Down
    sample blocks, just modified without the up/down sampling.
    '''

    def __init__(self):
        super().__init__()

    def forward(self, x, n_channels, in_rect_size, out_rect_size):
        pass

class UpSample(nn.Module):
    '''
    The UpSample block consists of exactly the same material as the DownSample:
        2 coupled convolutional layers
        A residual up sampling connection to increase matrix size.

    Note order is reversed. In the upsampling block we perform the residual
    up sampling connection BEFORE we run the coupled convolutional layers, unlike
    in the down sample block. 

    Further details are given in the forward call.
    No backward call is provided, gradients are tracked.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x, n_channels, in_rect_size, out_rect_size):
        pass

class DownSample(nn.Module):
    '''
    The DownSample block consists of the following:
        2 coupled convolutional layers
        A residual down sampling connection to decrease matrix size.

    Further details are given in the forward call.
    No backward call is provided, gradients are tracked.
    '''
    
    def __init__(self):
        super().__init__()

    def forward(self, x, n_channels, in_rect_size, out_rect_size):
        pass

class DDPM(nn.Module):

    def _preprocess_inputs(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def _initialize_internal_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _initialize_unet_arch(self):
        '''
        The UNET architecture I employ here is very similar to that in (https://arxiv.org/pdf/1505.04597.pdf), Ronneberger et al.
        I make one small modification of including self attention between downsampling.
        This modification is described in (https://arxiv.org/pdf/1503.03585.pdf), Ho et al.

        Note I combine the downsampling sets and upsampling sets into separate modules.
        '''

        # Downward direction
        self.d1 = DownSample()
        self.s1 = SelfAttention()
        self.d2 = DownSample()
        self.s2 = SelfAttention()
        self.d3 = DownSample()
        self.s3 = SelfAttention()
        self.d4 = DownSample()
        self.s4 = SelfAttention()

        # Bottom layers
        self.bottom = BottomConv()

        # Upwards
        self.u1 = UpSample()
        self.s5 = SelfAttention()
        self.u2 = UpSample()
        self.s6 = SelfAttention()
        self.u3 = UpSample()
        self.s7 = SelfAttention()
        self.u4 = UpSample()
        self.s8 = SelfAttention()


    def _initialize_hyperparameters(self, hyperparams):
        pass

    def __init__(self, X, Y, hyperparams):
        '''
        Initialization of DDPM model

        @X: Numpy array N x ... matrix of training values.
            This will be converted to a tensor INTERNALLY !
        @Y: Numpy vector (N,) of true values.
        @hyperparams: Dictionary of necessary hyperparameters
        '''

        super().__init__()

        self._preprocess_inputs(X, Y)                   # Prepare tensors
        self._initialize_internal_device()              # Initialize training device
        self._initialize_unet_arch()                    # Initialize sequential models
        self._initialize_hyperparameters(hyperparams)   # Initialize the hyperparameters as given

    def forward(self, batch):
        pass

        # The forward action of the DDPM model is ACTUALLY the reverse process of the denoising diffusion setup
        # This is where we take the noise given and attempt to recreate the actual values.
        # So the forward step will generate for us the images that were originally noised from the batch
        # (in theory).  

    def backward(self):
        pass

        # Now in the backward pass is where we update the model based on the loss, please see
        # https://arxiv.org/pdf/2006.11239.pdf, Ho et al. in their description of DDPM models.
        # I use the MSE loss.





