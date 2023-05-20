'''
File to store implementation of various network heads and models.
This stores all needed architectures.

@Author: George Witt
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

'''
Embeddings and doublets
'''

class Embedding_Model(nn.Module):
    ''' 
    The embedding model class encapsulates the embedding layer brought in at the ends of the 
    UNET architecture.
    '''

    def __init__(self, embedding_dimensions, out_channels):
        super(Embedding_Model, self).__init__()

        self.model = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dimensions, out_channels)
        )

    def forward(self, x):
        return self.model(x)

class Conv_Doublet(nn.Module):
    '''
    The conv doublet class encapsulates the functionality of having the doublet layers in the UNET diagram, while
    allowing for possible residual connections between the doublets. (the residual connections are the main reason
    for creating a separate class).
    '''

    def __init__(self, in_channels, middle_channels, out_channels, kern_size=3, padding_size=1, padding_mode='zeros', bias=False, residual=False):
        super(Conv_Doublet, self).__init__()

        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.padding_size = padding_size
        self.padding_mode = padding_mode
        self.bias = bias
        self.residual = residual

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=middle_channels, kernel_size=kern_size, padding=padding_size, padding_mode=padding_mode, bias=bias),
            nn.GroupNorm(1, middle_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=middle_channels, out_channels=out_channels, kernel_size=kern_size, padding=padding_size, padding_mode=padding_mode, bias=bias),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(self.model(x) + x) # Add in res. connection from input to the OUTPUT of the BLOCK, not output of Conv2d.  
        else:
            return self.model(x)

'''
Attention
'''

class SelfAttention(nn.Module):
    '''
    The self attention head is a standard self-attention head as described in
    see Vaswani et al in 'Attention is all you need', the famous paper. 

    Please see https://arxiv.org/pdf/1706.03762.pdf
    
    Further details are given in the forward call.
    No backward call is provided, gradients are tracked.
    '''

    def __init__(self, channels, size, heads=4):
        super(SelfAttention, self).__init__()
        self.multi_head = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.size = size
        self.channels = channels

        # for self attent we want to layer norm over our channels, combine these with GELU. 
        self.layer_norm = nn.LayerNorm([channels])
        self.self_attent = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, inc):
        #print("SELF_ATTENT INCOMING:", inc.shape)

        # Flatten the two image axes, and swap with the channels so that 
        # we have data of the form (B, img_axis * img_axis, C)
        inc = inc.view(-1, self.channels, self.size*self.size).swapaxes(1, 2)
        self_attent = self.self_attent(inc)

        # Apply attention over the normalized and linearly combined channel sets.
        attention_value, _ = self.multi_head(self_attent, self_attent, self_attent)

        # Add residual connection
        attention_value = attention_value + inc
        attention_value = self.self_attent(attention_value) + attention_value

        # Flip back
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

'''
Sampling / Convolution
'''

class BottomConv(nn.Module):
    '''
    The Bottom convolutional layers are the same as in the Up/Down
    sample blocks, just modified without the up/down sampling.
    '''

    def __init__(self, in_size, out_size):
        super(BottomConv, self).__init__()

        middle_size = in_size * 2
        mod1 = Conv_Doublet(in_size, middle_size, middle_size)
        mod2 = Conv_Doublet(middle_size, middle_size, middle_size)
        mod3 = Conv_Doublet(middle_size, out_size, out_size)
        self.model = nn.Sequential(
            mod1,
            mod2,
            mod3
        )

    def forward(self, x):
        return self.model(x)

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
    def __init__(self, in_channel_size, out_channel_size, middle_channel=None, residual=False, embedding_dimensions=100):
        super(UpSample, self).__init__()

        self.conv_residual = residual

        if middle_channel is None:
            middle_channel = out_channel_size

        conv_doublet1 = Conv_Doublet(in_channel_size, in_channel_size, in_channel_size, residual=True)
        conv_doublet2 = Conv_Doublet(in_channel_size, in_channel_size // 2, out_channel_size)

        self.comb_doub = nn.Sequential(
            conv_doublet1,
            conv_doublet2
        )

        self.emb_model = Embedding_Model(embedding_dimensions, out_channel_size)
        self.up_sample = nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear')

    def forward(self, x, residual_x, t):
        #print("UP INCOMING:", x.shape, residual_x.shape, t.shape)

        # Upsampling must first actually upsample the given x
        x = self.up_sample(x)
        #print("UPSAMPELD X", x.shape)

        # Don't forget the residual x coming from across layers
        x = torch.cat([residual_x, x], dim=1)
        #print("CATTED SHAPE:", x.shape)
        # Now we can finally pass through the convs
        x = self.comb_doub(x)
        #print("CONVED shape", x.shape)
        emb = self.emb_model(t)
        #print("EMBEDDING SHAPE", emb.shape)

        # Now in order to add the embedding we need to expand it
        # properly across each L, L set.
        # Res is now (B, NewC, L, L) where L comes from the convolution.
        # So now since the embedding is of the shape (1, NewC), we need to 
        # add 2 new dimensions for L, L.. and then repeat the embedding over
        # the the L, L sets.
        
        # Add the dimensions
        emb = emb[:, :, None, None]

        # Now we're gonna repeat. The first two terms are 1 so that we don't repeat
        # over the B or newC dimensions. Then we repeat across the L and L dimensions.
        # Recall we assumed Rows = Columns earlier. This needs to be modified for R != C.
        emb = emb.repeat(1, 1, x.shape[-1], x.shape[-1])

        return x + emb

class DownSample(nn.Module):
    '''
    The DownSample block consists of the following:
        2 coupled convolutional layers
        A residual down sampling connection to decrease matrix size.

    Further details are given in the forward call.
    No backward call is provided, gradients are tracked.
    '''
    
    def __init__(self, in_channel_size, out_channel_size, residual=False, embedding_dimensions=512):
        super(DownSample, self).__init__()
        
        self.conv_residual = residual

        conv_doublet1 = Conv_Doublet(in_channel_size, in_channel_size, in_channel_size)
        conv_doublet2 = Conv_Doublet(in_channel_size, out_channel_size, out_channel_size)

        self.comb_doub = nn.Sequential(
            nn.MaxPool2d(2), 
            conv_doublet1,
            conv_doublet2
        )
        
        self.emb_model = Embedding_Model(embedding_dimensions, out_channel_size)

    def forward(self, x, t):
        res = self.comb_doub(x)
        emb = self.emb_model(t)

        #print("DOWN RES:", res.shape)
        #print("DOWN EMB:", emb.shape)

        # Now in order to add the embedding we need to expand it
        # properly across each L, L set.
        # Res is now (B, NewC, L, L) where L comes from the convolution.
        # So now since the embedding is of the shape (1, NewC), we need to 
        # add 2 new dimensions for L, L.. and then repeat the embedding over
        # the the L, L sets.
        
        # Add the dimensions
        emb = emb[:, :, None, None]

        # Now we're gonna repeat. The first two terms are 1 so that we don't repeat
        # over the B or newC dimensions. Then we repeat across the L and L dimensions.
        # Recall we assumed Rows = Columns earlier. This needs to be modified for R != C.
        emb = emb.repeat(1, 1, res.shape[-1], res.shape[-1])

        return res + emb

'''
** DDPM **
'''

class DDPM(nn.Module):

    def _initialize_internal_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _initialize_unet_arch(self, n_channels):
        '''
        The UNET architecture I employ here is very similar to that in (https://arxiv.org/pdf/1505.04597.pdf), Ronneberger et al.
        I make one small modification of including self attention between downsampling.
        This modification is described in (https://arxiv.org/pdf/1503.03585.pdf), Ho et al.

        Note I combine the downsampling sets and upsampling sets into separate modules.

        The architecture consists of an input conv, and a 3 'down' blocks with residual connections. Then
        1 bottom block. Then 3 'up' blocks with the incoming residual connections. Then 
        a final output convolution
        '''

        # Incoming data is of the form (B, C, IMGR, IMGC)
        # Assume for simplicity that IMGR == IMGC (for this project)
        # So image data is of the form (B, C, R, R).
        # Pass the 

        # Downward direction
        self.start = Conv_Doublet(n_channels, 64, 64)
        self.d1 = DownSample(64, 128)
        self.s1 = SelfAttention(128, 14)
        self.d2 = DownSample(128, 256)
        self.s2 = SelfAttention(256, 7)
        #self.d3 = DownSample(256, 256)
        #self.s3 = SelfAttention(256, 3)

        # Bottom layers
        self.bottom = BottomConv(256, 512)

        # Upwards
        self.u1 = UpSample(640, 256, embedding_dimensions=512)
        self.s5 = SelfAttention(256, 14)
        self.u2 = UpSample(320, 128, embedding_dimensions=512)
        self.s6 = SelfAttention(128, 28)
        self.out = nn.Conv2d(128, n_channels, 1)

        self.to(self.device)

    def _initialize_hyperparameters(self, hyperparams):
        '''
        Hyperparameter loaded in... certain key hyperparameters are required.
        '''
        self.hypers = hyperparams
        
        try:
            self.optimizer = optim.Adam(self.parameters(), lr=hyperparams['lr'])
        except KeyError as e:
            #print("HYPERPARAMS requires LR as parameter")
            raise KeyError("Could not find mapping to learning rate")

    def pos_encoding(self, t, channels):
        # According to [https://arxiv.org/pdf/2006.11239.pdf]
        a_enc = torch.sin(t.repeat(1, channels // 2) * 1.0 / (10000** (torch.arange(0, channels, 2, device=self.device).float() / channels)))
        b_enc = torch.cos(t.repeat(1, channels // 2) * 1.0 / (10000** (torch.arange(0, channels, 2, device=self.device).float() / channels)))
        return torch.cat([a_enc, b_enc], dim=-1)

    def __init__(self, channels, time_dimensions, hyperparams):
        '''
        Initialization of DDPM model
        @channels: The number of channels of each image.
        @time_dimensions: The time (schedule) length
        @hyperparams: Dictionary of necessary hyperparameters
        '''
        super(DDPM, self).__init__()

        self.time_dimensions = time_dimensions

        self._initialize_internal_device()              # Initialize training device
        self._initialize_unet_arch(channels)            # Initialize sequential models
        self._initialize_hyperparameters(hyperparams)   # Initialize the hyperparameters as given

    def forward(self, x, t):
        # The forward action of the DDPM model is ACTUALLY the reverse process of the denoising diffusion setup
        # This is where we take the noise given and attempt to recreate the actual values.
        # So the forward step will generate for us the images that were originally noised from the batch
        # (in theory).  

        t = torch.tensor(t).to(self.device)
        x = x.to(self.device)
        
        t = self.pos_encoding(t, self.time_dimensions).to(self.device)
        
        #print("INPUT:", x.shape)
        #print("TIME:", t.shape)
        # ADJUST THIS!
        x1 = self.start(x)
        #print("X1", x1.shape)
        x2 = self.d1(x1, t)
        #print("X2 down:", x2.shape)
        x2 = self.s1(x2)
        #print("X2:", x2.shape)
        x3 = self.d2(x2, t)
        #print("X3 down:", x3.shape)
        x3 = self.s2(x3)
        #print("X3:", x3.shape)
        #x4 = self.d3(x3, t)
        ##print("X4 down:", x4.shape)
        #x4 = self.s3(x4)
        ##print("X4:", x4.shape)
        x4 = x3

        x5 = self.bottom(x4)
        #print("X5:", x5.shape)


        x = self.u1(x5, x2, t)
        #print("U1:", x.shape)
        x = self.s5(x)
        #print("U1s:", x.shape)
        x = self.u2(x, x1, t)
        #print("U2:", x.shape)
        x = self.s6(x)
        #print("U2s:", x.shape)
        #x = self.u3(x, x1, t)
        ##print("U3:", x.shape)
        #x = self.s7(x)
        ##print("U3s:", x.shape)
        output = self.out(x)

        return output

    def backward(self, predicted_noise, true_noise, no_grad=False):
        # Now in the backward pass is where we update the model based on the loss, please see
        # https://arxiv.org/pdf/2006.11239.pdf, Ho et al. in their description of DDPM models.
        # I use the MSE loss.

        # ENSURE both are on the same device
        predicted_noise = predicted_noise.to(self.device)
        true_noise = true_noise.to(self.device)

        loss = F.mse_loss(predicted_noise, true_noise)

        if not no_grad:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.clone().detach().to('cpu')


class ema_trainer:
    '''
    EMA Training as described in the supplemental PDF is a method of combining the parameters of two models
    trained in tandem. After some warm-up period, it begins training both models together by combining their weights.
    '''
    def __init__(self, warm_up, beta):
        '''
        @warm_up: The number of global steps to warm up
        @beta: The combination parameter
        '''

        self.warmup = warm_up
        self.beta = beta

    def step_training(self, base_model, ema_model, step):
        '''
        @base_model: The pytorch module that is the base model for prediction
        @ema_model: The pytorch module that is the ema model used for full prediction (goal of training)
        @step: The current global step
        '''
        if step > self.warmup:
            for ema_param, base_param in zip(ema_model.parameters(), base_model.parameters()):
                if ema_param.data is None:
                    break
                ema_param.data = ema_param.data * self.beta + (1 - self.beta) * base_param.data
        else:
            ema_model.load_state_dict(base_model.state_dict())




