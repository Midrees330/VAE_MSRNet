import torch.nn.functional as F
import math
import torch.nn as nn
import torch

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class Cvi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv.apply(weights_init('gaussian'))

        if after == 'BN':
            self.after = nn.InstanceNorm2d(out_channels)
        elif after == 'Tanh':
            self.after = torch.tanh
        elif after == 'sigmoid':
            self.after = torch.sigmoid

        if before == 'ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before == 'LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x


class CvTi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(CvTi, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                        output_padding=0, bias=bias)  # Specified output_padding
        self.conv.apply(weights_init('gaussian'))

        if after == 'BN':
            self.after = nn.InstanceNorm2d(out_channels)
        elif after == 'Tanh':
            self.after = torch.tanh
        elif after == 'sigmoid':
            self.after = torch.sigmoid

        if before == 'ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before == 'LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x
    
class VAE(nn.Module):
    def __init__(self, input_dim=512, latent_dim=128): 
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Assuming input is normalized to [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        # Encode
        latent_params = self.encoder(x)
        mu, logvar = latent_params[:, :self.latent_dim], latent_params[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        # Decode
        recon_x = self.decoder(z)
        recon_x = recon_x.view(-1, self.input_dim, 1, 1)
        return recon_x, mu, logvar


# Update Encoder and Decoder classes to incorporate VAE
class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):  #before 64
        super(Encoder, self).__init__()
        self.Cv0 = Cvi(input_channels, 64)
        self.Cv1 = Cvi(64, 128, before='LReLU', after='BN')
        self.Cv2 = Cvi(128, 256, before='LReLU', after='BN')
        self.Cv3 = Cvi(256, 512, before='LReLU', after='BN')
        self.Cv4 = Cvi(512, 512, before='LReLU', after='BN')
        self.Cv5 = Cvi(512, 512, before='LReLU')
        self.vae = VAE(512, latent_dim)
        
        self.x5 = None

    def forward(self, input):
        # Encoder
        x0 = self.Cv0(input)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        x4_1 = self.Cv4(x3)
        x4_2 = self.Cv4(x4_1)
        x4_3 = self.Cv4(x4_2)
        x5 = self.Cv5(x4_3)
       
        recon_x, mu, logvar = self.vae(x5)
        z = self.vae.reparameterize(mu, logvar)
        
        self.x5 = x5

        feature_dic = {
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4_1": x4_1,
            "x4_2": x4_2,
            "x4_3": x4_3,
            "x5": x5,
            "z": z,  # Store the latent space in the feature dictionary
            "mu": mu,
            "logvar": logvar,
            "recon_x": recon_x
        }

        return feature_dic


class Decoder(nn.Module):
    def __init__(self, output_channels=1):
        super(Decoder, self).__init__()
        self.CvT6 = CvTi(512, 512, before='ReLU', after='BN')
        self.CvT7 = CvTi(1024, 512, before='ReLU', after='BN')
        self.CvT8 = CvTi(1024, 256, before='ReLU', after='BN')
        self.CvT9 = CvTi(512, 128, before='ReLU', after='BN')
        self.CvT10 = CvTi(256, 64, before='ReLU', after='BN')
        self.CvT11 = CvTi(128, output_channels, before='ReLU', after='Tanh')

    def forward(self, feature_dic):
        
        # Decoder
        x6 = self.CvT6([feature_dic["x5"], feature_dic["recon_x"]])
        cat1_1 = torch.cat([x6, feature_dic["x4_3"]], dim=1)
        x7_1 = self.CvT7(cat1_1) + x6                 # Adding residual/Skip connection
        cat1_2 = torch.cat([x7_1, feature_dic["x4_2"]], dim=1)
        x7_2 = self.CvT7(cat1_2) + x7_1                   # Adding residual/Skip connection
        cat1_3 = torch.cat([x7_2, feature_dic["x4_1"]], dim=1)
        x7_3 = self.CvT7(cat1_3) + x7_2                   # Adding residual/Skip connection

        cat2 = torch.cat([x7_3, feature_dic["x3"]], dim=1)
        x8 = self.CvT8(cat2)  + x7_3                      # Adding residual/Skip connection

        cat3 = torch.cat([x8, feature_dic["x2"]], dim=1)
        x9 = self.CvT9(cat3) + x8                        # Adding residual/Skip connection

        cat4 = torch.cat([x9, feature_dic["x1"]], dim=1)
        x10 = self.CvT10(cat4) + x9                     # Adding residual/Skip connection

        cat5 = torch.cat([x10, feature_dic["x0"]], dim=1)
        out = self.CvT11(cat5) + x10  
        

        return out


class JointDecoder(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(JointDecoder, self).__init__()
        self.CvT6 = CvTi(2048, 1024, before='ReLU', after='BN')
        self.CvT7 = CvTi(2048, 1024, before='ReLU', after='BN')
        self.CvT8 = CvTi(2048, 512, before='ReLU', after='BN')
        self.CvT9 = CvTi(1024, 256, before='ReLU', after='BN')
        self.CvT10 = CvTi(512, 128, before='ReLU', after='BN')
        self.CvT11 = CvTi(256, output_channels, before='ReLU', after='Tanh')

    def forward(self, f_dic1, f_dic2):
        #decoder
        
        cat_0 = torch.cat([f_dic1["x5"], f_dic2["x5"], f_dic1["recon_x"], f_dic2["recon_x"]], dim=1)

        x6 = self.CvT6(cat_0) #channel=1024
        cat1_1 = torch.cat([x6, f_dic1["x4_3"], f_dic2["x4_3"]], dim=1)#cat(1024,1024)
        x7_1 = self.CvT7(cat1_1)
        cat1_2 = torch.cat([x7_1, f_dic1["x4_2"], f_dic2["x4_2"]], dim=1)#cat(1024, 1024)
        x7_2 = self.CvT7(cat1_2)
        cat1_3 = torch.cat([x7_2, f_dic1["x4_1"], f_dic2["x4_1"]], dim=1)#cat(1024, 1024)
        x7_3 = self.CvT7(cat1_3)

        cat2 = torch.cat([x7_3, f_dic1["x3"], f_dic2["x3"]], dim=1) #cat(1024, 1024)
        x8 = self.CvT8(cat2)

        cat3 = torch.cat([x8, f_dic1["x2"], f_dic2["x2"]], dim=1) #cat(512, 512)
        x9 = self.CvT9(cat3)

        cat4 = torch.cat([x9, f_dic1["x1"], f_dic2["x1"]], dim=1) #cat(256, 256)
        x10 = self.CvT10(cat4)

        cat5 = torch.cat([x10, f_dic1["x0"], f_dic2["x0"]], dim=1)
        out = self.CvT11(cat5)

        return out



class VAE_MSRNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(VAE_MSRNet, self).__init__()
        #self.vae = VAE(512, latent_dim)
        #different feature extractors: Es and Esf
        self.domain1_encoder = Encoder(input_channels)
        self.domain2_encoder = Encoder(input_channels)

        #common feature extractors: Ec
        self.general_encoder = Encoder(input_channels)

        #self-reconstruction extractors: Js and Jsf
        self.joint_decoder1 = JointDecoder(output_channels)
        self.joint_decoder2 = JointDecoder(output_channels)

        #shadow removal joint decoder Js2sf
        self.joint_decoderT = JointDecoder(output_channels)

        # zero placeholder for partly reconstructing in test_pair function
        self.placeholder = None


    def forward(self, input, GT):
        feature_dic1 = self.domain1_encoder(input)
        feature_dic2 = self.domain2_encoder(GT)

        general_dic1 = self.general_encoder(input)
        general_dic2 = self.general_encoder(GT)
        
        
        # shadow images reconstruction
        reconstruct_input = self.joint_decoder1(feature_dic1, general_dic1)
        # shadow free images reconstruction
        reconstruct_gt = self.joint_decoder2(feature_dic2, general_dic2)

        # shadow images to shadow free images
        reconstruct_tf = self.joint_decoderT(feature_dic1, general_dic1)

        return reconstruct_input, reconstruct_tf, reconstruct_gt

    def test(self, input):
        # shadow removal only function
        feature_dic1 = self.domain1_encoder(input)
        general_dic1 = self.general_encoder(input)
        
        
        reconstruct_input = self.joint_decoderT(feature_dic1, general_dic1)

        return reconstruct_input

    def test_pair(self, input):
        # extract the shadow images' common and the different features
        feature_dic1 = self.domain1_encoder(input)
        general_dic1 = self.general_encoder(input)

        if self.placeholder is None or self.placeholder["x1"].size(0) != feature_dic1["x1"].size(0):
            # create a zero placeholder with the same shape with common or different features
            self.placeholder = {}
            for key in feature_dic1.keys():
                self.placeholder[key] = torch.zeros(feature_dic1[key].shape, requires_grad=False) \
                    .to(torch.device(feature_dic1["x1"].device))
        
        
        rec_by_tg1 = self.joint_decoderT(self.placeholder, general_dic1)

        rec_by_td1 = self.joint_decoderT(feature_dic1, self.placeholder)

        reconstruct_tf = self.joint_decoderT(feature_dic1, general_dic1)

        return reconstruct_tf, rec_by_tg1, rec_by_td1


class Discriminator(nn.Module):
    def __init__(self, input_channels=4):
        super(Discriminator, self).__init__()

        self.Cv0 = Cvi(input_channels, 64)

        self.Cv1 = Cvi(64, 128, before='LReLU', after='BN')

        self.Cv2 = Cvi(128, 256, before='LReLU', after='BN')

        self.Cv3 = Cvi(256, 512, before='LReLU', after='BN')

        self.Cv4 = Cvi(512, 1, before='LReLU', after='sigmoid')

    def forward(self, input):
        x0 = self.Cv0(input)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        out = self.Cv4(x3)

        return out

if __name__ == '__main__':
    #BCHW
    size = (3, 3, 256, 256)
    input1 = torch.ones(size)
    input2 = torch.ones(size)
    l1 = nn.L1Loss()
    #input.requires_grad = True

    #convolution test
    # conv = Cvi(3, 3)
    # conv2 = Cvi(3, 3, before='ReLU', after='BN')
    # output = conv(input)
    # output2 = conv2(output)
    # print(output.shape)
    # print(output2.shape)
    # loss = l1(output, torch.randn(3, 3, 128, 128))
    # loss.backward()
    # print(loss.item())
    #
    # convT = CvTi(3, 3)
    # outputT = convT(output)
    # print(outputT.shape)


    #Generator test
    # model = RemoveGANVisualization()
    # out = model(input1, input2)
    # print()
    # print(out.shape)
    #loss = l1(output_input, torch.randn(3, 3, 256, 256))
    #loss.backward()
    #print(loss.item())
    #
    # #Discriminator test
    size = (3, 3, 256, 256)
    input = torch.ones(size)
    # l1 = nn.L1Loss()
    # input.requires_grad = True
    # model = Discriminator()
    # output = model(input)
    # print(output.shape)

    #domainTransfer test
    # size = (3, 512, 1, 1)
    # input = torch.ones(size)
    # model = DomainTransfer()
    # output = model(input)
    # print(output.shape)

    #model = NewEncoder()
    #output = model(input)
    #print(output.shape)