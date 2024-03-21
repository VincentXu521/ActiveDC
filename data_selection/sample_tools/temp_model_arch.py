import math
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
# model have BN
"""
class MyNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyNet, self).__init__()

        # mid_channels = (in_channels + out_channels) // 2
        # mid_channels = int(math.sqrt(in_channels * out_channels))
        mid_channels = out_channels * 2

        # Define BatchNorm layers
        self.input_norm = nn.BatchNorm1d(in_channels)

        # Attention Net
        self.conv_att_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.bn_att_1 = nn.BatchNorm1d(out_channels)
        
        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.bottleneck = nn.Linear(out_channels, mid_channels)
        self.channel_weight = nn.Linear(mid_channels, in_channels)

        # Conv Net
        self.conv_1 = nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=1, padding=0)
        self.batch_norm_1 = nn.BatchNorm1d(mid_channels)

        self.conv_2 = nn.Conv1d(mid_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.batch_norm_2 = nn.BatchNorm1d(out_channels)

        self.conv_tr_1 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.conv_tr_bn_1 = nn.BatchNorm1d(out_channels)

        self.conv_tr_2 = nn.ConvTranspose1d(out_channels, mid_channels, kernel_size=3, stride=1, padding=0)
        self.conv_tr_bn_2 = nn.BatchNorm1d(mid_channels)

        self.recons = nn.Conv1d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_input):
        n_channel = x_input.size(1)  # Get the number of input channels

        # Input normalization
        input_norm = self.input_norm(x_input)

        # Attention Net
        conv_att_1 = self.conv_att_1(input_norm)
        bn_att_1 = F.relu(self.bn_att_1(conv_att_1))

        # Global Pooling
        global_pool = self.global_pool(bn_att_1)
        global_pool = global_pool.view(global_pool.size(0), -1)
        bottleneck = F.relu(self.bottleneck(global_pool))
        channel_weight = torch.sigmoid(self.channel_weight(bottleneck))
        channel_weight_ = channel_weight.view(-1, n_channel, 1)
        reweight_out = channel_weight_ * input_norm

        # Conv Net
        conv_1 = self.conv_1(reweight_out)
        batch_norm_1 = F.relu(self.batch_norm_1(conv_1))

        conv_2 = self.conv_2(batch_norm_1)
        batch_norm_2 = F.relu(self.batch_norm_2(conv_2))

        conv_tr_1 = self.conv_tr_1(batch_norm_2)
        conv_tr_bn_1 = F.relu(self.conv_tr_bn_1(conv_tr_1))

        conv_tr_2 = self.conv_tr_2(conv_tr_bn_1)
        conv_tr_bn_2 = F.relu(self.conv_tr_bn_2(conv_tr_2))

        output = self.recons(conv_tr_bn_2)

        return channel_weight, output

# Instantiate the PyTorch model
# net = MyNet(total_num, sample_num)



"""
# model have no BN
"""
class BSNET_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # mid_channels = (in_channels + out_channels) // 2
        # mid_channels = int(math.sqrt(in_channels * out_channels))
        mid_channels = out_channels * 2

        self.conv1 = nn.Sequential(
        	nn.Conv1d(in_channels, out_channels, 3, 1, 1),
        	nn.ReLU(True))

        self.conv1_1 = nn.Sequential(
        	nn.Conv1d(in_channels, mid_channels, 3, 1),
        	nn.ReLU(True))
        self.conv1_2 = nn.Sequential(
        	nn.Conv1d(mid_channels, out_channels, 3, 1),
        	nn.ReLU(True))
        
        self.deconv1_2 = nn.Sequential(
        	nn.ConvTranspose1d(out_channels, out_channels, 3, 1),
        	nn.ReLU(True))
        self.deconv1_1 = nn.Sequential(
        	nn.ConvTranspose1d(out_channels, mid_channels, 3, 1),
        	nn.ReLU(True))

        self.conv2_1 = nn.Sequential(
        	nn.Conv1d(mid_channels, in_channels, 1, 1),
        	nn.Sigmoid())
        
        self.fc1 = nn.Sequential(
        	nn.Linear(out_channels, mid_channels),
        	nn.ReLU(True))
        self.fc2 = nn.Sequential(
        	nn.Linear(mid_channels, in_channels),
        	nn.Sigmoid())
        
        self.aap = nn.AdaptiveAvgPool1d(1)
        
    def BAM(self, x):
        x = self.conv1(x)
        x = self.aap(x)
        x = x.view(-1, self.out_channels)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 1, self.in_channels)
        x = x.permute(0, 2, 1)
        return x

    def RecNet(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.deconv1_2(x)
        x = self.deconv1_1(x)
        x = self.conv2_1(x)
        return x

    def forward(self, x):
        BRW = self.BAM(x)
        x = x * BRW
        ret = self.RecNet(x)
        return BRW, ret




"""
# Pipeline Model Parallel
"""
class Sample_Attn_RecNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # mid_channels = (in_channels + out_channels) // 2
        # mid_channels = int(math.sqrt(in_channels * out_channels))
        # mid_channels = out_channels * 2
        mid_channels = 100

        self.conv1 = nn.Sequential(
        	nn.Conv1d(in_channels, out_channels, 3, 1, 1),
        	nn.ReLU(True)).to('cuda:0')
        
        self.aap = nn.AdaptiveAvgPool1d(1).to('cuda:0')

        self.fc1 = nn.Sequential(
        	nn.Linear(out_channels, mid_channels),
        	nn.ReLU(True)).to('cuda:0')
        self.fc2 = nn.Sequential(
        	nn.Linear(mid_channels, in_channels),
        	nn.Sigmoid()).to('cuda:0')


        self.conv1_1 = nn.Sequential(
        	nn.Conv1d(in_channels, mid_channels, 3, 1),
        	nn.ReLU(True)).to('cuda:1')
        self.conv1_2 = nn.Sequential(
        	nn.Conv1d(mid_channels, out_channels, 3, 1),
        	nn.ReLU(True)).to('cuda:1')
        
        self.deconv1_2 = nn.Sequential(
        	nn.ConvTranspose1d(out_channels, out_channels, 3, 1),
        	nn.ReLU(True)).to('cuda:1')
        self.deconv1_1 = nn.Sequential(
        	nn.ConvTranspose1d(out_channels, mid_channels, 3, 1),
        	nn.ReLU(True)).to('cuda:1')

        self.conv2_1 = nn.Sequential(
        	nn.Conv1d(mid_channels, in_channels, 1, 1),
        	nn.Sigmoid()).to('cuda:1')
        
        
    def BAM(self, x):
        x = self.conv1(x)
        x = self.aap(x)
        x = x.view(-1, self.out_channels)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 1, self.in_channels)
        x = x.permute(0, 2, 1)
        return x

    def RecNet(self, x):
        x = self.conv1_1(x.to('cuda:1'))
        x = self.conv1_2(x)
        x = self.deconv1_2(x)
        x = self.deconv1_1(x)
        x = self.conv2_1(x)
        return x

    def forward(self, x):
        x = x.to('cuda:0')
        BRW = self.BAM(x)
        x = x * BRW
        ret = self.RecNet(x.to('cuda:1'))
        return BRW, ret.to('cuda:0')


# Instantiate the PyTorch model
# net = Sample_Attn_RecNet(total_num, sample_num)

