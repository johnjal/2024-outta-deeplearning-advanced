import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.models.vgg as vgg

# 모델의 층을 초기화 시킬 때 쓴 코드로, 필요하지 않으시다면 사용하지 않으셔도 됩니다.
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2

    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]

    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt

    return torch.from_numpy(weight).float()


class Res_block(nn.Module):
    def __init__(self, channels):
        super(Res_block, self).__init__()
        self.relu = nn.ReLU()
        self.CBR = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, 
                                           kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.BatchNorm2d(num_features=channels),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, 
                                           kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(num_features=channels))
        
    def forward(self, x):
        y = self.CBR(x)
        y = x + y
        return self.relu(y)
        

class segmentation_model(nn.Module):
    def __init__(self, n_class):
        super(segmentation_model, self).__init__()
        # [1] 빈칸을 작성하시오.
        
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=1, bias= True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]
            result = nn.Sequential(*layers)

            return result
        
        
        self.enc1_1 = CBR2d(in_channels=3, out_channels=64, kernel_size=7, padding=3)
        self.enc1_1_dilation = CBR2d(in_channels=3, out_channels=64, kernel_size=4, dilation=2, padding=3)
        self.enc1_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool1 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

        self.enc2_1 = Res_block(channels=128)
        self.enc2_2 = Res_block(channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = Res_block(channels=256)
        self.enc3_3 = Res_block(channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = Res_block(channels=512)
        self.enc4_3 = Res_block(channels=512)
        self.enc4_4 = Res_block(channels=512)
        self.enc4_5 = Res_block(channels=512)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
        
        self.enc5_2 = Res_block(channels=1024)
        
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)
        
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=128)

        self.unpool1 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_3 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec1_2 = CBR2d(in_channels=128, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.out = nn.Conv2d(in_channels=64, out_channels=n_class, 
                            kernel_size=1, stride=1, padding=0, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        # [2] 빈칸을 작성하시오.
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
        

    def forward(self, x):
        # [3] 빈칸을 작성하시오.
            
        height = x.size(2)
        width = x.size(3)
        
        output_padding_1 = (height % 2, width % 2)
        height = height // 2 + output_padding_1[0]
        width = width // 2 + output_padding_1[1]
        
        output_padding_2 = (height % 2, width % 2)
        height = height // 2 + output_padding_2[0]
        width = width // 2 + output_padding_2[1]
        
        output_padding_3 = (height % 2, width % 2)
        height = height // 2 + output_padding_3[0]
        width = width // 2 + output_padding_3[1]
        
        output_padding_4 = (height % 2, width % 2)
        
        enc1 = self.enc1_2(torch.cat((self.enc1_1(x), self.enc1_1_dilation(x)), dim=1))
        enc2 = self.enc2_2(self.enc2_1(self.pool1(enc1)))
        enc3 = self.enc3_3(self.enc3_2(self.enc3_1(self.pool2(enc2))))
        enc4 = self.enc4_5(self.enc4_4(self.enc4_3(self.enc4_2(self.enc4_1(self.pool3(enc3))))))
        enc5 = self.enc5_2(self.enc5_1(self.pool4(enc4)))
        
        dec5 = self.unpool4(self.dec5_1(enc5))
        dec5 = dec5[:, :, :dec5.size(2) - output_padding_4[0], :dec5.size(3) - output_padding_4[1]]

        dec4 = self.unpool3(self.dec4_1(self.dec4_2(torch.cat((dec5, enc4), dim=1))))
        dec4 = dec4[:, :, :dec4.size(2) - output_padding_3[0], :dec4.size(3) - output_padding_3[1]]

        dec3 = self.unpool2(self.dec3_1(self.dec3_2(torch.cat((dec4, enc3), dim=1))))
        dec3 = dec3[:, :, :dec3.size(2) - output_padding_2[0], :dec3.size(3) - output_padding_2[1]]

        dec2 = self.unpool1(self.dec2_1(self.dec2_2(torch.cat((dec3, enc2), dim=1))))
        dec2 = dec2[:, :, :dec2.size(2) - output_padding_1[0], :dec2.size(3) - output_padding_1[1]]

        dec1 = self.dec1_1(self.dec1_2(self.dec1_3(torch.cat((dec2, enc1), dim=1))))

        return self.out(dec1)
