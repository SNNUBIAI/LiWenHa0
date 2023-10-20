import torch
import torch.nn.functional as F
import torch.nn as nn
#类似restnet通道融合
in_channels = 2
out_channels = 2
kernel_size = 3
w = 9
h = 9
x = torch.ones(1,in_channels,w,w)
conv_2d = nn.Conv2d(in_channels,out_channels,kernel_size,padding="same")
conv_2d_posintwise = nn.Conv2d(in_channels,out_channels,1)
result1 = conv_2d(x) + conv_2d_posintwise(x) + x
#print(result)
#算子融合如将三个卷积块写为3*3卷积

pointwise_to_conv_weight = F.pad(conv_2d_posintwise.weight,[1,1,1,1,0,0,0,0]) #2*2*1*1 2*2*3*3
conv_2d_for_posintwise = nn.Conv2d(in_channels,out_channels,kernel_size,padding="same")
conv_2d_for_posintwise.weight = nn.Parameter(pointwise_to_conv_weight)
conv_2d_for_posintwise.bias = conv_2d_posintwise.bias
zeros = torch.unsqueeze(torch.zeros(kernel_size,kernel_size),0)
stars = torch.unsqueeze(F.pad(torch.ones(1,1),[1,1,1,1]),0)
#print(zeros)
#print(stars)
stars_zeros = torch.unsqueeze(torch.cat([stars, zeros]),0)
zeros_stars = torch.unsqueeze(torch.cat([zeros, stars]),0)
indentity_to_weight = torch.cat([stars_zeros, zeros_stars],0)
indentity_to_bias = torch.zeros([out_channels])
conv_2d_identity = nn.Conv2d(in_channels,out_channels,kernel_size,padding="same")
conv_2d_identity.weight = nn.Parameter(indentity_to_weight)
conv_2d_identity.bias = nn.Parameter(indentity_to_bias)
result2 = conv_2d(x) + conv_2d_for_posintwise(x) + conv_2d_identity(x)
print(result2)
print(torch.all(torch.isclose(result1,result2)))
conv_2d_fusion = nn.Conv2d(in_channels,out_channels,kernel_size,padding="same")
conv_2d_fusion.weight = nn.Parameter(conv_2d.weight.data + conv_2d_posintwise.weight.data + conv_2d_identity.weight.data)
conv_2d_fusion.bias = nn.Parameter(conv_2d.weight.bias + conv_2d_posintwise.bias.data + conv_2d_identity.bias.data)
result3 = conv_2d_fusion(x)
print(result3)
