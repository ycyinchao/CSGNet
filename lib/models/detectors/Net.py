from lib.RCAB import *
from lib.models.backbones.pvtv2 import pvt_v2_b4


# Channel Reduce
class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel, RFB=False):
        super(Reduction, self).__init__()
        # self.dyConv = Dynamic_conv2d(in_channel,out_channel,3,padding = 1)
        if (RFB):
            self.reduce = nn.Sequential(
                RFB_modified(in_channel, out_channel),
            )
        else:
            self.reduce = nn.Sequential(
                BasicConv2d(in_channel, out_channel, 1),
            )

    def forward(self, x):
        return self.reduce(x)


#
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7,channle=64):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.adativeSA = nn.Conv2d(channle, 1, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        adaptive_out = self.adativeSA(x)
        x = torch.cat([avg_out, max_out,adaptive_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SRM(nn.Module):
    def __init__(self, channel):
        super(SRM, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.semantic_fusion = nn.Conv2d(2*channel, channel, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, semantic_features, vision_features):
        # 低级视觉特征细化
        low_res_features = F.relu(self.conv1(semantic_features))
        high_res_semantics = F.relu(self.conv2(vision_features))

        # 融合高级语义
        fusion = torch.cat([low_res_features, high_res_semantics], dim=1)
        fusion = self.semantic_fusion(fusion)

        # 语义引导的特征增强
        attention_map = self.attention(fusion)
        enhanced_features = fusion * attention_map + fusion



        return enhanced_features



class CLIM(nn.Module):
    def __init__(self, channel=64):
        super(CLIM, self).__init__()

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.fuse_conv = nn.Conv2d(2 * channel, channel, kernel_size=1)

        self.spatial_attention = SpatialAttention()  # 添加空间注意力模块

    def forward(self, fg_high, fg_low):#up表示高分辨率特征图，down表示低分辨率特征图
        # 特征映射通过卷积处理
        x1 = F.relu(self.conv1(fg_high))
        x2 = F.relu(self.conv2(fg_low))

        # 融合特征
        fused_features = torch.cat([x1, x2], dim=1)
        fused_features = self.fuse_conv(fused_features)

        # 添加空间注意力
        spatial_attention_map = self.spatial_attention(fused_features)  # 计算空间注意力
        refined_features_with_spatial_attention = fused_features * spatial_attention_map + fused_features  # 应用空间注意力

        return refined_features_with_spatial_attention



class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64, imagenet_pretrained=True):
        super(Network, self).__init__()

        #  ---- PVTv2_B4 Backbone ----

        self.bkbone = pvt_v2_b4()  # [64, 128, 320, 512]
        # 获取预训练的参数
        save_model = torch.load('../weights/pvt_v2_b4.pth')
        # 获取当前模型的参数
        model_dict = self.bkbone.state_dict()
        # 加载部分能用的参数
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # 更新现有的model_dict
        model_dict.update(state_dict)
        # 加载真正需要的state_dict
        self.bkbone.load_state_dict(model_dict)
        enc_channels = [64, 128, 320, 512]

        self.reduce_1 = Reduction(enc_channels[0], channel, RFB=False)
        self.reduce_2 = Reduction(enc_channels[1], channel, RFB=False)
        self.reduce_3 = Reduction(enc_channels[2], channel, RFB=False)
        self.reduce_4 = Reduction(enc_channels[3], channel, RFB=False)

        self.clim1 = CLIM(channel)
        self.clim2 = CLIM(channel)
        self.clim3 = CLIM(channel)

        self.srm1 = SRM(channel)

        self.srm3 = SRM(channel)

        self.srm5 = SRM(channel)


        self.pre_fg0 = nn.Conv2d(channel,1,1)
        self.pre_fg1 = nn.Conv2d(channel, 1, 1)
        self.pre_fg2 = nn.Conv2d(channel, 1, 1)
        self.pre_fg3 = nn.Conv2d(channel, 1, 1)
        self.pre_fg4 = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        # Feature Extraction
        shape = x.size()[2:]


        #  ---- PVTv2_B4 Backbone ----
        x1, x2, x3, x4 = self.bkbone(x)

        # Channel Reduce
        x1_fg = self.reduce_1(x1)
        x1_low_fg = x1_fg.clone()
        x2_fg = self.reduce_2(x2)
        x3_fg = self.reduce_3(x3)
        x4_fg = self.reduce_4(x4)

    # stage 1
        # SGR in fg
        # sgr_fg = self.sgr1(x4_fg,x1_fg)

        # BFRE in fg
        x4_fg = F.interpolate(x4_fg, size=x3_fg.size()[2:], mode='bilinear')
        fg_1 = self.clim1(x3_fg, x4_fg)  # B*C*24*24
    # stage 2
        # SGR in fg
        fg_1 = F.interpolate(fg_1, size=x1_low_fg.size()[2:], mode='bilinear')
        sgr_fg = self.srm1(fg_1,x1_low_fg)

        # BFRE in fg
        fg_1 = F.interpolate(fg_1, size=x2_fg.size()[2:], mode='bilinear')
        fg_2 = self.clim2(x2_fg, fg_1)  # B*C*48*48
    # stage 3
        # SGR in fg
        fg_2 = F.interpolate(fg_2, size=sgr_fg.size()[2:], mode='bilinear')
        sgr_fg = self.srm3(fg_2,sgr_fg)

        # BFRE in fg
        fg_2 = F.interpolate(fg_2, size=x1_fg.size()[2:], mode='bilinear')
        fg_3 = self.clim3(sgr_fg, fg_2)

        fg_3 = F.interpolate(fg_3, size=sgr_fg.size()[2:], mode='bilinear')
        sgr_fg = self.srm5(fg_3,sgr_fg)

        pred_fg0 = F.interpolate(self.pre_fg0(x4_fg),size=shape,mode='bilinear')
        pred_fg1 = F.interpolate(self.pre_fg1(fg_1), size=shape, mode='bilinear')
        pred_fg2 = F.interpolate(self.pre_fg2(fg_2), size=shape, mode='bilinear')
        pred_fg3 = F.interpolate(self.pre_fg3(fg_3), size=shape, mode='bilinear')  # final pred
        pred_fg4 = F.interpolate(self.pre_fg4(sgr_fg), size=shape, mode='bilinear')


        return pred_fg0,pred_fg1, pred_fg2, pred_fg3, pred_fg4

if __name__ == '__main__':
    import numpy as np
    from time import time

    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 384, 384)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)
