import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

from model.ddpm_trans_modules.trans_block_eca import TransformerBlock_eca


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    # input = F.normalize(input, p=2, dim=1, eps=1e-12)
    # input = (input - torch.min(input)) / (torch.max(input) - torch.min(input))

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b)

# model



class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0, norm_groups=32):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class ResnetBloc_eca(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = nn.Sequential(*[TransformerBlock_eca(dim=int(dim), num_heads=2, ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class Encoder(nn.Module):
    def __init__(
            self,
            in_channel=6,
            inner_channel=32,
            norm_groups=32,
    ):
        super().__init__()

        dim = inner_channel
        time_dim = inner_channel

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, dim, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv2 = nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.PixelUnshuffle(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))

        self.block1 = ResnetBloc_eca(dim=dim, dim_out=dim, time_emb_dim=time_dim, norm_groups=norm_groups,
                                     with_attn=True)
        self.block2 = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)
        self.block3 = ResnetBloc_eca(dim=dim * 2 ** 2, dim_out=dim * 2 ** 2, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)
        self.block4 = ResnetBloc_eca(dim=dim * 2 ** 3, dim_out=dim * 2 ** 3, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)

        self.conv_up3 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 3), (dim * 2 ** 3) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

        self.conv_up2 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 2), (dim * 2 ** 2) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))
        self.conv_up1 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 1), (dim * 2 ** 1) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

        self.conv_cat3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=False)
        self.conv_cat2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=False)

        self.decoder_block3 = ResnetBloc_eca(dim=dim * 2 ** 2, dim_out=dim * 2 ** 2, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)
        self.decoder_block2 = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)
        self.decoder_block1 = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)

    def forward(self, x, t):
        x = self.conv1(x)
        x1 = self.block1(x, t)

        x2 = self.conv2(x1)
        x2 = self.block2(x2, t)

        x3 = self.conv3(x2)
        x3 = self.block3(x3, t)

        x4 = self.conv4(x3)
        x4 = self.block4(x4, t)

        de_level3 = self.conv_up3(x4)
        de_level3 = torch.cat([de_level3, x3], 1)
        de_level3 = self.conv_cat3(de_level3)
        de_level3 = self.decoder_block3(de_level3, t)

        de_level2 = self.conv_up2(de_level3)
        de_level2 = torch.cat([de_level2, x2], 1)
        de_level2 = self.conv_cat2(de_level2)
        de_level2 = self.decoder_block2(de_level2, t)

        de_level1 = self.conv_up1(de_level2)
        de_level1 = torch.cat([de_level1, x1], 1)
        mid_feat = self.decoder_block1(de_level1, t)

        return mid_feat, de_level2

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.eps = 1e-5

    def forward(self, x, y):
        mean_x, mean_y = torch.mean(x, dim=(2, 3), keepdim=True), torch.mean(y, dim=(2, 3), keepdim=True)
        std_x, std_y = torch.std(x, dim=(2, 3), keepdim=True) + self.eps, torch.std(y, dim=(2, 3), keepdim=True) + self.eps
        return std_y * (x - mean_x) / std_x + mean_y

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None

        dim = inner_channel

        self.encoder_water = Encoder(in_channel=in_channel, inner_channel=inner_channel, norm_groups=norm_groups)

        self.refine = ResnetBloc_eca(dim=dim*2**1, dim_out=dim*2**1, time_emb_dim=time_dim, norm_groups=norm_groups, with_attn=True)
        self.de_predict = nn.Sequential(nn.Conv2d(dim * 2 ** 1, out_channel, kernel_size=1, stride=1))


    def forward(self, x, time):
        # print(time.shape)
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        mid_feat, x1 = self.encoder_water(x, t)
        # mid_feat_air, x1_air = self.encoder_air(x_air, t)

        mid_feat2 = self.refine(mid_feat, t)

        return self.de_predict(mid_feat2)

if __name__ == '__main__':

    img = torch.zeros(2, 3, 128, 128)
    time = torch.tensor([1, 2])
    model = UNet(inner_channel=48, norm_groups=24, in_channel=3)
    output, a = model(img, img, time)
    # output = model2(img)
    print(output.shape)
    print(a.shape)
    # print(b)
    # total = sum([param.nelement() for param in model.parameters()])
    # print('parameter: %.2fM' % (total / 1e6))