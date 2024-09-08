import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.relative_rpe import DynamicRelativePositionBias1D, HopRelativePositionBias, DynamicRelativePositionBias


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=False,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation
                ),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        # print(len(self.branches))
        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, num_point=25):
        super(unit_tcn, self).__init__()
        self.num_point = num_point
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), groups=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x, *args):
        if x.dim() == 3:
            x = rearrange(x, 'n (t v) c -> n c t v', v=self.num_point)
        x = self.bn(self.conv(x))
        return x


def interpolate_A(A, n, mode='linear'):
    assert mode in ['linear', 'repeat']

    if mode == 'linear':
        A = rearrange(A, 'n H W -> H W n')
        A_interpolated = F.interpolate(A, size=(n), mode='linear', align_corners=False)
        out = rearrange(A_interpolated, 'H W n -> n H W')
    else:
        repeats_needed = n // A.shape[0]
        out = A.repeat(repeats_needed, 1, 1)

        # In case num_repeats is not a multiple of the original tensor's first dimension
        remaining_repeats = n % A.shape[0]
        if remaining_repeats > 0:
            out = torch.cat((out, A[:remaining_repeats]), dim=0)

    return out


class MHSA(nn.Module):
    def __init__(
            self, dim_in, dim, A, num_heads=6, qkv_bias=False, qk_scale=None,
            attn_drop=0., proj_drop=0., insert_cls_layer=0, pe=False,
            num_point=25, layer=0, use_hop=True, use_group=True,
            use_group_bias=True, use_outer=True, use_ajacency=False,
            use_hop_bias=False, hops=None, interpolate_mode='repeat',
            **kwargs,
    ):
        super().__init__()
        assert not (use_hop and use_hop_bias), \
            "Both use_hop and use_hop_bias cannot be True at the same time."

        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_point = num_point
        self.layer = layer
        self.use_hop = use_hop
        self.use_hop_bias = use_hop_bias
        self.use_group = use_group
        self.use_group_bias = use_group_bias
        self.use_outer = use_outer
        self.use_ajacency = use_ajacency

        self.kv = nn.Conv2d(dim_in, dim * 2, 1, bias=qkv_bias)
        self.q = nn.Conv2d(dim_in, dim, 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        # self.proj = nn.Conv2d(dim, dim, 1, groups=6)
        self.proj = nn.Conv2d(dim, dim, 1, groups=1)  # Zeyun: change it to 1 to support other dimensions

        self.proj_drop = nn.Dropout(proj_drop)

        if use_hop or use_hop_bias:
            assert hops is not None
            self.hops = torch.tensor(hops).long()

        if use_hop:
            # we multiply the hop positional encoding with query to compute attention bias
            self.rpe = nn.Parameter(torch.zeros((self.hops.max()+1, dim)))
        elif use_hop_bias:
            self.rpb = HopRelativePositionBias(
                num_points=num_point,
                A=A,
                num_heads=num_heads,
                num_frames=None,
                mlp_dim=dim,
            )

        if use_group_bias:
            self.w1 = nn.Parameter(torch.zeros(num_heads, head_dim))

        if use_outer:
            A = A.sum(0)
            self.outer = nn.Parameter(torch.stack([torch.eye(A.shape[-1]) for _ in range(num_heads)], dim=0),
                                      requires_grad=True)
            self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        if use_ajacency:
            assert not use_outer
            A = torch.from_numpy(A).float()
            A_interpolate = interpolate_A(A, num_heads, interpolate_mode)
            self.ajacency = nn.Parameter(A_interpolate, requires_grad=True)
            self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.apply(self._init_weights)
        self.insert_cls_layer = insert_cls_layer

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, e=None):
        N, C, T, V = x.shape
        kv = self.kv(x).reshape(N, 2, self.num_heads, self.dim // self.num_heads, T, V).permute(1, 0, 4, 2, 5, 3)
        k, v = kv[0], kv[1]

        ## n t h v c
        q = self.q(x).reshape(N, self.num_heads, self.dim // self.num_heads, T, V).permute(0, 3, 1, 4, 2)

        attn = q @ k.transpose(-2, -1)

        if self.use_hop:
            pos_emb = self.rpe[self.hops]
            k_r = pos_emb.view(V, V, self.num_heads,
                               self.dim // self.num_heads)
            b = torch.einsum("bthnc, nmhc->bthnm", q, k_r)
            attn += b
        elif self.use_hop_bias:
            hop_bias = self.rpb()
            attn += hop_bias

        if self.use_group:
            e_k = e.reshape(N, self.num_heads, self.dim // self.num_heads, T, V).permute(0, 3, 1, 4, 2)
            c = torch.einsum("bthnc, bthmc->bthnm", q, e_k)
            attn += c

        if self.use_group_bias:
            assert self.use_group
            d = torch.einsum("hc, bthmc->bthm", self.w1, e_k).unsqueeze(-2)
            attn += d

        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.use_outer:
            x = (self.alpha * attn + self.outer) @ v
        elif self.use_ajacency:
            x = (self.alpha * attn + self.ajacency) @ v
        else:
            x = attn @ v

        x = x.transpose(3, 4).reshape(N, T, -1, V).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# using conv2d implementation after dimension permutation
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x = self.fc1(x.transpose(1,2)).transpose(1,2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # x = self.fc2(x.transpose(1,2)).transpose(1,2)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class unit_vit(nn.Module):
    def __init__(
            self, dim_in, dim, A, num_of_heads, add_skip_connection=True,
            qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0,
            norm_layer=nn.LayerNorm, layer=0, insert_cls_layer=0, pe=False,
            num_point=25, use_mlp=False, ff_mult=4,
            use_group=True, use_hop=True, use_group_bias=True, use_outer=True,
            use_ajacency=False, use_hop_bias=False, use_learned_partition='none',
            hops=None, interpolate_mode='repeat', **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim_in)
        self.dim_in = dim_in
        self.dim = dim
        self.add_skip_connection = add_skip_connection
        self.num_point = num_point
        self.attn = MHSA(
            dim_in, dim, A, num_heads=num_of_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            insert_cls_layer=insert_cls_layer, pe=pe, num_point=num_point,
            layer=layer, use_hop=use_hop, use_group=use_group, use_outer=use_outer,
            use_group_bias=use_group_bias, use_ajacency=use_ajacency,
            use_hop_bias=use_hop_bias, hops=hops, interpolate_mode=interpolate_mode,
            **kwargs
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.dim_in != self.dim:
            self.skip_proj = nn.Conv2d(dim_in, dim, (1, 1), padding=(0, 0), bias=False)
        self.pe = pe
        self.use_group = use_group
        if use_group:
            self.pe_proj = nn.Conv2d(dim_in, dim, 1, bias=False)

        self.use_mlp = use_mlp
        if use_mlp:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(dim, dim * ff_mult, drop=drop)

        self.use_learned_partition = use_learned_partition
        if use_learned_partition == 'layerwise':
            self.joint_label = nn.Parameter(torch.eye(A.shape[-1], requires_grad=True))

    def forward(self, x, joint_label=None, groups=None):
        if self.use_group:
            if self.use_learned_partition == 'none':
                # same as in Hyperformer
                z = x @ (joint_label / joint_label.sum(dim=0, keepdim=True))
                z = self.pe_proj(z)
                e = z @ joint_label.t()
            elif self.use_learned_partition == 'layerwise':
                label = self.joint_label.softmax(dim=-1)
                z = x @ (label / label.sum(dim=0, keepdim=True))  # Group Embedding
                z = self.pe_proj(z)  # n d t v
                e = z @ label.t()
            else:
                label = joint_label.softmax(dim=-1)
                z = x @ (label / label.sum(dim=0, keepdim=True))  # Group Embedding
                z = self.pe_proj(z)  # n d t v
                e = z @ label.t()
        else:
            e = None

        if self.add_skip_connection:
            if self.dim_in != self.dim:
                x = self.skip_proj(x) + self.drop_path(
                    self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), e))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), e))
        else:
            x = self.drop_path(self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), e))

        if self.use_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))

        return x


class TCN_ViT_unit(nn.Module):
    def __init__(
            self, in_channels, out_channels, A, stride=1, num_of_heads=6,
            residual=True, kernel_size=5, dilations=[1, 2], pe=False,
            num_point=25, layer=0, use_mlp=False, ff_mult=4, drop_path=0,
            use_group=True, use_hop=True, use_group_bias=True, use_outer=True,
            ues_ajacency=False, use_multiscale=True, use_hop_bias=False,
            use_learned_partition='none', hops=None, interpolate_mode='repeat',
    ):
        super(TCN_ViT_unit, self).__init__()
        self.vit1 = unit_vit(
            in_channels, out_channels, A, add_skip_connection=residual, num_of_heads=num_of_heads,
            pe=pe, num_point=num_point, layer=layer, use_mlp=use_mlp, ff_mult=ff_mult,
            drop_path=drop_path, use_group=use_group, use_hop=use_hop,
            use_group_bias=use_group_bias, use_outer=use_outer, use_ajacency=ues_ajacency,
            use_hop_bias=use_hop_bias, use_learned_partition=use_learned_partition,
            hops=hops, interpolate_mode=interpolate_mode,
        )

        if use_multiscale:
            # redisual=True has worse performance in the end
            self.tcn1 = MultiScale_TemporalConv(
                out_channels, out_channels, kernel_size=kernel_size,
                stride=stride, dilations=dilations, residual=False
            )
        else:
            self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)

        self.act = nn.ReLU(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, joint_label, groups):
        y = self.act(self.tcn1(self.vit1(x, joint_label, groups)) + self.residual(x))
        return y


class CoupledAttention(nn.Module):
    def __init__(self, dim_in, dim, A, num_heads=6, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.,
                 num_points=25, num_frames=64,
                 relational_bias=False, hop_bias=True, hops=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_frames = num_frames
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim_in, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.hop_bias = hop_bias

        if hop_bias:
            # Temporal dynamic relative
            self.temporal_rpb = DynamicRelativePositionBias1D(
                num_heads=num_heads,
                window_size=num_frames,
                mlp_dim=dim,
                num_points=num_points
            )

            # Hop relative positional embedding
            self.hop_rpb = HopRelativePositionBias(
                num_points=num_points,
                A=A,
                num_heads=num_heads,
                num_frames=num_frames,
                mlp_dim=dim,
                hops=hops,
            )
        else:
            # temporal & joint-level relative
            self.rpb = DynamicRelativePositionBias(
                num_heads=num_heads,
                window_size=(num_frames, num_points),
                mlp_dim=dim,
            )

        # relational bias, we choose to initialize it with ajacency matrix
        self.relational_bias = relational_bias
        if relational_bias:
            A = A.sum(0)
            A /= A.sum(axis=-1, keepdims=True)
            # A /= num_frames  # because we flatten it later in the code t v -> t * v
            self.outer = nn.Parameter(
                torch.stack([torch.from_numpy(A).float() for _ in range(num_heads)], dim=0),
                requires_grad=True)

            self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        N, T, C = x.shape  # T = t * v
        assert T == self.num_frames * self.num_points
        qkv = self.qkv(x).reshape(N, T, 3, self.num_heads, self.head_dim)
        qkv = rearrange(qkv, 'n t o h c -> o n h t c')

        q, k, v = qkv  # n h (t v) c

        attn = q @ k.transpose(-2, -1)  # n h (t v) (t v)

        if self.hop_bias:
            # hop relative positional embedding
            attn_bias_hop = self.hop_rpb()

            # temporal relative positional bias
            attn_bias_temporal = self.temporal_rpb()

            attn = attn + attn_bias_hop + attn_bias_temporal

        else:
            attn_bias = self.rpb()
            attn = attn + attn_bias

        attn = attn * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # relational bias
        if self.relational_bias:
            # Initialize a list to hold the attention bias matrices
            attention_biases = []

            # Loop over each slice (i.e., each 25x25 matrix) and create a 64*25x64*25 matrix
            for i in range(self.num_heads):
                # Extract the i-th 25x25 matrix
                single_learnable_matrix = self.outer[i]

                # Create a 64*25x64*25 matrix using block_diag or manual filling
                blocks = [single_learnable_matrix for _ in range(self.num_frames)]
                attention_bias = torch.block_diag(*blocks)

                # Add the created matrix to the list
                attention_biases.append(attention_bias)

            # Stack the individual attention bias matrices to get the final tensor
            attn_relational = torch.stack(attention_biases)

            x = (self.alpha * attn + attn_relational) @ v
        else:
            x = attn @ v

        x = rearrange(x, 'n h t c -> n t (h c)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class unit_coupled(nn.Module):
    def __init__(
            self, dim_in, dim, A, num_heads, qkv_bias=False, qk_scale=None,
            attn_drop=0, drop=0., drop_path=0., norm_layer=nn.LayerNorm,
            num_points=25, num_frames=64, ff_mult=4, relational_bias=False,
            hop_bias=True, hops=None,
    ):
        super().__init__()
        self.norm = norm_layer(dim_in)
        self.attn = CoupledAttention(
            dim_in, dim, A, num_heads, qkv_bias, qk_scale, attn_drop,
            proj_drop=drop, num_points=num_points, num_frames=num_frames,
            relational_bias=relational_bias, hop_bias=hop_bias, hops=hops,
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.skip_proj = nn.Linear(
            dim_in, dim, bias=False) if dim_in != dim else nn.Identity()

        self.use_mlp = ff_mult > 0
        if self.use_mlp:
            self.norm1 = norm_layer(dim)
            self.mlp = Mlp(dim, dim * ff_mult, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, *args):
        if x.dim() == 4:
            x = rearrange(x, 'n c t v -> n (t v) c')

        x = self.skip_proj(x) + self.drop_path(self.attn(self.norm(x)))

        if self.use_mlp:
            x = x + self.drop_path(self.mlp(self.norm1(x)))

        return x