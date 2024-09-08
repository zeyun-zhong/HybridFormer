import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np

from model.module import TCN_ViT_unit, import_class, unit_tcn, conv_init, bn_init, unit_coupled


class HybridFormer(nn.Module):
    def __init__(
            self,
            *,
            num_class=60,
            dims=(64, 64, 128, 256),
            depths=(2, 2, 4, 2),
            mhsa_types=('l', 'l', 'l', 'g'),
            num_point=20,
            num_person=2,
            graph=None,
            graph_args=dict(),
            in_channels=3,
            global_ff_mult=4,
            local_dim_head=16,
            global_dim_head=64,
            global_relational_bias=False,
            global_hop_bias=True,
            drop=0,
            joint_label=[],
            num_frames=64,
            local_use_mlp=False,
            drop_path=0.,
            local_global_fusion=False,
            local_global_fusion_alpha=False,  # whether use a learnable alpha
            local_use_group=True,
            local_use_hop=True,
            local_use_group_bias=True,
            local_use_outer=True,
            local_use_ajacency=False,
            local_use_multiscale=True,
            local_hop_bias=False,
            local_learned_partition='none',  # ['none', 'single', 'layerwise']
            k=1,
            interpolate_mode='repeat',  # ['repeat', 'linear']
            **kwargs,
    ):
        super().__init__()

        assert len(dims) == len(depths)
        assert local_learned_partition in ['none', 'single', 'layerwise']

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25
        hops = self.graph.hops
        self.k = k

        if k > 0:
            A_vector = self.get_A(graph, k).to(torch.float32)
            self.register_buffer('A_vector', A_vector)

        self.num_class = num_class

        frames = [num_frames // (2 ** i) for i in range(len(dims))]
        mhsa_types = tuple(map(lambda t: t.lower(), mhsa_types))

        num_layers = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]  # stochastic depth decay rule

        self.layers = nn.ModuleList([])

        dim_in = in_channels
        cur_layer = 0
        for ind, (depth, mhsa_type) in enumerate(zip(depths, mhsa_types)):
            stage_dim = dims[ind]

            for j in range(1, depth+1):
                if mhsa_type == 'l':
                    heads = stage_dim // local_dim_head
                    Block = TCN_ViT_unit(
                        in_channels=dim_in,
                        out_channels=dims[ind],
                        A=A,
                        stride=2 if j == depth else 1,
                        num_of_heads=heads,
                        num_point=num_point,
                        use_mlp=local_use_mlp,
                        drop_path=dpr[cur_layer],
                        use_group=local_use_group,
                        use_hop=local_use_hop,
                        use_group_bias=local_use_group_bias,
                        use_outer=local_use_outer,
                        ues_ajacency=local_use_ajacency,
                        use_multiscale=local_use_multiscale,
                        use_hop_bias=local_hop_bias,
                        use_learned_partition=local_learned_partition,
                        hops=hops,
                        interpolate_mode=interpolate_mode,
                    )
                    self.layers.append(Block)

                elif mhsa_type == 'g':
                    heads = stage_dim // global_dim_head
                    global_block = unit_coupled(
                        dim_in=dim_in,
                        dim=dims[ind],
                        A=A,
                        num_heads=heads,
                        num_points=num_point,
                        num_frames=frames[ind],
                        ff_mult=global_ff_mult,
                        drop_path=dpr[cur_layer],
                        relational_bias=global_relational_bias,
                        hop_bias=global_hop_bias,
                        hops=hops,
                    )
                    self.layers.append(global_block)

                    # For shallow stages, we need to reduce frames in the last layer
                    # in each stage
                    if ind < len(dims) - 1 and j == depth:  # the last layer in the stage
                        self.layers.append(
                            unit_tcn(
                                dims[ind], dims[ind], kernel_size=3, stride=2,
                                num_point=num_point,
                            )
                        )
                else:
                    raise ValueError('unknown mhsa_type')

                dim_in = dims[ind]
                cur_layer += 1

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # Learned body partition
        if local_learned_partition == 'single':
            self.joint_label = nn.Parameter(torch.eye(A.shape[-1], requires_grad=True))
        else:
            if len(joint_label) > 0:
                joint_label = F.one_hot(torch.tensor(joint_label)).float()
            else:
                joint_label = None
            self.register_buffer('joint_label', joint_label)

        self.fc = nn.Linear(dims[-1], num_class)
        if drop:
            self.drop_out = nn.Dropout(drop)
        else:
            self.drop_out = lambda x: x

        # Local global fusion
        self.local_global_fusion = local_global_fusion
        self.local_global_fusion_alpha = local_global_fusion_alpha
        self.last_local_layer = sum(depths)
        if local_global_fusion:
            if local_global_fusion_alpha:
                self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
            g_index = mhsa_types.index('g')
            self.last_local_layer = sum(depths[:g_index])
            stride = 1 * (len(depths) - g_index)
            ks = stride + 1 if stride > 1 else 1
            if dims[g_index-1] == dims[-1] and stride == 1:
                self.skip_local = nn.Identity()
            else:
                self.skip_local = unit_tcn(
                    dims[g_index-1], dims[-1], kernel_size=ks, stride=stride,
                    num_point=num_point,
                )

        self.apply(self._init_weights)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        return torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            conv_init(m)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_init(m, 1)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        N, C, T, V, M = x.size()

        if self.k > 0:
            x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V)
            x = self.A_vector.expand(N * M * T, -1, -1) @ x
            x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T)
        else:
            x = rearrange(x, 'n c t v m -> n (m v c) t')

        x = self.data_bn(x)

        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V, c=C)

        x_local = 0
        for i, layer in enumerate(self.layers):
            x = layer(x, self.joint_label, None)
            if i == self.last_local_layer - 1:  # last local layer
                x_local = x

        if self.local_global_fusion:
            x_local = self.skip_local(x_local)
            x_local = rearrange(x_local, 'n c t v -> n (t v) c')

            if self.local_global_fusion_alpha:
                x = torch.sigmoid(self.alpha) * x + (1 - torch.sigmoid(self.alpha)) * x_local
            else:
                x = x + x_local

        x = rearrange(x, '(n m) t c -> n m t c', n=N, m=M)
        x = x.mean(2).mean(1)
        x = self.fc(self.drop_out(x))

        return x
