import torch
import torch.nn as nn


class RowdyActivation(nn.Module):
    def __init__(self, K=2, n=10.0, enable_bias=True):
        super().__init__()
        self.K = K
        self.n = n
        self.omega = nn.Parameter(torch.ones(K))
        self.alpha = nn.Parameter(torch.tensor([1.0] + [0.0] * (K - 1)))
        self.a = nn.Parameter(torch.full((K,), 0.1))
        self.c = nn.Parameter(torch.full((K,), 0.1) if enable_bias else torch.zeros(K))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            total = torch.zeros_like(x)
            for k in range(self.K):
                if k == 0:
                    branch = torch.tanh(self.n * self.a[k] * self.omega[k] * x + self.c[k])
                else:
                    branch = self.n * self.a[k] * torch.sin(
                        k * self.n * self.omega[k] * x + self.c[k])
                total = total + self.alpha[k] * branch
            return total


class FiLMFusion(nn.Module):
    def __init__(self, branch_dim, trunk_dim):
        super().__init__()
        self.scale_net = nn.Sequential(nn.Linear(branch_dim, trunk_dim))
        self.shift_net = nn.Sequential(nn.Linear(branch_dim, trunk_dim))

    def forward(self, trunk_feat, branch_feat):
        gamma = self.scale_net(branch_feat).unsqueeze(1)
        beta = self.shift_net(branch_feat).unsqueeze(1)
        return gamma * trunk_feat + beta + trunk_feat * 0.1


class MSFiLM_DON(nn.Module):
    def __init__(self, num_param=2, pos_dim=3, branch_hidden_dim=128,
                 trunk_hidden_dim=128, dropout=0.1, MS_num=8, scale_factors=2):
        super().__init__()
        self.num_param = num_param
        self.pos_dim = pos_dim
        self.branch_hidden_dim = branch_hidden_dim
        self.trunk_hidden_dim = trunk_hidden_dim
        self.MS_num = MS_num
        self.bias = nn.Parameter(torch.zeros(1))

        self.fourier_a = nn.Parameter(torch.ones(MS_num, pos_dim))
        init_freq = torch.stack([
            torch.full((pos_dim,), float(scale_factors ** i)) for i in range(MS_num)
        ])
        self.fourier_b = nn.Parameter(init_freq)

        self.scale_weights = nn.Parameter(torch.ones(MS_num))

        self.fourier_dim = 2 * pos_dim

        self.branch = nn.ModuleDict({
            'b0': nn.Sequential(nn.Linear(self.num_param, self.branch_hidden_dim), RowdyActivation()),
            'b1': nn.Sequential(nn.Linear(self.branch_hidden_dim, self.branch_hidden_dim), RowdyActivation()),
            'b2': nn.Sequential(nn.Linear(self.branch_hidden_dim, self.branch_hidden_dim), RowdyActivation()),
            'b3': nn.Sequential(nn.Linear(self.branch_hidden_dim, self.branch_hidden_dim), RowdyActivation()),
            'b4': nn.Sequential(nn.Linear(self.branch_hidden_dim, self.branch_hidden_dim))
        })

        self.trunk_nets = nn.ModuleList([
            self._create_trunk_net(self.trunk_hidden_dim, dropout) for _ in range(self.MS_num)
        ])

        self.output_proj = nn.Linear(self.branch_hidden_dim, self.trunk_hidden_dim)

    def _create_trunk_net(self, hidden_dim, dropout):
        return nn.ModuleDict({
            't0': nn.Sequential(nn.Linear(self.fourier_dim, hidden_dim), RowdyActivation()),
            't1': nn.Sequential(nn.Linear(hidden_dim, hidden_dim), RowdyActivation()),
            't2': nn.Sequential(nn.Linear(hidden_dim, hidden_dim), RowdyActivation()),
            't3': nn.Sequential(nn.Linear(hidden_dim, hidden_dim), RowdyActivation()),
            't4': nn.Sequential(nn.Linear(hidden_dim, hidden_dim), RowdyActivation()),
            'film0': FiLMFusion(self.branch_hidden_dim, hidden_dim),
            'film1': FiLMFusion(self.branch_hidden_dim, hidden_dim),
            'film2': FiLMFusion(self.branch_hidden_dim, hidden_dim),
            'film3': FiLMFusion(self.branch_hidden_dim, hidden_dim),
            'film4': FiLMFusion(self.branch_hidden_dim, hidden_dim),
        })

    def fourier_encode(self, x, a, b):
        sin_enc = a * torch.sin(b * x)
        cos_enc = a * torch.cos(b * x)
        return torch.cat([sin_enc, cos_enc], dim=-1)

    def forward_branch(self, params):
        b0 = self.branch['b0'](params)
        s0 = b0
        b1 = self.branch['b1'](b0)
        s1 = b1 + s0
        b2 = self.branch['b2'](s1)
        s2 = b2 + s1
        b3 = self.branch['b3'](s2)
        s3 = b3 + s2
        b4 = self.branch['b4'](s3)
        s4 = b4 + s3

        return b4, (s0, s1, s2, s3, s4)

    def forward_single_trunk(self, trunk_net, locs, branch_residuals):
        t0 = trunk_net['t0'](locs)
        t0 = trunk_net['film0'](t0, branch_residuals[0])

        t1 = trunk_net['t1'](t0)
        t1 = trunk_net['film1'](t1, branch_residuals[1])

        t2 = trunk_net['t2'](t1)
        t2 = trunk_net['film2'](t2, branch_residuals[2])

        t3 = trunk_net['t3'](t2)
        t3 = trunk_net['film3'](t3, branch_residuals[3])

        t4 = trunk_net['t4'](t3)

        return t4

    def forward(self, params, sensor_locs):
        mn = sensor_locs.amin(dim=1, keepdim=True)
        mx = sensor_locs.amax(dim=1, keepdim=True)
        locs_norm = (sensor_locs - mn) / (mx - mn).clamp(min=1e-6)

        b4, branch_residuals = self.forward_branch(params)
        trunk_outputs = []
        for i, trunk_net in enumerate(self.trunk_nets):
            encoded_locs = self.fourier_encode(locs_norm, self.fourier_a[i], self.fourier_b[i])
            trunk_t4 = self.forward_single_trunk(trunk_net, encoded_locs, branch_residuals)
            trunk_outputs.append(trunk_t4)

        w = torch.softmax(self.scale_weights, dim=0)
        b4_proj = self.output_proj(b4)
        pred = w[0] * torch.sum(b4_proj.unsqueeze(1) * trunk_outputs[0], dim=-1)
        for i in range(1, self.MS_num):
            pred = pred + w[i] * torch.sum(b4_proj.unsqueeze(1) * trunk_outputs[i], dim=-1)
        pred = pred + self.bias

        return pred
