import torch as th
from torch import nn
import geoopt as gt
import numpy as np
import math


class HyperGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, c):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.c = c
        self.ball = gt.Stereographic(-self.c)

        self.b_z = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, manifold=self.ball).zero_()
        )
        self.b_r = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, manifold=self.ball).zero_()
        )
        self.b_h = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, manifold=self.ball).zero_()
        )

        self.fc_Wz = Hyp_plus_FC(self.hidden_size, self.hidden_size, self.c, False)
        self.fc_Uz = Hyp_plus_FC(self.hidden_size, self.input_size, self.c, False)

        self.fc_Wh = Hyp_plus_FC(self.hidden_size, self.hidden_size, self.c, False)
        self.fc_Uh = Hyp_plus_FC(self.hidden_size, self.input_size, self.c, False)

        self.fc_Wr = Hyp_plus_FC(self.hidden_size, self.hidden_size, self.c, False)
        self.fc_Ur = Hyp_plus_FC(self.hidden_size, self.input_size, self.c, False)

    def forward(self, hyp_x, hidden):
        z = self.ball.mobius_add(self.fc_Wz(hidden), self.fc_Uz(hyp_x))
        z = self.ball.mobius_add(z, self.b_z)
        z = th.sigmoid(self.ball.logmap0(z))

        r = self.ball.mobius_add(self.fc_Wr(hidden), self.fc_Ur(hyp_x))
        r = self.ball.mobius_add(r, self.b_r)
        r = th.sigmoid(self.ball.logmap0(r))

        r_point_h = self.ball.mobius_pointwise_mul(hidden, r)
        h_tilde = self.ball.mobius_add(self.fc_Wh(r_point_h), self.fc_Uh(hyp_x))
        h_tilde = self.ball.mobius_add(r, self.b_h)

        minus_h_oplus_htilde = self.ball.mobius_add(-hidden, h_tilde)
        new_h = self.ball.mobius_add(
            hidden, self.ball.mobius_pointwise_mul(minus_h_oplus_htilde, z)
        )

        return new_h


class HyperGRU(nn.Module):
    def __init__(self, input_size, hidden_size, c, default_dtype=th.float64):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.c = c
        self.ball = gt.Stereographic(-self.c)
        self.default_dtype = default_dtype

        self.gru_cell = HyperGRUCell(self.input_size, self.hidden_size, self.c)

    def init_gru_state(self, batch_size, hidden_size, cuda_device):
        return th.zeros(
            (batch_size, hidden_size), dtype=self.default_dtype, device=cuda_device
        )

    def forward(self, inputs):
        hidden = self.init_gru_state(inputs.shape[0], self.hidden_size, inputs.device)
        outputs = []
        for x in inputs.transpose(0, 1):
            hidden = self.gru_cell(x, hidden)
            outputs += [hidden]
        return th.stack(outputs).transpose(0, 1)


class Hyp_MLR(nn.Module):
    def __init__(self, num_class, dim, c=1):
        super().__init__()

        self.num_class = num_class
        self.dim = dim
        self.c = c # curvature
        self.ball = gt.Stereographic(-self.c)

        self.a_mlr = nn.Parameter(data=th.zeros(self.num_class, self.dim))
        self.p_mlr = nn.Parameter(data=th.zeros(self.num_class, self.dim))    # scalar
        nn.init.uniform_(self.a_mlr, -0.05, 0.05)
        nn.init.uniform_(self.p_mlr, -0.05, 0.05)

        self.hyper_para = [self.p_mlr]
        self.euclid_para = [self.a_mlr]
    
    @staticmethod
    def _lambda(vector):
        return 2. / (1-th.sum(vector * vector, dim=1))

    def forward(self, output_before):
        logits = []
        for cl in range(self.num_class):
            minus_p_plus_x = self.ball.mobius_add(-self.p_mlr[cl], output_before)    # [batch, hidden]
            norm_a = th.norm(self.a_mlr[cl])
            lambda_px = self._lambda(minus_p_plus_x)    # [batch, 1]
            px_dot_a = th.sum(minus_p_plus_x * nn.functional.normalize(self.a_mlr[cl].unsqueeze(0), p=2), dim=1)   # [batch, 1]
            logit = 2. * norm_a * th.asinh(px_dot_a * lambda_px)
            logits.append(logit)
        
        logits = th.stack(logits, axis=1)
        return logits

class Hyp_plus_MLR(nn.Module):
    def __init__(self, num_class, dim, c=1, bias=True):
        super().__init__()

        self.num_class = num_class
        self.dim = dim
        self.c = c # curvature
        self.bias = bias
        self.ball = gt.Stereographic(-self.c)

        self.z_mlr = nn.Parameter(data=th.zeros(self.num_class, self.dim))
        self.mlr_r = nn.Parameter(data=th.zeros(self.num_class, 1))    # scalar
        nn.init.uniform_(self.z_mlr, -0.05, 0.05)

        self.hyper_para = []
        self.euclid_para = [self.z_mlr]
        if not self.bias:
            nn.init.uniform_(self.mlr_r, -1, 1)
            self.euclid_para += [self.mlr_r]
    
    @staticmethod
    def _lambda(vector):
        return 2. / (1-th.sum(vector * vector, dim=1))

    def forward(self, output_before):
        logits = []
        for cl in range(self.num_class):
            a_k = (1 / th.cosh(self.mlr_r[cl]*math.sqrt(self.c))**2) * self.z_mlr[cl]
            q_k = self.ball.expmap0(self.mlr_r*self.z_mlr[cl])

            minus_p_plus_x = self.ball.mobius_add(-q_k, output_before)    # [batch, hidden]
            norm_a = th.norm(a_k)
            lambda_px = self._lambda(minus_p_plus_x)    # [batch, 1]
            px_dot_a = th.sum(minus_p_plus_x * nn.functional.normalize(a_k.unsqueeze(0), p=2), dim=1)   # [batch, 1]
            logit = 2. * norm_a * th.asinh(px_dot_a * lambda_px)
            logits.append(logit)
        
        logits = th.stack(logits, axis=1)
        return logits


class Hyp_plus_FC(nn.Module):
    def __init__(self, out_dim, in_dim, c=1, bias=True):
        super().__init__()

        self.out_dim = out_dim  # m
        self.in_dim = in_dim
        self.c = c              # curvature
        self.bias = bias

        self.mlr = Hyp_plus_MLR(self.out_dim, self.in_dim, self.c, self.bias)

        self.hyper_para = self.mlr.hyper_para
        self.euclid_para = self.mlr.euclid_para

    def forward(self, output_before):
        v_k = self.mlr(output_before)
        w = th.sinh(v_k * math.sqrt(self.c))
        y = (th.sqrt(self.c * th.norm(w, dim=-1)**2) + 1) * w
        return y
