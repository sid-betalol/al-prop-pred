import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing, radius_graph
from torch.nn import Dropout

def _merge(s, v):
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)

def _split(x, nv):
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v

class RBFEmb(nn.Module):
    def __init__(self, num_rbf = 20, cutoff=5.0):
        super(RBFEmb, self).__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
    def forward(self, d):
        d = d.unsqueeze(-1)
        coeff = (torch.arange(1, self.num_rbf + 1).float() * torch.pi/self.cutoff).view(1, -1)
        epsilon = 1e-15
        rbf = torch.sin(d*coeff)/(d+epsilon)
        cutoff_vals = 0.5*(torch.cos(torch.pi*d/self.cutoff)+1)
        rbf = rbf*cutoff_vals
        return rbf * (d < self.cutoff).float()
    
class Message(MessagePassing):
    def __init__(self, hc, num_rbf = 20, cutoff = 5.0):
        super(Message, self).__init__(aggr="add")
        self.hc = hc
        self.cutoff = cutoff
        self.rbf = RBFEmb(num_rbf, cutoff)
        
        self.phi = nn.Sequential(
            nn.Linear(hc, hc),
            nn.SiLU(),
            nn.Linear(hc, hc*3)
        )
        self.W_s = nn.Linear(num_rbf, hc)
        self.W_v = nn.Linear(num_rbf, hc)
        self.W_vs = nn.Linear(num_rbf, hc)
        
    def forward(self, s, v, edge_index, edge_attr):
        return self.propagate(edge_index, s=s, v=v, edge_attr = edge_attr)
    
    def message(self, edge_index, s, v, edge_attr):
        row, col = edge_index
        s_row = s[row]  # Source node features
        s_col = s[col]  # Target node features

        d = torch.norm(edge_attr, dim=1)
        rbf = self.rbf(d)

        phi_s, phi_v, phi_vs = torch.split(self.phi(s_row), self.hc, dim=-1)

        # weights for each feature based on RBF
        W_s = self.W_s(rbf)
        W_v = self.W_v(rbf).unsqueeze(-1)
        W_vs = self.W_vs(rbf).unsqueeze(-1)
        
        # messages
        ds = phi_s * W_s
        dv = v[row] * (phi_v.unsqueeze(-1) * W_v) + phi_vs.unsqueeze(-1) * W_vs * edge_attr.unsqueeze(1)

        return _merge(ds, dv)
    
    def update(self, aggr_out, s, v):
        ds, dv = _split(aggr_out, self.hc)
        s = s + ds
        v = v + dv
        return s, v
    
class GatedEquivariantBlock(nn.Module):
    def __init__(self, hc):
        super(GatedEquivariantBlock, self).__init__()
        self.hc = hc
        
        self.W_v = nn.Linear(hc, hc)
        self.W_s = nn.Linear(2*hc, hc)
        self.W_g = nn.Linear(2*hc, hc)
        
        self.layernorm = nn.LayerNorm(hc)
        self.act = nn.SiLU()
        
    def forward(self, s, v):
        #Stack operation
        v_norm = torch.norm(v, dim =-1)
        sv_stack = torch.cat([s, v_norm], dim=-1)
        
        #Gating Mechanism
        gate = torch.sigmoid(self.W_g(sv_stack))
        
        #Update scalar features
        s_out = self.W_s(sv_stack)
        s_out = self.layernorm(s_out)
        s_out = self.act(s_out)
        s_out = s + gate*s_out
        
        #Update vector features
        v_out = self.W_v(v.transpose(1,2)).transpose(1,2)
        v_out = v + gate.unsqueeze(-1)*v_out
        
        return s_out, v_out
    
class Update(nn.Module):
    def __init__(self, hc):
        super(Update, self).__init__() 
        self.hc = hc
        self.gate = GatedEquivariantBlock(hc)
        self.U = nn.Linear(hc, hc)
        self.V = nn.Linear(hc, hc)
        self.a = nn.Sequential(
            nn.Linear(hc*2, hc),
            nn.SiLU(),
            nn.Linear(hc, hc*3)
        )
        
    def forward(self, s, v):
        s, v = self.gate(s, v)
        
        v_norm = torch.norm(v, dim=-1)
        sv_stack = torch.cat([s, v_norm], dim=-1)
        a = self.a(sv_stack)
        a_ss, a_sv, a_vv = torch.split(a, self.hc, dim=-1)
        
        U_v = self.U(v.transpose(1,2)).transpose(1,2)
        V_v = self.V(v.transpose(1,2)).transpose(1,2)
        
        #Scalar residual
        ds = a_ss + a_sv*torch.sum(U_v*V_v, dim = -1)
        
        #Vector residual
        dv = a_vv.unsqueeze(-1) * U_v
        
        s = s + ds
        v = v + dv
        
        return s, v
    
class VectorDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(VectorDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = torch.bernoulli(torch.full((x.size(0), x.size(1), 1), 1 - self.p)).to(x.device) / (1 - self.p)
        return x * mask
    
class FinalModel(nn.Module):
    def __init__(self, hc = 128, num_layers = 4, num_rbf = 20, cutoff = 5.0, dropout_rate=0.1):
        super(FinalModel, self).__init__()
        self.hc = hc
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        
        self.embed = nn.Embedding(118, hc)
        self.message_layers = nn.ModuleList([Message(hc, num_rbf, cutoff) for _ in range(num_layers)])
        self.update_layers = nn.ModuleList([Update(hc) for _ in range(num_layers)])
        self.scalar_dropout = Dropout(p=dropout_rate)
        self.vector_dropout = VectorDropout(p=dropout_rate)
        
        self.out = nn.Sequential(
            nn.Linear(hc, hc),
            nn.SiLU(),
            Dropout(p=dropout_rate),
            nn.Linear(hc, 1)
        )
        
    def forward(self, z, pos, batch):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        edge_attr = pos[edge_index[0]] - pos[edge_index[1]]
        
        s = self.embed(z)
        v = torch.zeros(s.size(0), self.hc, 3, device = s.device)
        
        for message, update in zip(self.message_layers, self.update_layers):
            s, v = message(s, v, edge_index, edge_attr)
            s, v = update(s, v)
            s = self.scalar_dropout(s)
            v = self.vector_dropout(v)
            
        s = scatter(s, batch, dim = 0, reduce = 'sum')
        return self.out(s).squeeze(-1)
    
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, (nn.Dropout, VectorDropout)):
                m.train()