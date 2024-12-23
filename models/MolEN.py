import torch,math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
from torch import nn
from torch_scatter import scatter
from torch_geometric.nn import global_max_pool as gmp
from KGprocess import DataProcess

num_atom_type = 120+1+1
num_chirality_tag = 3
num_bond_type = 6+1+1
num_bond_direction = 3
seg_size = 3

try:
    import apex
    import apex.normalization
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
    APEX_IS_AVAILABLE = True
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    APEX_IS_AVAILABLE = False

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.shape = torch.Size((hidden_size,))
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.apex_enabled = APEX_IS_AVAILABLE

    @torch.jit.unused
    def fused_layer_norm(self, x):
        return FusedLayerNormAffineFunction.apply(
                    x, self.weight, self.bias, self.shape, self.eps)

    def forward(self, x):
        if self.apex_enabled and not torch.jit.is_scripting():
            x = self.fused_layer_norm(x)
        else:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight * x + self.bias
        return x


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


class LinearActivation(nn.Module):  # 模型的第二部分：feed-forward network with gelu activation
    r"""Fused Linear and activation Module.
    """
    def __init__(self, in_features, out_features,  bias=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features                                                                 #
        if bias:  # compatibility
            self.biased_act_fn =bias_gelu
        else:
            self.act_fn = gelu
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if not self.bias is None:
            return self.biased_act_fn(self.bias, F.linear(input, self.weight, None))
        else:
            return self.act_fn(F.linear(input, self.weight, self.bias))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class AttentionOut(nn.Module):  # 归一化之类操作
    def __init__(self, hidden,dropout):
        super(AttentionOut, self).__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = BertLayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Intermediate(nn.Module):  # 模型第二部分：feed-forward network with gelu activation
    def __init__(self, hidden):
        super(Intermediate, self).__init__()
        self.dense_act = LinearActivation(hidden, 4*hidden)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        return hidden_states


class GTOut(nn.Module):
    def __init__(self, hidden,dropout):
        super(GTOut, self).__init__()
        self.dense = nn.Linear(hidden*4, hidden)
        self.LayerNorm = BertLayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GraphAttentionConv(MessagePassing):
    def __init__(self, hidden, heads,dropout):
        super(GraphAttentionConv, self).__init__()
        self.hidden=hidden
        self.heads = heads
        assert hidden%heads==0
        self.query = nn.Linear(hidden, heads * int(hidden/heads))
        self.key = nn.Linear(hidden, heads * int(hidden/heads))
        self.value = nn.Linear(hidden, heads * int(hidden/heads))
        self.attn_drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.query.weight.data)
        torch.nn.init.xavier_uniform_(self.key.weight.data)
        torch.nn.init.xavier_uniform_(self.value.weight.data)

    def forward(self, x, edge_index, edge_attr, size=None):
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, size=size, x=x, pseudo=pseudo)

    def message(self, edge_index_i, x_i, x_j, pseudo, size_i):
        query = self.query(x_i).view(-1, self.heads, int(self.hidden / self.heads))
        key = self.key(x_j + pseudo).view(-1, self.heads, int(self.hidden / self.heads))
        value = self.value(x_j + pseudo).view(-1, self.heads, int(self.hidden / self.heads))

        alpha = (query * key).sum(dim=-1) / math.sqrt(int(self.hidden / self.heads))
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = self.attn_drop(alpha.view(-1, self.heads, 1))
        return alpha * value


    def aggregate(self, inputs, edge_index, ptr=None, dim_size=None):
        row, col = edge_index
        out = scatter(inputs, row, dim=0, reduce='sum')
        return out

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * int(self.hidden/self.heads))
        return aggr_out


class GTLayer(nn.Module):
    def __init__(self, hidden, heads,  dropout,num_message_passing):
        super(GTLayer, self).__init__()
        self.attention = GraphAttentionConv(hidden, heads, dropout)
        self.att_out = AttentionOut(hidden,dropout)

        self.intermediate = Intermediate(hidden)
        self.output = GTOut(hidden,dropout)

        self.gru = nn.GRU(hidden, hidden)
        self.LayerNorm = BertLayerNorm(hidden, eps=1e-12)
        self.time_step = num_message_passing

    def forward(self, x, edge_index, edge_attr):
        h = x.unsqueeze(0)
        for i in range(self.time_step):
            attention_output = self.attention.forward(x, edge_index, edge_attr)
            print(attention_output.size())
            print(x.size())
            attention_output = self.att_out.forward(attention_output,x)

            intermediate_output = self.intermediate.forward(attention_output)
            m = self.output.forward(intermediate_output, attention_output)
            x, h = self.gru(m.unsqueeze(0), h)
            x = self.LayerNorm.forward(x.squeeze(0))
        return x


class MolEN(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, heads, num_message_passing, drop_ratio = 0):
        super(MolEN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.x_embedding = torch.nn.Embedding(178, emb_dim)
        self.edge_embedding = torch.nn.Embedding(18, emb_dim)
        self.reset_parameters()
        self.gnns = torch.nn.ModuleList(
            [GTLayer(emb_dim, heads, drop_ratio, num_message_passing) for _ in range(self.num_layer)])

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)


    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr= argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding(x).sum(1)
        edge_attr = self.edge_embedding(edge_attr).sum(1)

        for gnn in self.gnns:
            x = gnn(x,edge_index,edge_attr)
        return x
