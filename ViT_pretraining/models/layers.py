import torch
import torch.nn as nn
import torch.nn.functional as F


def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

def softmax_off_by_one(x, dim=-1):
    # f(x) = e^x / (1 + e^x)
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True).values)  # For numerical stability
    return e_x / (1 + e_x.sum(dim=dim, keepdim=True))
        
class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0., 
                 slim_msa=False, msa_prune_ratio=.0, slim_mlp=False, mlp_prune_ratio=.0,
                 slim_before=False, soft_by_one=False):
        
        super(TransformerEncoder, self).__init__()

        # Attention: layernorm -> multihead self attention -> layernorm
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout, 
                                          slim=slim_msa, prune_ratio=msa_prune_ratio,
                                          slim_before=slim_before, soft_by_one=soft_by_one)
        self.la2 = nn.LayerNorm(feats)

        # MLP: intermediate -> layernorm -> output
        self.mlp = TransformerMLP(feats, mlp_hidden, feats, dropout=dropout, slim=slim_mlp, prune_ratio=mlp_prune_ratio)

    def prune_heads(self, heads):
        self.msa.prune_heads(heads)

    def prune_neurons(self, neurons):
        self.mlp.prune_neurons(neurons)
       
    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out

class TransformerMLP(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, out_features:int, dropout:float=0., 
                 slim=False, prune_ratio=.0):

        super().__init__()
        self.slim =  slim
        self.prune_ratio = prune_ratio

        self.intermediate = nn.Linear(feats, mlp_hidden)
        self.gelu1 = nn.GELU()
        self.dropout1 =  nn.Dropout(dropout)
        
        self.output = nn.Linear(mlp_hidden, out_features)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        if self.slim:
            self.slimming_coef = nn.Parameter(torch.ones(mlp_hidden).reshape(1,-1) * 1.0)
        self.pruned_neurons = set()

    def forward(self, x):
        out = self.intermediate(x)
        if self.slim:
            out = out * self.slimming_coef
        out = self.gelu1(out)
        out = self.dropout1(out)
        
        out = self.output(out)
        out = self.gelu2(out)
        out = self.dropout2(out)
        return out

    def prune_neurons(self, neurons):
        if len(neurons) == 0:
            return
        mask = torch.ones(self.mlp_hidden)
        neurons = set(neurons) - self.pruned_neurons  # Convert to set and remove already pruned neurons
        if self.slim:
            slimming_mask = torch.ones(self.mlp_hidden)
        for neuron in neurons:
            # Compute how many pruned neurons are before the neuron and move the index accordingly
            neuron = neuron - sum(1 if n < neuron else 0 for n in self.pruned_neurons)
            mask[neuron] = 0
            if self.slim:
                slimming_mask[neuron] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        if self.slim:
            slimming_mask = slimming_mask.view(-1).contiguous().eq(1)
            slimming_index = torch.arange(len(slimming_mask))[slimming_mask].long()

        # Prune linear layers
        self.intermediate = prune_linear_layer(self.intermediate, index)
        self.output = prune_linear_layer(self.output, index, dim=1)

        # Update hyper params and store pruned neurons
        self.pruned_neurons = self.pruned_neurons.union(neurons)

        # Prune slimming coefficients
        if self.slim:
            slimming_index = slimming_index.to(self.slimming_coef.device)
            new_data = self.slimming_coef.data.index_select(1, slimming_index).clone().detach()
            with torch.no_grad():
                self.slimming_coef = nn.Parameter(new_data)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0., bias:bool=True, 
                 slim=False, prune_ratio=.0, slim_before=False, soft_by_one=False):
        super(MultiHeadSelfAttention, self).__init__()

        self.feats = feats
        self.slim = slim
        self.prune_ratio = prune_ratio
        self.soft_by_one = soft_by_one
        self.slim_before = slim_before

        self.head =  round(head * (1 - self.prune_ratio))
        self.all_head_size = round(feats * (1 - self.prune_ratio))
        self.attention_head_size = int(self.all_head_size / self.head)
        self.sqrt_d = self.attention_head_size**0.5
        self.head = head

        self.q = nn.Linear(self.feats,  self.all_head_size, bias)
        self.k = nn.Linear(self.feats,  self.all_head_size, bias)
        self.v = nn.Linear(self.feats,  self.all_head_size, bias)
        self.o = nn.Linear(self.all_head_size,  self.feats)

        self.dropout = nn.Dropout(dropout)

        if self.slim:
            self.slimming_coef = nn.Parameter(torch.ones(self.head).reshape(1,-1,1,1) * 1.0)
        self.pruned_heads = set()

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.attention_head_size).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.attention_head_size).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.attention_head_size).transpose(1,2)

        att_weights = torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d

        if self.slim and self.slim_before:
            att_weights = att_weights * self.slimming_coef

        if self.soft_by_one:
            att_probs = softmax_off_by_one(att_weights, dim=-1)
        else:
            att_probs = F.softmax(att_weights, dim=-1) #(b,h,n,n)

        if self.slim and not self.slim_before:
            att_probs = att_probs * self.slimming_coef

        # attention_probs * v
        attn = torch.einsum("bhij, bhjf->bihf", att_probs, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o
    
    def prune_heads(self, heads):
        if len(heads)==0:
            return

        mask = torch.ones(self.head, self.attention_head_size)
        heads = list(set(heads) - self.pruned_heads)

        if self.slim:
            slimming_mask = torch.ones(self.head)

        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
            if self.slim:
                slimming_mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        if self.slim:
            slimming_mask = slimming_mask.view(-1).contiguous().eq(1)
            slimming_index = torch.arange(len(slimming_mask))[slimming_mask].long()

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        
        # Update hyper params and store pruned heads
        self.head = self.head - len(heads)
        self.all_head_size = self.attention_head_size * self.head
        self.pruned_heads = self.pruned_heads.union(heads)

        if self.slim:
            slimming_index = slimming_index.to(self.slimming_coef.device)
            new_data = self.slimming_coef.data.index_select(1, slimming_index).clone().detach()
            with torch.no_grad():
                self.slimming_coef = nn.Parameter(new_data)

