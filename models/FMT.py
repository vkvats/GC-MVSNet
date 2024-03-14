import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .module import *

'''
- We provide two different positional encoding methods as shown below.
- You can easily switch different pos-enc in the __init__() function of FMT.
- In our experiments, PositionEncodingSuperGule usually cost more GPU memory.
'''
from .position_encoding import PositionEncodingSuperGule, PositionEncodingSine


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = lambda x: torch.nn.functional.elu(x) + 1
        self.eps = eps

    def forward(self, queries, keys, values):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, d_ff=None, dropout=0.0,
                 activation="relu"):
        super(EncoderLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        inner_attention = LinearAttention()
        attention = AttentionLayer(inner_attention, d_model, n_heads, d_keys, d_values)

        d_ff = d_ff or 2 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, source):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]

        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, source, source,
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)


class FMT(nn.Module):
    def __init__(self, config):
        super(FMT, self).__init__()

        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = EncoderLayer(config['d_model'], config['nhead'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

        # self.pos_encoding = PositionEncodingSuperGule(config['d_model'])
        self.pos_encoding = PositionEncodingSine(config['d_model'])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ref_feature=None, src_feature=None, feat="ref"):
        """
        Args:
            ref_feature(torch.Tensor): [N, C, H, W]
            src_feature(torch.Tensor): [N, C, H, W]
        """

        assert ref_feature is not None

        if feat == "ref": # only self attention layer

            assert self.d_model == ref_feature.size(1)
            _, _, H, _ = ref_feature.shape

            ref_feature = einops.rearrange(self.pos_encoding(ref_feature), 'n c h w -> n (h w) c')

            ref_feature_list = []
            for layer, name in zip(self.layers, self.layer_names): # every self attention layer
                if name == 'self':
                    ref_feature = layer(ref_feature, ref_feature)
                    ref_feature_list.append(einops.rearrange(ref_feature, 'n (h w) c -> n c h w', h=H))
            return ref_feature_list

        elif feat == "src":

            assert self.d_model == ref_feature[0].size(1)
            _, _, H, _ = ref_feature[0].shape

            ref_feature = [einops.rearrange(_, 'n c h w -> n (h w) c') for _ in ref_feature]

            src_feature = einops.rearrange(self.pos_encoding(src_feature), 'n c h w -> n (h w) c')

            for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
                if name == 'self':
                    src_feature = layer(src_feature, src_feature)
                elif name == 'cross':
                    src_feature = layer(src_feature, ref_feature[i // 2])
                else:
                    raise KeyError
            return einops.rearrange(src_feature, 'n (h w) c -> n c h w', h=H)
        else:
            raise ValueError("Wrong feature name")



class FMT_with_pathway(nn.Module):
    def __init__(self,
            base_channels=8,
            FMT_config={
                'd_model': 32,
                'nhead': 8,
                'layer_names': ['self', 'cross'] * 4}):

        super(FMT_with_pathway, self).__init__()

        self.FMT = FMT(FMT_config)

        self.dim_reduction_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 1, bias=False)
        self.dim_reduction_2 = nn.Conv2d(base_channels * 2, base_channels * 1, 1, bias=False)

        self.smooth_1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, bias=False)
        self.smooth_2 = nn.Conv2d(base_channels * 1, base_channels * 1, 3, padding=1, bias=False)

    def _upsample_add(self, x, y):
        """_upsample_add. Upsample and add two feature maps.

        :param x: top feature map to be upsampled.
        :param y: lateral feature map.
        """

        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y


    def forward(self, features):
        """forward.

        :param features: multiple views and multiple stages features
        """

        for nview_idx, feature_multi_stages in enumerate(features):
            if nview_idx == 0: # ref view
                ref_fea_t_list = self.FMT(feature_multi_stages["stage1"].clone(), feat="ref")
                feature_multi_stages["stage1"] = ref_fea_t_list[-1]
                feature_multi_stages["stage2"] = self.smooth_1(self._upsample_add(self.dim_reduction_1(feature_multi_stages["stage1"]), feature_multi_stages["stage2"]))
                feature_multi_stages["stage3"] = self.smooth_2(self._upsample_add(self.dim_reduction_2(feature_multi_stages["stage2"]), feature_multi_stages["stage3"]))

            else: # src view
                feature_multi_stages["stage1"] = self.FMT([_.clone() for _ in ref_fea_t_list], feature_multi_stages["stage1"].clone(), feat="src")
                feature_multi_stages["stage2"] = self.smooth_1(self._upsample_add(self.dim_reduction_1(feature_multi_stages["stage1"]), feature_multi_stages["stage2"]))
                feature_multi_stages["stage3"] = self.smooth_2(self._upsample_add(self.dim_reduction_2(feature_multi_stages["stage2"]), feature_multi_stages["stage3"]))

        return features
## -------------------------------------------------------------------------------------------------------------------
############################ 2D attention Feature Matching Transformer #####################
## single head attention
class SingleHead2DAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.soft_row = torch.nn.Softmax(dim=3)
        self.soft_col = torch.nn.Softmax(dim=2)
    
    def min_max_norm(self, val): 
        min_, max_, mean_ = val.min(), val.max(), val.mean()
        val = 2.0 + (val - min_)/(max_ - min_) 
        # val = (val - min_)/(max_ - min_)*2.0
        return val
    
    def q_transpose_k(self, q, k):
        w_x_w = torch.einsum("bchw,bcxy->bcwy", q, k) + self.eps
        w_row = self.min_max_norm(torch.einsum("bcij->bcj", w_x_w))
        w_col = self.min_max_norm(torch.einsum("bcij->bci", w_x_w))
        return w_row, w_col
    
    def q_k_transpose(self, q, k):
        h_x_h = torch.einsum("bchw,bcxy->bchx", q, k) + self.eps
        h_row = self.min_max_norm(torch.einsum("bcij->bcj", h_x_h))
        h_col = self.min_max_norm(torch.einsum("bcij->bci", h_x_h))
        return h_row, h_col

    def outer_product(self, m1, m2):
        return torch.einsum("bch, bcw->bchw", m1, m2)
    
    def attention_score(self, m1, m2, n1, n2):
        """Calculate all possible MxN attention scores""" 
        m1n1 = self.outer_product(m1, n1)
        m1n2 = self.outer_product(m1, n2)
        m2n1 = self.outer_product(m2, n1)
        m2n2 = self.outer_product(m2, n2)
        return m1n1, m1n2, m2n1, m2n2

    def row_column_softmax(self, score):
        """calculate softmax along row and column, add and then normalize"""
        row_col = self.soft_row(score) + self.soft_col(score)
        return row_col/torch.max(row_col)

    def attention_softmax(self, s1, s2, s3, s4):
        s1 = self.row_column_softmax(s1)
        s2 = self.row_column_softmax(s2)
        s3 = self.row_column_softmax(s3)
        s4 = self.row_column_softmax(s4)
        score = s1 + s2 + s3 + s4
        return score/torch.max(score)
        
    def forward(self, queries, keys, values):
        B,C,H,W = queries.shape
        z = torch.zeros(values.shape)
        for idx in range(C): 
            ## select the query
            q_tem = queries[:, idx, :, :].view(B,1,H,W)
            q_tem = q_tem.repeat(1, keys.shape[1], 1, 1)
            ## calculate dot product
            q_kt_row, q_kt_col = self.q_k_transpose(q_tem, keys)
            qt_k_row, qt_k_col = self.q_transpose_k(q_tem, keys)
            ## calculate score using outer product
            s1, s2, s3, s4 = self.attention_score(q_kt_row, q_kt_col, qt_k_row, qt_k_col) 
            #normalize and calculate two way softmax
            atten_score = self.attention_softmax(s1, s2, s3, s4)
            ## multiply attention score with values 
            z_idx = torch.einsum("bchw, bchw->bhw", atten_score, values) 
            z[:, idx, :,:] = z_idx
        return z, atten_score
            

## Multi-head Attention
class MultiHead2DAttention(nn.Module):
    def __init__(self, c_in, heads=4, gn_g=4):
        super().__init__()
        self.head = heads 
        self.key = Conv2d(c_in, c_in*heads, 3, 1, padding=1, bias=False, gn_group=gn_g)
        self.query = Conv2d(c_in, c_in*heads, 3, 1, padding=1, bias=False, gn_group=gn_g)
        self.value = Conv2d(c_in, c_in*heads, 3, 1, padding=1, bias=False, gn_group=gn_g)
        self.attention = SingleHead2DAttention()
        
    def rearrange_tensor(self,z_in,B,C,H,W):
        """Rearrange feature maps from each head, such that same 
        feature index from all heads are placed together. This is
        to facilitate groupd convolution for final aggregation"""
        z_out = torch.zeros(B,C*self.head,H,W)
        for f_idx in range(C): 
            if f_idx == 0: 
                step = 0
            else: 
                step += self.head
            for idx_, head_ in enumerate(z_in): 
                z_out[:,idx_ + step,:,:] = head_[:,f_idx,:,:]
        return z_out
        
    def forward(self, k,q,v):
        B,C,H,W = k.shape
        z_out = []
        key_h = self.key(k).reshape(B,self.head,C,H,W)
        query_h = self.query(q).reshape(B,self.head,C,H,W)
        value_h = self.value(v).reshape(B,self.head,C,H,W)
        ## for each head calculate single head attention 
        for idx in range(self.head): 
            k_h = key_h[:,idx,:,:,:]
            q_h = query_h[:,idx,:,:,:]
            v_h = value_h[:,idx,:,:,:]
            z_h, atten_score = self.attention(q_h,k_h,v_h)
            ## collect tensor in a list
            z_out.append(z_h)
        return self.rearrange_tensor(z_out,B,C,H,W), atten_score

## Feed-Forward Block
class ConvFeedForwardBlock(nn.Module):
    def __init__(self, c_in, gamma=4, gn_g=4, depth_wise=True):
        super().__init__()
        self.gamma = gamma
        self.dw = depth_wise
        self.conv_1x1_0 = Conv2d(c_in, c_in*gamma, 1, 1, padding=0, bias=True, gn_group=gn_g)
        if self.dw:
            self.dw_3x3 = Conv2d(c_in*gamma, c_in*gamma, 3, 1, padding=1, bias=True, groups=c_in*gamma, gn_group=gn_g)
        self.conv_1x1_1 = Conv2d(c_in*gamma, c_in, 1, 1, padding=0, bias=True, gn_group=gn_g)
        
    def forward(self, x):
        if self.dw:
            x = x + self.conv_1x1_1(self.dw_3x3(self.conv_1x1_0(x)))
        else: 
           x = x + self.conv_1x1_1(self.conv_1x1_0(x))
        
        return x

## Transformer Block
class FeatureMatchingTransformerBlock(nn.Module):
    def __init__(self, c_in, heads=4, gamma=4, gn_g=4, depth_wise=True):
        super().__init__()
        self.aggregate = Conv2d(c_in*heads, c_in, 3, 1, padding=1, groups=heads, gn_group=gn_g)
        self.attention = MultiHead2DAttention(c_in, heads, gn_g=gn_g)
        self.ffn = ConvFeedForwardBlock(c_in, gamma, gn_g=gn_g, depth_wise=depth_wise)
        
    def forward(self, k,q,v):
        ## Multi-head Attention
        z,atten_score = self.attention(k,q,v)
        z = self.aggregate(z)
        ## Feed Forward layer
        z = self.ffn(z)
        
        return z,atten_score