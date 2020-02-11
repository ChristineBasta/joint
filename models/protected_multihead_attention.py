# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils

# Adapted from faiserq/modules/multihead_attention to deal with local attention
# Local attetion masking in combination with padding masking can lead to 
# all -Inf attention rows. This version detects and corrects this situation
#Christine 31-1-2020
#adding the parts from the transformer that will make the relative and the multihead attention


# all sublayers of the model  and embedding layers produce outputs of dimension d_model
class RelMultiHeadAttn(nn.Module):
    #self, embed_dim, num_heads, dropout = 0., bias = True, add_bias_kv = False, add_zero_attn = False

    def __init__(self, n_head, d_model, d_head, dropout,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        # first difference
        # d_model=dimension of embeddings, will be divided on three and so n_head and d_head
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        #removing this from parameters (Christine 5-2-2020)
        #self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    # Append one "column" of zeros to the left
    # Reshape the matrix from [3 x 4] into [4 x 3]
    # Remove the first "row"
    # Mask out the upper triangle
    # see issue https://github.com/kimiyoung/transformer-xl/issues/8
    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


#adding the superclass part to adapt from the transformer
#Christine (31-1-2020)
class ProtectedMultiheadAttention(RelMultiHeadAttn):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """
    #already the RelMultiHeadAttention has some of these so ...have to send only what is needed more
    # self, embed_dim, num_heads, dropout = 0., bias = True, add_bias_kv = False, add_zero_attn = False

    #self, n_head, d_model, d_head, dropout, dropatt=0,tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False
    #d_model=Embed_dim
    #num_heads=n_head
    #dropatt=0,tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False ..are the different paarameters from above
    # bias=True, add_bias_kv=False, add_zero_attn=False
    def __init__(self, n_head, d_model, d_head, dropout=0.,tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False, bias=True, add_bias_kv=False, add_zero_attn=False):
        #super().__init__()
        #we need to change this to be similar to the init of protected to be initilaized with similar parameters
        #tab lw 3yza ab3at elparameters elly fo2 w hagat tanya kaman....a3ml eh?
        #super(ProtectedMultiheadAttention, self).__init__(*args, **kwargs)

        super().__init__(n_head, d_model, d_head, dropout)

        self.n_head = n_head
        self.d_model = d_model
        # removing this from the parameters (Christine 5-2-2020)
        #self.dropatt=dropatt
        self.tgt_len=tgt_len
        self.ext_len=ext_len
        self.mem_len=mem_len
        self.pre_lnorm=pre_lnorm


        self.dropout = dropout
        self.d_head= d_head   #we need to know how to initialize

        #we need this Christine 5-2-2020..bbas sibk dlw2ty
        #assert self.head_dim * n_head == self.d_model, "embed_dim must be divisible by num_heads"
        self.scaling = self.d_head ** -0.5

        #replacing all embed_dim by d_model
        self.in_proj_weight = Parameter(torch.Tensor(3 * d_model, d_model))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * d_model))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, d_model))
            self.bias_v = Parameter(torch.Tensor(1, 1, d_model))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        #(Christine 31-1-2020) for transformer attention
        #d_model os changed to embedding dimension, n_head to num_heads and d_head to head_dim
        self.r_net = nn.Linear(self.d_model , self.n_head * self.d_head, bias=False)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

        # the forward method takes  w, r, r_w_bias, r_r_bias,, mems = None
        # w is the word embeddings..need to be sure again that it is the query here and that it means the mebddings there
        # r is the positional relative embeddings and the other two are biases that need learning
        # I think here w is any of query, or key or value

        # we have to add the r, r_w_bias, r_r_bias
    def forward(self, query, key, value,r, r_w_bias, r_r_bias, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None, mems=None):


        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()
        tgt_len, bsz, embed_dim = query.size() #make sure it is ok, Christine 3-2-2020

        assert embed_dim== self.d_model,"both embed_dim and d_model should be the same"
        #we need to make sure if the shape works in both cases

        #transforerxl starts here Christine 3-2-2020 ...so tgt_len is equal to qlen..make sure
        qlen, rlen, bsz = query.size(0), r.size(0), query.size(1)  # assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # in the paper they are using w for query, I believe query is more representable
        w=query

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        # is qlen different that klen??
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        # they are computing linear transmformation
        # this is for the bias of the context
        # Equation we follow is : EWq * Wk E
        # w_head_q is E*Wq
        # r_w_bias= u
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        # the r_w_bias seems to be the learnt part u
        # w_head_k Wk*E
        # w_head_k is the common part of the two terms A, C
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        # this for the bias of the relative embeddings
        rr_head_q = w_head_q + r_r_bias
        # we do some linear transformation
        # r_head_k is W(k)*R
        # r_r_bias is v
        # w_head_q = E*Wq
        # r_head_k is the common part of two terms B, D
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        # score, scale
        attn_score.mul_(self.scale)
        #end of taken part from Transformer Xl

        # mask
        # Christine 3-2-2020....should be changed with the other kind of mask
        #from here we should use the maks of Adrian insted of the other
        #gathering both together should be tested for dimensiojn and mask
        attn_weights=attn_score
        src_len= klen # need to make sure Christine 3-2-2020
        tgt_len=qlen  # need to make sure Christine 3-2-2020
        assert list(attn_weights.size()) == [bsz * self.n_head, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            #maybe we don't have to attend to this now as Cralos said, will leave it till further review
            # Christine 3-2-2020
            attn_weights = attn_weights.view(bsz, self.n_head, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.n_head, tgt_len, src_len)
            all_inf = torch.isinf(attn_weights).all(dim=-1)
            if all_inf.any():
                attn_weights = attn_weights.float().masked_fill(
                    all_inf.unsqueeze(-1),
                    0,
                ).type_as(attn_weights)  # FP16 support: cast to float and back

        # ghalban attention_Weiths btrepresent elscore
        # fa momkn
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)



        v=w_head_v # need to make sure Christine 3-2-2020
        attn = torch.bmm(attn_weights, v)

        assert list(attn.size()) == [bsz * self.n_head, tgt_len, self.d_head]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, self.d_model)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.d_model)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.n_head, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.n_head
        else:
            attn_weights = None

        # the only remaining part here from transformer Xl is the residual connection and layer normalization






        # in our case.....need to handle increment state (Christine 3-2-2020)
        '''
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None
        '''


        #do we need this type of handling for mask computation
        '''
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
        '''



        #to here handled other way


        #to save state, but we still need to save other state not this
        '''
        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)
        
        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)
        '''


        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.d_model).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.d_model)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.d_model, end=2 * self.d_model)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.d_model)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )
