"""Fairseq-based implementation of the model proposed in
   `"Joint Source-Target Self Attention with Locality Constraints" (Fonollosa, et al, 2019)
    <https://>`_.
   Author: Jose A. R. Fonollosa, Universitat Politecnica de Catalunya.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options
from fairseq import utils

from fairseq.modules import PositionalEmbedding

from fairseq.models import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqModel, register_model, register_model_architecture
)

from .protected_multihead_attention import ProtectedMultiheadAttention


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        # ???(Christine)
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        # outer product of pos_seq and inv_freq
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sinusoid_inp = torch.ger(pos_seq, self.inv_freq).to(device)

        # compute relative positional embedding
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1).to(device)

        if bsz is not None:
            # first -1 means the size does not change
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        # feed forward layer
        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        # layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        # first difference
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
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

    # no method is using it (Christine)
    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
            .view(qlen, klen, x.size(2), x.size(3))

        return x

    # Append one "column" of zeros to the left
    # Reshape the matrix from [3 x 4] into [4 x 3]
    # Remove the first "row"
    # Mask out the upper triangle

    def _rel_shift(self, x, zero_triu=False):
        # CARLOS 25-2-2020 changing zeroes to ones to fit fairseq model
        #zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        #x_padded = torch.cat([zero_pad, x], dim=1)
        one_pad = torch.ones((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([one_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


# RelPartialLearnableMultiHeadAttn is the "relative positional encoding" they proposed in this work.
class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        #the linear unction is always computig WX+ Bias, by setting bias to zero,
        #
        # it is only computing W x which is W R in our case W (k,R) * R(i−j)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    # not that these are the parameters passed from the layer below
    # r is pos_emb which starts by the p
    # mems = mems[i]
    # r_w_bias starts with parameter of number of heads and dimensions and then it is learnt during the forward
    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

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

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        #they are computing linear transmformation
        #this is for the bias of the context
        # Equation we follow is : EWq * Wk E
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        #the r_w_bias seems to be the learnt part u
        #w_head_k Wk*E
        #w_head_k is the common part of the two terms A, C
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        #this for the bias of the relative embeddings
        rr_head_q = w_head_q + r_r_bias
        #we do some linear transformation
        # r_head_k is W(k)*R
        # r_r_bias is v
        #w_head_q = E*Wq
        #r_head_k is the common part of two terms B, D
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        #score, scale
        attn_score.mul_(self.scale)

        #### compute attention probability
        #mask
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        #einsum is working like mulmut
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output



class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self,args):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        #we should check how they are calling the params
        n_heads=args.decoder_attention_heads
        embed_dim=args.decoder_embed_dim
        d_heads=embed_dim/n_heads
        dropout=args.attention_dropout
        d_inner= args.decoder_ffn_embed_dim
        #we reomved **kwargs Carlos 25-2-2020
        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_heads, embed_dim,
                                                         d_heads, dropout)

        self.pos_ff = PositionwiseFF(embed_dim, d_inner, dropout,
                                     pre_lnorm= args.decoder_normalize_before)

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

@register_model('transformer_xl_translation')
class TransformerXLTranslationModel(FairseqModel):
    """
    Args:
        encoder (TranformerXLEncoder): the encoder
        decoder (TranformerXLDecoder): the decoder
    The joint source-target model provides the following named architectures and
    command-line arguments:
    .. argparse::
        :ref: fairseq.models.joint_attention_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        # control the memories, but u need top overwrite the forward of fairmodel (Carlos)
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num layers')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='embedding dimension for FFN')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num attention heads')
        parser.add_argument('--kernel-size-list', type=lambda x: options.eval_str_list(x, int),
                            help='list of kernel size (default: None)')
        parser.add_argument('--language-embeddings', action='store_true',
                            help='use language embeddings')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()

            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    'The joint_attention model requires --encoder-embed-dim to match --decoder-embed-dim')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = TranformerXLEncoder(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        decoder = TranformerXLDecoder(args, tgt_dict, decoder_embed_tokens, left_pad=args.left_pad_target)
        return TransformerXLTranslationModel(encoder, decoder)


class TranformerXLEncoder(FairseqEncoder):
    """
    JointAttention encoder is used only to compute the source embeddings.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool): whether the input is left-padded
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        # in our case, should be replaced by the other embeddings(Christine)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None
        self.embed_language = LanguageEmbedding(embed_dim) if args.language_embeddings else None

        self.register_buffer('version', torch.Tensor([2]))

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
        Returns:
            dict:
                - **encoder_out** (Tensor): embedding output of shape
                  `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)

        # this should be edited by transformerXL (Christine)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)

        # language embedding
        if self.embed_language is not None:
            lang_emb = self.embed_scale * self.embed_language.view(1, 1, -1)
            x += lang_emb
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        #????? (Christine)
        #why???
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TranformerXLDecoder(FairseqIncrementalDecoder):
    """
    JointAttention decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`ProtectedTransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.kernel_size_list = args.kernel_size_list

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)

        # if the input embed dim coming from encoder is (Christine)
        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        #Carlos 25-2-2020
        '''
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None
        '''
        self.embed_language = LanguageEmbedding(embed_dim) if args.language_embeddings else None

        # will we leave this I guess? (Christine)
        self.layers = nn.ModuleList([])
        # so the new transformer will replace this (Christine)
        # how many layers are we going to extend
        self.layers.extend([
            #we removed all the parameters , eplace it by args and then in other forward functions, we should take care of this
            #Carlos 25-2-2020
            RelPartialLearnableDecoderLayer(args)
            for _ in range(args.decoder_layers)
        ])
        # will we leave this I guess? (Christine)
        #this is the layer just for the ouput
        self.project_out_dim = Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        #???? (Christine)
        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))

        # will we leave this I guess? because we need this normalization(Christine)
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)


        # initializing these embeddings for the relative embeddings
        #we take these from args Carlos 25-2-2020
        self.embed_positions = PositionalEmbedding(embed_dim)
        #change parameters
        n_heads=args.decoder_attention_heads
        d_heads=embed_dim/n_heads
        self.r_w_bias = nn.Parameter(torch.Tensor(n_heads, d_heads))
        self.r_r_bias = nn.Parameter(torch.Tensor(n_heads, d_heads))
        dropout = args.attention_dropout
        self.drop = nn.Dropout(dropout)
        #we change this and later remove

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        # I think here we should remove and add indices as suggested (2-3-2020)

        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.

        # keep the hiddens of the current index only
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None,*mems):
        """
        Args:
            input (dict): with
                prev_output_tokens (LongTensor): previous decoder outputs of shape
                    `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        # I think the mems should be initialized and updated in the forward of the jointTransformer
        if not mems: mems = self.init_mems()
        tgt_len = prev_output_tokens.size(1)
        #check to know 26-2-2020
        embedding_dim = prev_output_tokens.size(2)

        # embed positions
        ## will be edited with the new positional embeddings ? (Christine)
        positions = self.embed_positions(
            embedding_dim
        ) if self.embed_positions is not None else None

        # what is incremental state? (Christine)
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        #forget about in our case (Carlos)
        if positions is not None:
            x += positions

        # language embedding
        if self.embed_language is not None:
            lang_emb = self.embed_scale * self.embed_language.view(1, 1, -1)
            x += lang_emb
        #here x is the target
        x = F.dropout(x, p=self.dropout, training=self.training)
        source = encoder_out['encoder_out']
        #concat source with x for implementing the new approach Carlos 25-2-2020
        concat_input_output = torch.cat(source, x, dim=1)
        #checkk here if the memory is the same shape
        concat_with_memory =torch.cat(mems, concat_input_output, dim=1)


        # B x T x C -> T x B x C
        # ????
        x = x.transpose(0, 1)
        attn = None
        # hidden?
        # inner states should take the whole concat with memory I guess (26-2-2020)
        inner_states = [x]

        process_source = incremental_state is None or len(incremental_state) == 0

        # stopping here Carlos 25-2-2020
        # compute target mask the way Adrian is doing


        # extended padding mask
        # this is the padding mask, Carlos was speaking about? (Christine)
        #not used yet 26-2-2020
        source_padding_mask = encoder_out['encoder_padding_mask']
        if source_padding_mask is not None:
            target_padding_mask = source_padding_mask.new_zeros((source_padding_mask.size(0), tgt_len))
            self_attn_padding_mask = torch.cat((source_padding_mask, target_padding_mask), dim=1)
        else:
            self_attn_padding_mask = None

        # transformer layers
        # why are these target masking
        # I think these layers should change
        #transpose to be going with transformerxl shapes
        concat_input_output=concat_input_output.transpose(0,1)
        mlen = mems[0].size(0) if mems is not None else 0
        # the length of the current batch
        #qlen..ength of src and trg in our case ...or only one of them 26-2-2020
        qlen = concat_input_output.size(0)
        klen = mlen + qlen
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pos_seq = torch.arange(klen - 1, -1, -1.0, device=device,
                               dtype=concat_with_memory.dtype)

        # po0s_emb is the positional embedding of the sequence, R in the paper
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(concat_input_output)
        pos_emb = self.drop(pos_emb)

        for i, layer in enumerate(self.layers):

            if self.kernel_size_list is not None:
                target_mask = self.local_mask(x, self.kernel_size_list[i], causal=True, tgt_len=tgt_len)
            elif incremental_state is None:
                target_mask = self.buffered_future_mask(x)
            else:
                target_mask = None

            if target_mask is not None:
                zero_mask = target_mask.new_zeros((target_mask.size(0), source.size(0)))
                self_attn_mask = torch.cat((zero_mask, target_mask), dim=1)
            else:
                self_attn_mask = None
            '''
            state = incremental_state
            if process_source:
                if state is None:
                    state = {}
                if self.kernel_size_list is not None:
                    source_mask = self.local_mask(source, self.kernel_size_list[i], causal=False)
                else:
                    source_mask = None
                source, attn = layer(
                    source,
                    None,
                    None,
                    state,
                    self_attn_mask=source_mask,
                    self_attn_padding_mask=source_padding_mask
                )
                inner_states.append(source)

            x, attn = layer(
                x,
                None,
                None,
                state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask
            )
            '''

            mems_i = None if mems is None else mems[i]
            # here it goes to the PartialRelLeranablelayer which we have above
            core_out = layer(core_out, pos_emb, self.r_w_bias,
                             self.r_r_bias, dec_attn_mask=self_attn_mask, mems=mems_i)
            #hids.append(core_out)
            inner_states.append(core_out)
        core_out = self.drop(core_out)

        new_mems = self._update_mems(inner_states, mems, mlen, qlen)

        #should I adapt?? Christine 26-2-2020
        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        # project back to size of vocabulary
        if self.share_input_output_embed:
            x = F.linear(x, self.embed_tokens.weight)
        else:
            x = F.linear(x, self.embed_out)

        pred = x
        info = {'attn': attn, 'inner_states': inner_states}

        return pred, info

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        """Cached future mask."""
        dim = tensor.size(0)
        #pylint: disable=access-member-before-definition, attribute-defined-outside-init
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def local_mask(self, tensor, kernel_size, causal, tgt_len=None):
        """Locality constraint mask."""
        rows = tensor.size(0)
        cols = tensor.size(0) if tgt_len is None else tgt_len
        if causal:
            if rows == 1:
                mask = utils.fill_with_neg_inf(tensor.new(1, cols))
                mask[0, -kernel_size:] = 0
                return mask
            else:
                diag_u, diag_l = 1, kernel_size
        else:
            diag_u, diag_l = ((kernel_size + 1) // 2, (kernel_size + 1) // 2) if kernel_size % 2 == 1 \
                else (kernel_size // 2, kernel_size // 2 + 1)
        mask1 = torch.triu(utils.fill_with_neg_inf(tensor.new(rows, cols)), diag_u)
        mask2 = torch.tril(utils.fill_with_neg_inf(tensor.new(rows, cols)), -diag_l)

        return mask1 + mask2


# Adapted from fairseq/model/transformer.py to use ProtectedMultiheadAttention
class ProtectedTransformerDecoderLayer(nn.Module):
    """Decoder layer block.
    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = ProtectedMultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            # we can encapsulate the  transformer xl attention here and leave th rest as it is (Carlos)
            self.encoder_attn = ProtectedMultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state,
                prev_self_attn_state=None, prev_attn_state=None, self_attn_mask=None,
                self_attn_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LanguageEmbedding(embedding_dim):
    m = nn.Parameter(torch.Tensor(embedding_dim))
    nn.init.normal_(m, mean=0, std=embedding_dim ** -0.5)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


@register_model_architecture('joint_attention', 'joint_attention')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)

    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_layers = getattr(args, 'decoder_layers', 14)

    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.kernel_size_list = getattr(args, 'kernel_size_list', None)
    assert args.kernel_size_list is None or len(args.kernel_size_list) == args.decoder_layers, "kernel_size_list doesn't match decoder_layers"
    args.language_embeddings = getattr(args, 'language_embeddings', True)


@register_model_architecture('joint_attention', 'joint_attention_iwslt_de_en')
def joint_attention_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('joint_attention', 'local_joint_attention_iwslt_de_en')
def local_joint_attention_iwslt_de_en(args):
    args.kernel_size_list = getattr(args, 'kernel_size_list', [3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 29, 33, 37, 41])
    joint_attention_iwslt_de_en(args)


@register_model_architecture('joint_attention', 'joint_attention_wmt_en_de')
def joint_attention_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture('joint_attention', 'joint_attention_wmt_en_de_big')
def joint_attention_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('joint_attention', 'local_joint_attention_wmt_en_de_big')
def local_joint_attention_wmt_en_de_big(args):
    args.kernel_size_list = getattr(args, 'kernel_size_list', [7, 15, 31, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63])
    joint_attention_wmt_en_de_big(args)


@register_model_architecture('joint_attention', 'joint_attention_wmt_en_fr_big')
def joint_attention_wmt_en_fr_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    joint_attention_wmt_en_de_big(args)


@register_model_architecture('joint_attention', 'local_joint_attention_wmt_en_fr_big')
def local_joint_attention_wmt_en_fr_big(args):
    args.kernel_size_list = getattr(args, 'kernel_size_list', [7, 15, 31, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63])
    joint_attention_wmt_en_fr_big(args)