
from fairseq.data import FairseqDataset
import fairseq.data.data_utils as data_utils
from data_loading.batch_loader_tokens import BatchTokenLoader
import numpy as np
import torch


# it is implemented and changed but methods should be tested individually ( 22-12-2019)
#  make sure that it gets what we wants when we run it using the task
def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
    ):
    if len(samples) == 0:
        return {}
    #padding method that we have for the batches we have
    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])


    src_tokens = merge('source', left_pad=left_pad_source)

    # sort by descending source length
    # the problem if it was reordering and the batches do not see consequent sentences
    # we should remove then
    # The next four sentences commented by Christine (18-12-2019)
    #need only this to send the parameters, but we are supposed not to need them Christine (27-2-2020)
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    #src_lengths, sort_order = src_lengths.sort(descending=True)
    #id = id.index_select(0, sort_order)
    #src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None

    #if there is a target
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        # order according to the order to src_tokens
        # next sentence commented by Christine (18-12-2019)
        # target = target.index_select(0, sort_order)

        ntokens = sum(len(s['target']) for s in samples)
        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            # order according to the order to src_tokens
            # next sentence commented by Christine (18-12-2019)
            #prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    #
    # need to revise if ntokens are got right, and make sure what it affects (18-12-2019)
    # do we need to change anything to this if the batch is already the same dimensions
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },

        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=False, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
        batch_size=8
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()

        #our src and target
        self.src = src
        self.tgt = tgt


        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None

        #the dictionary of Fairseq
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict



        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target

        #I guess this is the max tokens
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions

        #always set this to False
        self.shuffle = shuffle
        #??
        self.input_feeding = input_feeding

        # because eos is not important in source but it is important to stop suggesting in target
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.batch_size= batch_size

        self.batchLoader = BatchTokenLoader(self.src, self.batch_size)
    #should be reimplemented
    #should I get the batch here or where
    def __getitem__(self, index):



        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]

        #get a sentence when given index..but should be given a document too


        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]



        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shapeº `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    # we need to be sure that this goes with the concept of Fairseq
    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        indices= self.batchLoader.ordered_indices()

        return indices

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)




if __name__ == "__main__":
    dataset= LanguagePairDataset()