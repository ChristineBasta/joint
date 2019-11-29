import fairseq.data.iterators
import numpy as np
import torch
import fairseq.data.data_utils as data_utils

class EpochBatchIterator(fairseq.EpochBatchIterating):
    """A multi-epoch iterator over a :class:`torch.utils.data.Dataset`.

    Compared to :class:`torch.utils.data.DataLoader`, this iterator:

    - can be reused across multiple epochs with the :func:`next_epoch_itr`
      method (optionally shuffled between epochs)
    - can be serialized/deserialized with the :func:`state_dict` and
      :func:`load_state_dict` methods
    - supports sharding with the *num_shards* and *shard_id* arguments

    Args:
        dataset (~torch.utils.data.Dataset): dataset from which to load the data
        collate_fn (callable): merges a list of samples to form a mini-batch
        batch_sampler (~torch.utils.data.Sampler): an iterator over batches of
            indices
        seed (int, optional): seed for random number generator for
            reproducibility (default: 1).
        num_shards (int, optional): shard the data iterator into N
            shards (default: 1).
        shard_id (int, optional): which shard of the data iterator to
            return (default: 0).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means the data will be loaded in the main process
            (default: 0).
        epoch (int, optional): the epoch to start the iterator from
            (default: 0).
    """

    def __init__(
        self, dataset, collate_fn, batch_sampler, seed=1, num_shards=1, shard_id=0,
        num_workers=0, epoch=0,
    ):
        assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.collate_fn = collate_fn
        #most probably we do not have frozen..because  we randomize things
        self.frozen_batches = tuple(batch_sampler)
        self.seed = seed
        #how this is supposed to reaplace for us
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.num_workers = num_workers


        self.epoch = epoch
        self._cur_epoch_itr = None
        self._next_epoch_itr = None
        #?????????
        self._supports_prefetch = getattr(dataset, 'supports_prefetch', False)

    def __len__(self):
        return len(self.frozen_batches)

    # shuffle should be false for all methods, make sure of this??? Christ

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus: ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
        """
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self.epoch += 1
            #so here the iterator is moving to the next...so should take care of this
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch, shuffle, fix_batches_to_gpus=fix_batches_to_gpus,
            )
        self.dataset.set_epoch(self.epoch)
        return self._cur_epoch_itr

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        """The number of consumed batches in the current epoch."""
        #I do not quite get what this is
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.count
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.count
        return 0

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        return {
            'epoch': self.epoch,
            'iterations_in_epoch': self.iterations_in_epoch,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict['epoch']
        itr_pos = state_dict.get('iterations_in_epoch', 0)
        if itr_pos > 0:
            # fast-forward epoch iterator
            self._next_epoch_itr = self._get_iterator_for_epoch(
                self.epoch,
                shuffle=state_dict.get('shuffle', True),
                offset=itr_pos,
            )

    def _get_iterator_for_epoch(self, epoch, shuffle, fix_batches_to_gpus=False, offset=0):

        def shuffle_batches(batches, seed):
            # set seed based on the seed and epoch number so that we get
            # reproducible results when resuming from checkpoints
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
            return batches

        if self._supports_prefetch:
            batches = self.frozen_batches

            if shuffle and not fix_batches_to_gpus:
                batches = shuffle_batches(list(batches), self.seed + epoch)

            batches = list(ShardedIterator(
                batches, self.num_shards, self.shard_id, fill_value=[]
            ))
            self.dataset.prefetch([i for s in batches for i in s])

            if shuffle and fix_batches_to_gpus:
                batches = shuffle_batches(batches, self.seed + epoch + self.shard_id)
        else:
            if shuffle:
                batches = shuffle_batches(list(self.frozen_batches), self.seed + epoch)
            else:
                batches = self.frozen_batches
            batches = list(ShardedIterator(
                batches, self.num_shards, self.shard_id, fill_value=[]
            ))

        if offset > 0 and offset >= len(batches):
            return None

        return CountingIterator(
            torch.utils.data.DataLoader(
                self.dataset,
                collate_fn=self.collate_fn,
                batch_sampler=batches[offset:],
                num_workers=self.num_workers,
            ),
            start=offset,
        )

