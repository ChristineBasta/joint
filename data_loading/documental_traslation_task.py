# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import copy
from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset)

from fairseq.tasks import FairseqTask, register_task
from data_loading.data_reader import DataRawTextReader
from data_loading.documental_dataset import LanguagePairDataset
import numpy as np

#the batch_by_size is tested but make sure it goes with the other parts well
#maybe others methods need a special test or handling (22-12-2019)


#max_Sentences only...because in our case the max tokens will be commented all the way to make sure thatw
def _is_batch_full( batch,  max_sentences):
    if len(batch) == 0:
        return 0
    if max_sentences > 0 and len(batch) == max_sentences:
        return 1
    #if max_tokens > 0 and num_tokens > max_tokens:
    #    return 1
    return 0

#taken from data_utils
# max_tokens inm our case should not be added
def batch_by_size(
        indices, max_sentences=24
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.
    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """

    max_tokens = max_tokens if max_tokens is not None else -1
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    return batch_by_size_fast(indices, max_sentences)

#remove all related to number of tokens and sample length as we donot need them
def batch_by_size_fast(indices, max_sentences):

    batch = []
    batches = []

    idx=0
    indices_view = copy.deepcopy(indices)

    for i in range(len(indices_view)):
        idx = indices_view[i]

        #num_tokens = num_tokens_fn(idx)

        #sample_lens.append(num_tokens)
        #sample_len = maxmax_sentences(sample_len, num_tokens)

        #assert max_tokens <= 0 or sample_len <= max_tokens, (
        #    "sentence at index {} of size {} exceeds max_tokens "
        #    "limit of {}!".format(idx, sample_len, max_tokens)
        #)
        #num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, max_sentences):
            #mod_len = max(
            #    bsz_mult * (len(batch) // bsz_mult),
            #    len(batch) % bsz_mult,
            #)
            batch_size=max_sentences
            #batches.append(batch[:mod_len])
            #batch = batch[mod_len:]
            batches.append(batch[:batch_size])
            batch = batch[batch_size:]
            #sample_lens = sample_lens[mod_len:]
            #sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches

# should specify the batch_size here
def load_langpair_dataset(data_path, split,
                          src, src_dict,
                          tgt, tgt_dict,
                          combine, dataset_impl, upsample_primary,
                          left_pad_source, left_pad_target, max_source_positions, max_target_positions,
):
    #check if the file exists, nothing more
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []

    tgt_datasets = []

    #to get the file prefix for each split, train, valid or test
    #check thus
    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        textReader_src = DataRawTextReader(prefix + src, src_dict)
        src_dataset_tokens= textReader_src.dictionary_tokens_of_sentences
        src_dataset_sentences= textReader_src.dictionary_sentences

        textReader_trg = DataRawTextReader(prefix + tgt, tgt_dict)
        trg_dataset_tokens = textReader_trg.dictionary_tokens_of_sentences
        trg_dataset_tokens = textReader_trg.dictionary_sentences


        #our datasets should be uplaoded here
        src_datasets.append(textReader_src)
        tgt_datasets.append(textReader_trg)

        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    #not sure about this part (Christine)
    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        #should I have to comment this? (Christine)
        #related to use more than one siurce of data
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
    )


@register_task('translation_transformer')
class TranslationTransformerTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically

        #implement again this part
        #data_utils.infer_language_pair dets names for trg and src for example src= en and trg=de
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        #so we loaded it from the dataset
        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    # implement this method...a3ml fiha eh.-.elmafrood eh
    #ake sure that src_tokens and src_lengths
    #what deos this exactly do??? (Christine)
    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    #make sure we do not need to add anthing  (Christine)
    # if this is what determines the batch size..so we have to control it..right? (Christine)
    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    # to be implemented
    def get_batch_iterator(
            self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
            ignore_invalid_inputs=False, required_batch_size_multiple=1,
            seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
            (default: None). (we do not use it anymore and must ensure that everywhere)
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None). (we do not use it anymore and must ensure that everywhere)
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
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

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        assert isinstance(dataset, FairseqDataset)

        # get indices ordered by example size
        #this should be already fixed in the dataset
        with data_utils.numpy_seed(seed):
            #will get our ordered_indices
            #will be filtering by our size
            indices = dataset.ordered_indices()

        # filter_by_size was removed as we believe we do not need it (Christine)(18-12-2019)

        # create mini-batches which has batches with given size constraints
        # it just adjusts the batch by size (works by batch_by_size implemented above)
        # should be tested later if it suits the framework(Christine)(18-12-2019)
        batch_sampler =batch_by_size(
            indices, max_sentences=max_sentences
        )

        # batches should be here returned correctly, mini batches should be ???
        # return a reusable, sharded iterator
        return iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )



