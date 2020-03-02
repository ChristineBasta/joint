import torchtext.data.utils as utils
import torchtext
from data_loading.data_reader import DataRawTextReader
import copy
import torch
from fairseq.data import (
    Dictionary,
    data_utils
)


# once we have the documents loading in batch..we have to make sure that the same document is used till
# Its sentences are finished
# make sure that all documents are loaded
# load tokens not sentences, the same like batch_loader_tokens, just loading tokens to have  afull story

# Carlos was commenting of some small changes (Writen in the noteboook to check)..make sure they are implemented when we are back (22-12-2019)
# 1-that trg only needs eos
# 2-parse documents in 2 ways ..whatever makes it easier later
# 3-clean dictionaries before the new epoch starts
# 4-randomization of documents can work on the free ones only
# 5-make sure that u have indices that change when deleting a document and know how to inetgarte this in the new architecture
# 6- when batch sizes start to change just because the documents are beginning to finish, stop filling batches
# for now we are leaving the padding as they are (maybe a problem later)
# for now we do not know how to fill the initialization of hidden states the doc when it is entering

class BatchTokenLoader(torchtext.data.Iterator):
    def __init__(self, data_reader_object, batch_size):
        # this dictionary should contain each documents with its sentences
        self.data_reader_object = data_reader_object  # object of dictionary

        self.batch = []  # should return tensor, but for now just sentences
        self.random_shuffler = utils.RandomShuffler()
        self.batch_size = batch_size
        self.total_documents_batched = []  # should be emptied each epoch (Carlos)
        self.documents_in_prevbatch = {}
        self.sentence_count = 0
        self.batch_changed = False
        self.start_index_change = -1

    def get_documents_size(self):
        # print(self.data_reader_object.dictionary_sentences)
        return len(self.data_reader_object.dictionary_sentences)

    def get_batch_next(self):
        # should I empty the batch each batch?
        self.batch = []

        # first batch
        if not self.total_documents_batched:
            random_documents = self.pick_random_documents(range(1, (self.get_documents_size() + 1)),
                                                          self.batch_size)
            self.fill_batch_with_random_documents(random_documents)
        # next batches, not the first
        else:
            # make a copy so no change affects the loop
            prev_items = copy.deepcopy(self.documents_in_prevbatch)
            # loop on the items oif dictionary to see which document finshes and which is not
            # if a doc finishes, just remove it and then we will fill random document
            index_iterator=0
            deleted_indices=[]
            added_indices=[]
            for k, v in prev_items.items():
                # document has still sentences
                doc_length = len(self.data_reader_object.dictionary_documents_indices_sentences[k])
                first_index_doc = self.data_reader_object.dictionary_documents_indices_sentences[k][0]
                if (v < ((first_index_doc + doc_length) - 1)):
                    # load next sentence
                    self.batch.append(v + 1)
                    self.documents_in_prevbatch[k] = v + 1
                    # already in documents_batched_so_no_need_to_add
                    self.sentence_count += 1
                else:  # this document finishes
                    del (self.documents_in_prevbatch[k])
                    deleted_indices.append(index_iterator)
                index_iterator=index_iterator+1

                    # should keep this indices too for later


            # SHOULD REMOVE FROM MEMORY OF HIDDEN LAYERS THE INDEX OF THIS BATCH(Christine)

            # start of changing index is the size of batch-1 before filling
            if self.indices_to_fill() > 0:
                # we want to keep track of which indices changed, so from this till the end of the batch
                self.batch_changed = True
                self.start_index_change = len(self.batch)

                random_documents = self.pick_random_documents(range(1, (self.get_documents_size() + 1)),
                                                              self.indices_to_fill())
                added_indices.append(self.indices_to_fill())
                # print(random_documents)

                # handling case if documents are finished
                if (random_documents):
                    self.fill_batch_with_random_documents(random_documents)

            # there is a case that the sentences are less than a batch size but the documents finished
            # make sure documents are finished also
            # batch_size is different..will this make a problem
            return deleted_indices, added_indices


    # this should randomize documents, we are getting in the batch
    # range_shuffle: the documents we need to shuffle
    # no_documents_needed: to randomize when a document is finished
    def pick_random_documents(self, range_shuffle, no_documents_needed):
        # Carlos:  we can subtract total from what we have to get the free ones
        random_documents = []
        #we can randomize only above certain size to discard the high changes of memory 2-3-2020
        shuffler_index = self.random_shuffler(range_shuffle)
        for i in shuffler_index:
            if (i not in self.total_documents_batched):
                random_documents.append(i)
            if (len(random_documents) >= no_documents_needed):
                break
        return random_documents

    def get_max_size_tensors(self):
        if self.batch:
            max_size = self.batch[0].size()
            for batch_item in self.batch:
                print(batch_item.size())
                if (batch_item.size() > max_size):
                    max_size = batch_item.size()
                return max_size

    # get the masking too
    def add_padding(self, sequences):
        """
            :param sequences: list of tensors
            :return:
            """
        num = len(sequences)
        max_len = max([s.size(0) for s in sequences])
        print(max_len)

        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        mask = sequences[0].data.new(*out_dims).fill_(0)

        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1

        return out_tensor, mask

    def add_padding_as_rnn(self, sequences):
        """
            :param sequences: list of tensors
            :return:
            """
        if (sequences):
            tensor = torch.nn.utils.rnn.pad_sequence((sequences), batch_first=True, padding_value=1)
            print(tensor)
        return tensor

    # to fill random documents in case of first fill or when documents just finish and we need to replace them
    def fill_batch_with_random_documents(self, random_documents):

        for index_random_document in random_documents:
            # add the index of dictionary
            self.sentence_count += 1

            self.batch.append(self.data_reader_object.dictionary_documents_indices_sentences[index_random_document][0])

            # should be ordered like the order of the batch
            #self.documents_in_prevbatch[index_random_document] = self.data_reader_object.dictionary_documents_indices_sentences[index_random_document][0]
            self.documents_in_prevbatch[index_random_document] = \
            self.data_reader_object.dictionary_documents_indices_sentences[index_random_document][0]
            self.total_documents_batched.append(index_random_document)
            self.sentence_count += 1

            # check if we reached batch size
            if (len(self.batch) > self.batch_size):
                break

    # this is to prepare the ordered indices for the dataset
    def ordered_indices(self):
        ordered_indices = []
        deleted_indices=[]
        added_indices=[]
        # make sure that the condition is right
        while (len(ordered_indices)< len(self.data_reader_object.tokens_list)):

            deleted_indices_per_batch, added_indices_per_batch=self.get_batch_next()
            added_indices.append(added_indices_per_batch)
            deleted_indices.append(deleted_indices_per_batch)
            if(self.batch_size > len(self.batch)):
                break
            print(self.documents_in_prevbatch)
            print(self.batch)

            ordered_indices.extend(self.batch)
            print(ordered_indices)
        return ordered_indices

    # different indices to fill
    def indices_to_fill(self):
        return (self.batch_size - len(self.batch))


if __name__ == "__main__":
    # test different batch sizes
    # test different loops
    # test different document sizes

    path_src = '../corpus_trial.en'
    src_dict = Dictionary.load('../data-bin/iwslt14.joined-dictionary.31K.de-en/dict.en.txt')

    sourceTextReader = DataRawTextReader(path_src, src_dict)
    # print(sourceTextReader.dictionary_tokens_of_sentences)
    # print(sourceTextReader.dictionary_sentences)

    batchLoader = BatchTokenLoader(sourceTextReader, 4)
    # load till we finish( agree on criteria)
    batchLoader.ordered_indices()
    '''
    for i in range(20):
        batchLoader.get_batch_next()
        print(batchLoader.batch)
        # we have to load the same batch for the other language
        print(batchLoader.documents_in_prevbatch)
        print(batchLoader.total_documents_batched)
        print('paddings:')
        batchLoader.add_padding_(batchLoader.batch)
    '''
