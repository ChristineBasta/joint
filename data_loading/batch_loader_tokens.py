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

class BatchTokenLoader(torchtext.data.Iterator):
    def __init__(self, data_reader_object, batch_size):
        # this dictionary should contain each documents with its sentences
        self.data_reader_object = data_reader_object  # object of dictionary

        self.batch = []  # should return tensor, but for now just sentences
        self.random_shuffler = utils.RandomShuffler()
        self.batch_size = batch_size
        self.total_documents_batched = []
        self.documents_in_prevbatch = {}
        self.sentence_count = 0
        self.batch_changed=False
        self.start_index_change=-1


    def get_documents_size(self):
        #print(self.data_reader_object.dictionary_sentences)
        return len(self.data_reader_object.dictionary_sentences)


    def get_batch_i(self, i):
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
            for k, v in prev_items.items():
                # document has still sentences
                if (v < (len(self.data_reader_object.dictionary_tokens_of_sentences[k]) - 1)):
                    # load next sentence
                    self.batch.append(self.data_reader_object.dictionary_tokens_of_sentences[k][v + 1])
                    self.documents_in_prevbatch[k] = v + 1
                    # already in documents_batched_so_no_need_to_add
                    self.sentence_count += 1
                else:  # this document finishes
                    del (self.documents_in_prevbatch[k])
                    #should keep this indices too for later

            # SHOULD REMOVE FROM MEMORY OF HIDDEN LAYERS THE INDEX OF THIS BATCH

            # start of changing index is the size of batch-1 before filling
            if self.indices_to_fill() > 0:
                # we want to keep track of which indices changed, so from this till the end of the batch
                self.batch_changed = True
                self.start_index_change = len(self.batch)

                random_documents = self.pick_random_documents(range(1, (self.get_documents_size() + 1)),
                                                              self.indices_to_fill())
                print(random_documents)

                # handling case if documents are finished
                if (random_documents):
                    self.fill_batch_with_random_documents(random_documents)

            # there is a case that the sentences are less than a batch size but the documents finished
            # make sure documents are finished also
            #batch_size is different..will this make a problem

    # this should randomize documents, we are getting in the batch
    # range_shuffle: the documents we need to shuffle
    # no_documents_needed: to randomize when a document is finished
    def pick_random_documents(self, range_shuffle, no_documents_needed):

        random_documents = []
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
                if(batch_item.size() > max_size):
                    max_size=batch_item.size()
                return max_size

    #get the masking too
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
            tensor=torch.nn.utils.rnn.pad_sequence((sequences), batch_first=True,padding_value=1)
            print(tensor)
        return tensor


    # to fill random documents in case of first fill or when documents just finish and we need to replace them
    def fill_batch_with_random_documents(self, random_documents):
        for index_random_document in random_documents:
            # add the index of dictionary
            self.sentence_count += 1
            self.batch.append(self.data_reader_object.dictionary_tokens_of_sentences[index_random_document][0])

            # should be ordered like the order of the batch
            self.documents_in_prevbatch[index_random_document] = 0
            self.total_documents_batched.append(index_random_document)
            self.sentence_count += 1

            # check if we reached batch size
            if (len(self.batch) > self.batch_size):
                break

    # fill the target batches with the same here
    # data_reader_trg_object: should be the target reader with its dictionary of sentences and tokens
    def fill_batch_target(self, data_reader_trg_object):
        trg_batch = []
        for k, v in self.documents_in_prevbatch.items():
            trg_batch.append(data_reader_trg_object.dictionary_tokens_of_sentences[k][v])
        return self.add_padding_as_rnn(trg_batch)



if __name__ == "__main__":



    # test different batch sizes
    # test different loops
    # test different document sizes

    path_src = '/home/christine/Phd/Cristina_cooperation/joint/corpus_trial.en'
    src_dict = Dictionary.load(
        '/home/christine/Phd/Cristina_cooperation/joint/data-bin/iwslt14.joined-dictionary.31K.de-en/dict.en.txt')

    sourceTextReader = DataRawTextReader(path_src, src_dict)
    #print(sourceTextReader.dictionary_tokens_of_sentences)
    #print(sourceTextReader.dictionary_sentences)

    batchLoader = BatchTokenLoader(sourceTextReader, 6)
    # load till we finish( agree on criteria)


    for i in range(20):
        batchLoader.get_batch_i(i)
        print(batchLoader.batch)
        # we have to load the same batch for the other language
        print(batchLoader.documents_in_prevbatch)
        print(batchLoader.total_documents_batched)
        print('paddings:')
        batchLoader.add_padding_(batchLoader.batch)

