import torchtext.data.utils as utils
import torchtext
from data_loading.data_loader import Dictionary
import copy


# once we have the documents loading in batch..we have to make sure that the same document is used till
# Its sentences are finished
# make sure that lall documents are loaded



class BatchLoader(torchtext.data.Iterator):
    def __init__(self, dict_object, batch_size):
        # this dictionary should contain each documents with its sentences
        self.dict_object = dict_object  # object of dictionary

        self.batch = []  # should return tensor, but for now just sentences
        self.random_shuffler = utils.RandomShuffler()
        self.batch_size = batch_size
        self.total_documents_batched = []
        self.documents_in_prevbatch = {}

    # we should keep track of documents we are loading
    # we should keep track of documents that ends and replace it with documents we are loading
    # we should keep

    def get_batch_i(self, i):
        # should I empty the batch each batch?
        self.batch = []
        sentence_count = 0
        # first batch

        if not self.total_documents_batched:
            random_documents = self.pick_random_documents(range(1, (len(self.dict_object.dictionary_count) + 1)),
                                                          self.batch_size)
            # print(random_documents)
            for index_random_document in random_documents:
                # add the index of dictionary
                # print(index_random_document)
                # print(self.dict_object.dictionary[index_random_document])
                self.batch.append(self.dict_object.dictionary[index_random_document][0])
                # should be ordered like the order of the batch
                self.documents_in_prevbatch[index_random_document] = 0
                self.total_documents_batched.append(index_random_document)
                sentence_count += 1
                # check if the batch size
                if (len(self.batch) > self.batch_size):
                    break



        # next batches, not the first
        else:
            # v is index
            prev_items = copy.deepcopy(self.documents_in_prevbatch)
            print(prev_items)
            print(self.total_documents_batched)
            for k, v in prev_items.items():
                # document has still sentences
                if (v < (len(self.dict_object.dictionary[k]) - 1)):
                    # load next sentence
                    self.batch.append(self.dict_object.dictionary[k][v + 1])
                    self.documents_in_prevbatch[k] = v + 1
                    # already in documents_batched_so_no_need_to_add
                    sentence_count += 1
                else:
                    del (self.documents_in_prevbatch[k])
                    # SHOULD REMOVE FROM MEMORY OF HIDDEN LAYERS THE INDEX OF THIS BATCH
                    #
                    random_document = self.pick_random_documents(range(1, (len(self.dict_object.dictionary_count) + 1)),
                                                                 1)
                    print(random_document)
                    if (random_document):
                        new_index = random_document[0]
                        self.documents_in_prevbatch[new_index] = 0
                        self.batch.append(self.dict_object.dictionary[new_index][0])
                        self.total_documents_batched.append(new_index)
                        sentence_count += 1
            # there is a case that the sentences are less than a batch size but the documents finishes
            # handling case that the sentences are coming out from these stops less than batch size
            #handling case if documents are finished

    # this should randomize documents, we are getting in the batch
    # range_shuffle: the documents we need to shuffle
    # no_documents_needed: to randomize when a document is finished
    def pick_random_documents(self, range_shuffle, no_documents_needed):
        # range_shuffle=range(len([1, 5, 6, 7]))
        random_documents = []
        shuffler_index = self.random_shuffler(range_shuffle)
        for i in shuffler_index:
            if (i not in self.total_documents_batched):
                random_documents.append(i)
            if (len(random_documents) >= no_documents_needed):
                break
        return random_documents

    def add_padding(self):
        return ''

    # test


if __name__ == "__main__":
    path = '/home/christine/Phd/Cristina_cooperation/joint/corpus_trial.en'
    dict_object = Dictionary(path)
    dict_object.load_dictonary()
    print(dict_object.dictionary)
# test different batch sizes
# test different loops
# test different document sizes

    batchLoader = BatchLoader(dict_object, 4)

    for i in range(6):
        batchLoader.get_batch_i(i)
        print(batchLoader.batch)
