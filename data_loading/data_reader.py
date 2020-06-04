import numpy as np
import re

import os
import torch
from fairseq.data import (
    Dictionary,
    FairseqDataset
)
#this is tested except for the sizes and method startinf from start_index
#do not forget to  make a quick test

#just make sure that this is the format of file we need

class DataRawTextReader(FairseqDataset):
    # load in a dictionary instead that List
    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.dictionary=dictionary      #DICT of FAIRSEQ

        self.dictionary_sentences = {}
        self.dictionary_tokens_of_sentences = {}
        self.dictionary_documents_indices_sentences={}

        self.count_documents = 0
        self.tokens_list = []
        self.lines = []
        self.sizes = []



        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    # any data should come in this format
    #the file should b
    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            document_sentences = []
            document_sentence_tokens = []

            for ls in f:
                #self.lines.append(line.strip('\n'))
                #print(line)


                if ls.startswith("<title>"):
                    ls = re.sub("(^\<title\>)(.*)(\</title\>)", "\g<2>", ls).strip()
                    self.count_documents += 1
                    #we can get rid later of those two dictionaries
                    range_indices = []
                    start_index = len(self.tokens_list)
                    self.dictionary_sentences[self.count_documents] = document_sentences
                    self.dictionary_tokens_of_sentences [self.count_documents]= document_sentence_tokens
                    self.tokens_list.extend(document_sentence_tokens) # to add all tokes with indices
                    self.lines.extend(document_sentences) #lines that maps to indices
                    end_index=len(self.tokens_list)
                    #get the range of indices of the
                    # Muestra 3,4,5
                    for x in range(start_index, end_index):
                        range_indices.append(x)
                    self.dictionary_documents_indices_sentences[self.count_documents]=range_indices

                    document_sentences = []
                    document_sentence_tokens = []



                elif not ls.startswith("<"):
                    if ls.strip() != "":
                        document_sentences.append(ls.strip())
                        tokens = dictionary.encode_line(
                            ls, add_if_not_exist=False,
                            append_eos=self.append_eos, reverse_order=self.reverse_order,
                        ).long()
                        document_sentence_tokens.append(tokens)

                    self.sizes.append(len(tokens)) #sizes of tokens of lines, just need to know if that is right or not
                    #print(tokens)

                    # self.tokens_list.append(tokens)

        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    # returns the list of sentences for a document
    # i should refer to the document index
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    # should adapt

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]


    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    def add_src_token (self, line_src):
        return '<src> '+line_src

    def add_trg_token(self, line_trg):
        return '<trg> ' + line_trg

if __name__ == "__main__":
    #path_src = '/home/christine/Phd/Cristina_cooperation/joint/corpus/train.tags.de-en.en'
    #path_trg = '/home/christine/Phd/Cristina_cooperation/joint/corpus/train.tags.de-en.de'

    path_src = '../corpus_trial.en'
    # load dictionaries
    # src_dict=
    src_dict = Dictionary.load(
        '../data-bin/iwslt14.joined-dictionary.31K.de-en/dict.en.txt')
    trg_dict = Dictionary.load(
        '../data-bin/iwslt14.joined-dictionary.31K.de-en/dict.de.txt')
    #tgt_dict = Dictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)))

    textReader_src = DataRawTextReader(path_src, src_dict)

    print(textReader_src.tokens_list)
    print(textReader_src.lines)
    print(textReader_src.dictionary_documents_indices_sentences)

    print(len(textReader_src.tokens_list))
    print(len(textReader_src.lines))
    print(len(textReader_src.dictionary_documents_indices_sentences))



    #textReader_trg = DataRawTextReader(path_trg, trg_dict)
    #print(textReader_trg.dictionary_tokens_of_sentences)
    #print(textReader_trg.dictionary_sentences)

    #by this we have the two dictionaries


