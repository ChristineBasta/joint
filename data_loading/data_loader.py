import torch
import sys, re
import pickle
import torchtext.data.utils as utils

import fairseq.tokenizer
import fairseq.data.indexed_dataset
#the document file should contain a tag for the start of document
class Dictionary():
    def __init__(self, path):

        self.path = path
        self.dictionary= {}
        self.dictionary_count={}
    # this method will load the documents in dictionary
    # dictionary should look like dict={1:[s1,s2,s3], 2: [s1,s2,s3,s4]}
    # where the number refers to the document number
    # and s1, s2, s3 refer to sentences 1, 2, 3 of the document respectively
    def load_dictonary(self):
        self.read_as_sentences()


    def read_as_tokens(self):
        file_to_read = open(self.path, 'r')
        count_lines = 0
        count_documents = 0
        document_sentences = []
        for ls in file_to_read:

            # this is a special handling, should not be like this
            if ls.startswith("<title>"):

                # if the target file differs from the source, should be handled later

                # if not lt.startswith("<title>"):
                #    print("error " + str(count))
                #    break

                ls = re.sub("(^\<title\>)(.*)(\</title\>)", "\g<2>", ls).strip()
                print(ls)
                # lt = re.sub("(^\<title\>)(.*)(\</title\>)", "\g<2>", lt).strip()

                # print(str(count_lines) + "\n")
                # print()

                # f_s_doc.write(ls + "\n")
                # f_t_doc.write(lt + "\n")

                count_documents += 1
                self.dictionary[count_documents] = document_sentences
                self.dictionary_count[count_documents] = count_lines
                document_sentences = []

            elif not ls.startswith("<"):
                # if ls.strip() != "" and lt.strip() != "":
                if ls.strip() != "":
                    # f_s_o.write(ls.strip() + "\n")
                    # f_t_o.write(lt.strip() + "\n")
                    # print(ls.strip())
                    count_lines += 1
                    document_sentences.append(ls.strip())

    def read_as_sentences(self):
        file_to_read = open(self.path, 'r')
        count_lines = 0
        count_documents = 0
        document_sentences = []
        for ls in file_to_read:

            # this is a special handling, should not be like this
            if ls.startswith("<title>"):

                # if the target file differs from the source, should be handled later

                # if not lt.startswith("<title>"):
                #    print("error " + str(count))
                #    break

                ls = re.sub("(^\<title\>)(.*)(\</title\>)", "\g<2>", ls).strip()
                print(ls)
                # lt = re.sub("(^\<title\>)(.*)(\</title\>)", "\g<2>", lt).strip()

                # print(str(count_lines) + "\n")
                # print()

                # f_s_doc.write(ls + "\n")
                # f_t_doc.write(lt + "\n")

                count_documents += 1
                self.dictionary[count_documents] = document_sentences
                self.dictionary_count[count_documents] = count_lines
                document_sentences = []

            elif not ls.startswith("<"):
                # if ls.strip() != "" and lt.strip() != "":
                if ls.strip() != "":
                    # f_s_o.write(ls.strip() + "\n")
                    # f_t_o.write(lt.strip() + "\n")
                    # print(ls.strip())
                    count_lines += 1
                    document_sentences.append(ls.strip())
    def save_obj(self, obj, name ):
        with open('../obj/'+ name + '.pkl', 'wb+') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name ):
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)



if __name__ == "__main__":
    path='/home/christine/Downloads/de-en/train.tags.de-en.en'

    dict= Dictionary(path)
    dict.load_dictonary()
    #dict.save_obj(dict.dictionary,'corpus_src')
    print(dict.dictionary)
    print(dict.dictionary[1])
    print(dict.dictionary[5])
    print(dict.dictionary[1360])
    print(dict.dictionary[1361])
    print(dict.dictionary_count)

