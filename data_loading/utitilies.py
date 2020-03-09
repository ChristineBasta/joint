



#add src and traget tokens to lines of the
#have to set is_srcto False for trg, and True for src
def add_token(file_to_read, file_to_write, is_src):
        file_src = open(file_to_write, 'w')
        with open(file_to_read) as f:
            for line in f:
                print(line)
                if '<title>' not in line and '<speaker>' not in line and '<talkid>' not in line and '<keywords>' not in line and '<url>' not in line:
                    if is_src:
                        if '\n' in line:
                            file_src.write('<src> '+line)
                        else:
                            file_src.write('<src> ' + line+'\n')

                    else:
                        if '\n' in line:
                            file_src.write('<trg> ' + line)
                        else:
                            file_src.write('<trg> ' + line + '\n')
                elif '<speaker>' not in line and '<talkid>' not in line and '<keywords>' not in line and '<url>' not in line:
                    if '\n' in line:
                        file_src.write(line)
                    else:
                        file_src.write(line+'\n')
                    print('**')




add_token('../data-bin/Europarl/train.tags.de-en.en', '../data-bin/Europarl/train.tags.de-en.tokened.en', True)
