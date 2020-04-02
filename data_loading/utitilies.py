


tags=['< speaker >', '< talkid >', '< keywords >', '< url >', '< reviewer >', '< translator >', '< description >']
#add src and traget tokens to lines of the
#have to set is_srcto False for trg, and True for src
def add_token(file_to_read, file_to_write, is_src):
        file_src = open(file_to_write, 'w')
        first_line_written=False
        with open(file_to_read) as f:
            for line in f:
                print(line)
                if '< title >' not in line and  not has_tag(line, tags):
                    if is_src:
                        if '\n' in line:
                            file_src.write('<src> '+line)
                        else:
                            file_src.write('<src> ' + line+'\n')
                        first_line_written=True
                    else:
                        if '\n' in line:
                            file_src.write('<trg> ' + line)
                        else:
                            file_src.write('<trg> ' + line + '\n')
                        first_line_written = True
                elif  not has_tag(line, tags) and first_line_written: #has title
                    if '\n' in line:
                        line=line.replace("< title >", "<title>")
                        line = line.replace("< / title >", "</title>")
                        file_src.write(line)
                    else:
                        file_src.write(line+'\n')
                    print('**')




def  has_tag(line, tags):
    has_tag = False
    for tag in tags:
        if tag in line:
            has_tag = True
    return has_tag



add_token('../data-bin/Europarl/train.tags.de-en.tok.tc.clean.de', '../data-bin/Europarl/train.tags.de-en.tok.tc.clean.tagged.de', False)
