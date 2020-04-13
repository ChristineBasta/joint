
import argparse
import sys


tags=['< speaker >', '< talkid >', '< keywords >', '< url >', '< reviewer >', '< translator >', '< description >']
#add src and traget tokens to lines of the
#have to set is_srcto False for trg, and True for src
def add_token(file_to_read, file_to_write, is_src):
        file_src = open(file_to_write, 'w',encoding='utf8')
        first_line_written=False
        with open(file_to_read,'r', encoding='utf8') as f:
            for line in f:
                #print(line)
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
                    #print('**')




def  has_tag(line, tags):
    has_tag = False
    for tag in tags:
        if tag in line:
            has_tag = True
    return has_tag



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", help="The file of the source language")
    parser.add_argument("--src_file_prepared", help="The file of the source language after preparation")
    parser.add_argument("--trg_file", help="The file of the target language")
    parser.add_argument("--trg_file_prepared", help="The file of the source language after preparation")


    args = parser.parse_args()

    src_file = args.src_file
    src_file_prepared = args.src_file_prepared
    trg_file = args.trg_file
    trg_file_prepared = args.trg_file_prepared




    #source adding <src>
    add_token(src_file,src_file_prepared, True)
    #target adding <trg>
    add_token(trg_file, trg_file_prepared, False)
