# This script reads the training transcript and write the nonsilence_phones.txt and lexicon.txt in the given
# dict directory. 

import sys
import os

if len(sys.argv) != 3:
    print("usage: python xxx.py <nonsilence_phones.txt> <lexicon.txt>")
    exit(1)

input_file_path = sys.argv[1]
lex_path = sys.argv[2]

if __name__ == "__main__":
    non_sil=set()
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f):
            non_sil.add(line.upper())

    lexicon_list=[]
    #append sil phones
    lexicon_list.append('SIL SIL\n')
    lexicon_list.append('<NON/> NSN\n')
    lexicon_list.append('<UNK> SPN\n')
    lexicon_list.append('<SPK/> SPN\n')

    for val in sorted(list(non_sil)):
        val = val.rstrip()
        lexicon_list.append(val+' '+val+'\n')

    with open(lex_path, 'w', encoding='utf-8') as f:
        f.writelines(lexicon_list)
