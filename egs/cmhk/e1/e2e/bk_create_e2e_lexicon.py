# This script reads the training transcript and write the nonsilence_phones.txt and lexicon.txt in the given
# dict directory. 

import sys
import os

if len(sys.argv) != 3:
    print("usage: python xxx.py <input_text_file_path> <dict_path>")
    exit(1)

input_file_path = sys.argv[1]
dict_path = sys.argv[2]

if __name__ == "__main__":
    non_sil=set()
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f):
            splits = line.split()
            utt_id = splits[0]
            sentence = ''.join(splits[1:])
          
            for val in sentence:
                non_sil.add(val.upper()+'\n')
            #print(utt_id)
            #print(sentence)
            #print(len(sentence))
            #text_result.append("%s\t%s\n"%(utt_id, sentence))


    with open(dict_path+'nonsilence_phones.txt', 'w', encoding='utf-8') as f:
        f.writelines(sorted(list(non_sil)))

    lexicon_list=[]
    #append sil phones
    lexicon_list.append('SIL SIL\n')
    lexicon_list.append('<NON/> NSN\n')
    lexicon_list.append('<UNK> SPN\n')
    lexicon_list.append('<SPK/> SPN\n')

    for val in sorted(list(non_sil)):
        val = val.rstrip()
        lexicon_list.append(val+' '+val+'\n')

    with open(dict_path+'lexicon.txt', 'w', encoding='utf-8') as f:
        f.writelines(lexicon_list)
