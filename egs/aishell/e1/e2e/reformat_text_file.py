# Add a space between each character in the transcript. This is necessary because
# we are considering each character as a word. 

import sys
import os

if len(sys.argv) != 3:
    print("usage: python xxx.py <input_text_file_path> <output_text_file_path>")
    exit(1)

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

if __name__ == "__main__":
    text_result=[]
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f):
            splits = line.split()
            utt_id = splits[0]
            sentence = ''.join(splits[1:])
            sentence = ' '.join(sentence)
            #print(utt_id)
            #print(sentence)
            #print(len(sentence))
            output = utt_id +' '+sentence+'\n'
            text_result.append(output)
            #text_result.append("%s\s%s\n"%(utt_id, sentence))


    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.writelines(text_result)
