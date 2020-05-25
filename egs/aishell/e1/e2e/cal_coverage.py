#!/usr/bin/env python
import re
import sys
import string

def read_kaldi_text(filename):
    zh_punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    
    l=[x.strip() for x in open(filename)]

    l2=[]

    for item in l:
        s=item.split(' ')[1] #remove id  
        s=re.sub(r'<.*>','',s)#remove tag        
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in s if ch not in exclude) #remove punct  
        exclude = set(zh_punc)
        s = ''.join(ch for ch in s if ch not in exclude) #remove zh punct  
        s = s.upper() #uppercase
        #BAN_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" 
        BAN_CHARS = "0123456789" 
        s = ''.join(c for c in s if c not in BAN_CHARS)   #remove alphanumeric (ie number)     
        s=s.replace(' ','') #remove space
        l2.append(s)
        
    return l2

def sort_dict_to_list_of_tuple(d):
    items=d.items() 
    backitems=[(item[1],item[0]) for item in items] 
    backitems.sort(reverse=True) 
    
    sorted_items=[(item[1],item[0]) for item in backitems] 
    
    return sorted_items

def spectrum_freq_sum(freq_spectrum):
    freq_values=[item[1] for item in freq_spectrum]
    return sum(freq_values)
    
    
def main(argv):        
    #argv
    if 'ipykernel_launcher.py' not in argv[0]:
        print('Run in console')
        input_filename=argv[1]
        k_top_freq=int(argv[2])
        try:
            output_filename=argv[3]
        except:
            output_filename='top_k_chars.txt'
    else:
        print('Run in Jupyter')
        input_filename='eu_cmhk_k_c1_2'        
        k_top_freq=2000
        output_filename='top_k_chars.txt'
        
    print(input_filename)
    print(k_top_freq)
    print(output_filename)
    
    print('\nProcessing, please wait ...\n')
        
        
    l_all=read_kaldi_text(input_filename)

    freq_spectrum={}

    for test_str in l_all:
        for c in test_str: 
            if c in freq_spectrum: 
                freq_spectrum[c] += 1
            else: 
                freq_spectrum[c] = 1

    freq_spectrum=sort_dict_to_list_of_tuple(freq_spectrum)
    #freq_spectrum=freq_spectrum[:-2] #remove invisible char (just hard code, so -2)

    top_freq_spectrum=freq_spectrum[:k_top_freq]
    
    #write top K char to file
    with open(output_filename,'w') as f_w:
        for item in top_freq_spectrum:
            f_w.write(item[0]+'\n')            

    #total freq
    total_freq=spectrum_freq_sum(freq_spectrum)
    print('total_freq')
    print("{:,}".format(total_freq))
    print('')

    #top k freq
    top_freq=spectrum_freq_sum(top_freq_spectrum)
    print('top_freq (%d)' % k_top_freq)
    print("{:,}".format(top_freq)) 
    print('')

    #coverage
    try:
        coverage=float(top_freq)/total_freq*100
        print('coverage')
        print('%.2f%%'%coverage)
    except:
        print('coverage cannot be calculated because total freq is 0')
    print('')

    print('num of unique chars')
    print("{:,}".format(len(freq_spectrum)))
    print('')

    try:
        print('--')
        print('')
        print('Top %d chars info:' % 10)
        print('')
        for i in range(10):
            print(freq_spectrum[i][0])
            print("{:,}".format(freq_spectrum[i][1]))
            print('\n')
    except:
        dummy=1   
        
    #find small_occ_items
    small_occ_items=[item for item in freq_spectrum if item[1]<=5]
    print('num of small occurance words (occurance <= 5)')
    print(len(small_occ_items))
    try:
        occ_ratio=round(len(small_occ_items)/len(freq_spectrum)*100)
        print('i.e.: '+"{:,}".format(occ_ratio)+'%')
    except:
        dummy=1
    print('')        
        
    
if __name__=="__main__":
    main(sys.argv)
    
    





