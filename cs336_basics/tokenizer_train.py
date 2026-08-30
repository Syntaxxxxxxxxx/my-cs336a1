import regex
import os
from . import pretokenization as pret

dataset="TinyStoriesV2-GPT4-valid.txt"
data_path="data"
dataset_path=os.path.join(data_path,dataset)
def train_bpe(path =dataset_path,
              vocab_size=500,
              special_tokens=None):


    counter={}
    with open(path, "rb") as f:
        num_processes = 4
        b_tokens=[]
        str=[]
        for token in special_tokens:
            b_tokens.append(token.encode("utf-8"))
            str.append(regex.escape(token))
        regex_str='|'.join(str)
        regex_str=regex.compile(regex_str)
        boundaries = pret.find_chunk_boundaries(f, num_processes, b_tokens)

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        PAT=r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            try:
                chunk = f.read(end - start).decode("utf-8")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                texts=regex.split(regex_str,chunk)
                for text in texts:
                    matchs=regex.finditer(PAT,text)
                    for mat in matchs:
                        tokens=mat.group()
                        bytes_pair=tokens.encode("utf-8")
                        char_pair=[bytes([b]) for b in bytes_pair]
                        pair_list=tuple(char_pair)
                        if pair_list not in counter:
                            counter[pair_list]=1
                        else:
                            counter[pair_list]+=1
            except:
                raise("counting error")
        f.close()

    vocab_list=[]
    for i in range(0,256):
        byte=bytes([i])
        vocab_list.append((byte,i))
    for token in special_tokens:
        vocab_list.append((token.encode("utf-8"),len(vocab_list)))
    curr_size=len(vocab_list)
    freq={}
    pair_to_seq={}
    merge_list=[]
    for item in counter.items():
        for pair in zip(item[0][:-1],item[0][1:]):
            if pair not in freq:
                freq[pair]=item[1]
            else:
                freq[pair]+=item[1]
            if pair not in pair_to_seq:
                pair_to_seq[pair]=set()
                pair_to_seq[pair].add(item[0])
            else:
                pair_to_seq[pair].add(item[0])

    while curr_size < vocab_size:
        if len(freq)==0:
            break
        merge_pair=max(freq,
                    key=lambda pair : (freq[pair],pair),
                    default=None)
        merge_list.append(merge_pair)
        new_vocab=merge_pair[0]+merge_pair[1]
        vocab_list.append((new_vocab,curr_size))
        curr_size+=1
        affected_words=pair_to_seq[merge_pair].copy()
        for word in affected_words:
            old_list=word
            new_list=()
            i=0
            while i<len(word)-1:
                pair=(word[i],word[i+1])
                if pair == merge_pair:
                    new_list+=(new_vocab,)
                    i+=2
                else:
                    new_list+=(word[i],)
                    i+=1
            if i==len(word)-1:
                new_list+=(word[i],)

            for pair in zip(old_list[:-1],old_list[1:]):
                freq[pair]-=counter[word]
                if freq[pair]==0:
                    del(freq[pair])
                pair_to_seq[pair].discard(old_list)
            for pair in zip(new_list[:-1],new_list[1:]):
                if pair not in freq:
                    freq[pair]=0
                freq[pair]+=counter[word]
                if pair not in pair_to_seq:
                    pair_to_seq[pair]=set()
                pair_to_seq[pair].add(new_list)
            counter[new_list]=counter[old_list]   
            counter[old_list]=0

    tokenizer_dict={}
    for item in vocab_list:
        tokenizer_dict[item[1]]=item[0]
    return tokenizer_dict,merge_list