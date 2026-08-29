import regex
import os
import pretokenization as pret

dataset="TinyStoriesV2-GPT4-valid.txt"
data_path="data"
dataset_path=os.path.join(data_path,dataset)

counter={}
with open(dataset_path, "rb") as f:
    num_processes = 4
    boundaries = pret.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    PAT="'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        try:
            chunk = f.read(end - start).decode("utf-8")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            texts=chunk.split('<|endoftext|>')
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

vocab_size=500
curr_size=257
freq={}
pair_to_seq={}
for item in counter.items():
    for pair in zip(item[0][:-1],item[0][1:]):
        if pair not in freq:
            freq[pair]=item[1]
        else:
            freq[pair]+=item[1]
        if pair not in pair_to_seq:
            pair_to_seq[pair]=[item]
        else:
            pair_to_seq[pair].append(item)

while curr_size < vocab_size:
    merge_pair=max(freq,
                   key=lambda pair : (freq[pair],pair),
                   default=None)
    for item in pair_to_seq[merge_pair]:
        freq[merge_pair]-=item[1]