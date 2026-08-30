from .tokenizer_train import train_bpe
import argparse
import pickle

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--vocab-size",type=int,default=500)
    parser.add_argument("--special-tokens",default="<|endoftext|>",nargs="+")
    parser.add_argument("output")
    args=parser.parse_args()
    vocab,merges=train_bpe(args.path,
                          args.vocab_size,
                          args.special_tokens)
    with open(args.output,"wb") as f:
        pickle.dump(
            {
                "vocab":vocab,
                "merges":merges,
                "special_tokens":args.special_tokens
            },
            f,
        )

if __name__ == "__main__":
    main()
