import requests
from nltk.util import ngrams
from collections import Counter
import pickle

from tqdm import tqdm

if __name__ == "__main__":
    ptb_url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt"
    raw_ptb = requests.get(ptb_url).text
    ptb_lines = [line.strip().split() for line in raw_ptb.split("\n")]

    N = 5
    freqs = Counter()
    for line in tqdm(ptb_lines):
        for n in range(N):
            freqs += Counter(ngrams(line, n))
    with open("ngram_counter.pkl", "wb") as f:
        pickle.dump(freqs, f)
    with open("ngram_sorted.pkl", "wb") as f:
        pickle.dump(sorted(freqs.items(), key=lambda x: x[1]), f)

