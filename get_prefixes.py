import requests
from nltk.util import ngrams
from collections import Counter
import pickle

from tqdm import tqdm

if __name__ == "__main__":
    # get writing prompts prefixes
    prompts = []
    with open("story_data/writingPrompts/train.wp_source", "r") as f:
        for line in f:
            # append tokenized writing prompt without WP prefix
            prompt = line.strip().split()[3:]
            prompts.append((prompt, len(prompt)))
    with open("writing_prompts.pkl", "wb") as f:
        pickle.dump(prompts, f)
