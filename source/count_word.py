from nltk.util import ngrams
from collections import Counter
from preprocess import stem_lema

def count_word(text_tokens, n=2):
    n_gram_list = ngrams(text_tokens, n)
    profile = Counter(n_gram_list)
    return profile