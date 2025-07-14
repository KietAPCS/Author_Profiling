import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

def init_nlp_tools():
    nltk.download('punkt')
    nltk.download('wordnet')
    
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    return stemmer, lemmatizer

def stem_lema(text):
    text = text.lower()
    tokens = word_tokenize(text)
    processed_tokens = []
    stemmer, lemmatizer = init_nlp_tools()

    for token in tokens:
        if token in string.punctuation:
            continue
        
        lemma = lemmatizer.lemmatize(token)
        stem = stemmer.stem(lemma)
        processed_tokens.append(stem)

    return processed_tokens