import urllib.request
import re
from collections import Counter
import numpy as np

def fetch_text_from_url(url):
    """
    Downloads text from a Project Gutenberg URL.
    """
    print(f"Downloading content from: {url}")
    try:
        with urllib.request.urlopen(url) as f:
            return f.read().decode('utf-8').lower()
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def preprocess_text(text):
    clean_content = re.sub(r'[^a-z\s]', ' ', text)
    words = clean_content.split()
    return words

def build_vocab(words, min_count=5):
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    
    word2id = {word: i for i, word in enumerate(vocab)}
    id2word = {i: word for i, word in enumerate(vocab)}
    
    return vocab, word2id, id2word

def generate_skipgram_pairs(words, word2id, window_size=2):
    word_ids = [word2id[w] for w in words if w in word2id]
    pairs = []
    for i, target_id in enumerate(word_ids):
        start = max(0, i - window_size)
        end = min(len(word_ids), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:
                context_id = word_ids[j]
                pairs.append((target_id, context_id))
                
    return np.array(pairs)