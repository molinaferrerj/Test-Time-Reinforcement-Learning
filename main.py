import sys
from util import fetch_text_from_url, preprocess_text, build_vocab, generate_skipgram_pairs
from model import Word2Vec

def run_experiment():
    config = {
        "url": "https://www.gutenberg.org/files/1661/1661-0.txt",
        "dim": 50,        
        "window": 3,         
        "lr": 0.02,
        "epochs": 10
    }

    raw = fetch_text_from_url(config["url"])
    if not raw:
        sys.exit("Failed to load data. Check your connection.")

    tokens = preprocess_text(raw)
    vocab, w2id, id2w = build_vocab(tokens, min_count=5)
    
    print(f">> Loaded {len(tokens)} tokens. Vocabulary: {len(vocab)} unique words.")

    pairs = generate_skipgram_pairs(tokens, w2id, window_size=config["window"])

    engine = Word2Vec(
        vocab_size=len(vocab), 
        embedding_dim=config["dim"], 
        learning_rate=config["lr"]
    )

    print(f">> Training on {len(pairs)} pairs for {config['epochs']} epochs...")
    try:
        engine.train(pairs, epochs=config['epochs'])
    except KeyboardInterrupt:
        print("\n>> Training interrupted by user. Saving current state...")

    test_words = ['sherlock', 'watson', 'man', 'woman']
    print("\n--- Quick Vector Snapshot ---")
    for w in test_words:
        if w in w2id:
            vec_sum = engine.get_embedding(w2id[w])[:3] 
            print(f"Word: {w:10} | Vector Fragment: {vec_sum}")

if __name__ == "__main__":
    run_experiment()