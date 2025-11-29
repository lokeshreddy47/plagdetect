# stylometry.py
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download("punkt", quiet=True)

def extract_stylometric_features(text: str):
    if not isinstance(text, str):
        text = str(text)
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    num_sentences = len(sentences)
    num_words = len(words)
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    avg_word_length = sum(len(w) for w in words) / num_words if num_words > 0 else 0

    return {
        "sentence_count": num_sentences,
        "word_count": num_words,
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length
    }

def stylometry_verdict(text: str):
    features = extract_stylometric_features(text)
    if features["avg_sentence_length"] > 25:
        return "⚠️ Stylometry: text looks AI-like (very long sentences)."
    if features["avg_word_length"] < 4:
        return "ℹ️ Stylometry: simple vocabulary (likely human, short words)."
    return "✅ Stylometry: writing style looks human-like."
