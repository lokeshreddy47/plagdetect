# dataset_loader.py
import wikipedia
import json
import os

CACHE_FILE = "reference_texts.json"

# ✅ Correct topic list (with underscores for Wikipedia)
REFERENCE_TOPICS = [
    "Machine_learning",
    "Natural_language_processing",
    "Deep_learning",
    "Artificial_neural_network",
    "Technology",
    "Innovation",
    "Plagiarism",
    "Academic_integrity"
]

# ✅ Fallback summaries (used if Wikipedia fails)
FALLBACKS = {
    "Machine_learning": "Machine learning is a branch of artificial intelligence focused on building systems that learn from data to improve their performance without explicit programming.",
    "Natural_language_processing": "Natural language processing (NLP) is a field of AI that enables computers to understand, interpret, and generate human language.",
    "Deep_learning": "Deep learning is a subset of machine learning that uses multi-layered neural networks to model complex patterns in data.",
    "Innovation": "Innovation is the process of creating and applying new ideas, methods, or products that bring improvement or transformation.",
    "Artificial_neural_network": "An artificial neural network is a computational model inspired by the way biological neural networks process information.",
    "Technology": "Technology refers to the application of scientific knowledge for practical purposes, including tools, machines, and systems.",
    "Plagiarism": "Plagiarism is the act of presenting someone else's work, ideas, or words as your own without proper attribution.",
    "Academic_integrity": "Academic integrity is the commitment to honesty, trust, fairness, respect, and responsibility in scholarly work."
}


def load_reference_texts():
    reference_texts = []
    for topic in REFERENCE_TOPICS:
        try:
            print(f"🔎 Loading reference text for: {topic}")
            summary = wikipedia.summary(topic, sentences=5, auto_suggest=True)
            reference_texts.append({"topic": topic, "text": summary})
            print(f"✅ Loaded reference text for: {topic}")
        except Exception:
            # Use fallback if Wikipedia fails
            if topic in FALLBACKS:
                print(f"⚠ Could not load {topic} from Wikipedia. Using fallback.")
                reference_texts.append({"topic": topic, "text": FALLBACKS[topic]})
            else:
                print(f"⚠ No summary available for {topic}, skipping.")

    # Save to cache
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(reference_texts, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(reference_texts)} reference texts to cache.")

    # Print sample
    if reference_texts:
        print("\n=== SAMPLE REFERENCE TEXT ===")
        for ref in reference_texts[:2]:
            print(f"\n{ref['topic']}: {ref['text'][:300]}...")

    return reference_texts


if __name__ == "__main__":
    load_reference_texts()