# plagiarism_checker.py
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)

def clean_text(text):
    tokens = word_tokenize(text.lower())
    return " ".join([t for t in tokens if t.isalnum()])

def check_plagiarism(input_text, reference_texts, min_similarity=0.3):
    """
    Compare input text against a list of reference texts.
    Returns overall similarity percentage + matched sentences.
    """
    input_sentences = sent_tokenize(input_text)
    matches = []
    all_scores = []

    for ref in reference_texts:
        ref_sentences = sent_tokenize(ref)

        for s in input_sentences:
            cleaned_s = clean_text(s)
            if not cleaned_s.strip():
                continue

            # Compute similarity sentence by sentence
            vectorizer = TfidfVectorizer().fit([cleaned_s] + ref_sentences)
            vectors = vectorizer.transform([cleaned_s] + ref_sentences)
            sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

            best_score = max(sims) if len(sims) > 0 else 0
            all_scores.append(best_score)

            if best_score >= min_similarity:
                matches.append({
                    "sentence": s,
                    "similarity": round(best_score * 100, 2),
                    "reference": ref_sentences[sims.argmax()]
                })

    overall_similarity = round((sum(all_scores) / len(all_scores)) * 100, 2) if all_scores else 0

    return {
        "overall_similarity": overall_similarity,
        "matches": matches
    }

if __name__ == "__main__":
    refs = [
        "Artificial Intelligence is transforming industries across the globe. Machine learning is a subset of AI.",
        "Natural language processing helps computers understand human language."
    ]

    text = """Artificial Intelligence is transforming industries worldwide.
              Deep learning is an advanced approach to AI."""
    result = check_plagiarism(text, refs)

    print("Overall Similarity:", result["overall_similarity"], "%")
    for m in result["matches"]:
        print(f"🔗 {m['similarity']}% → {m['sentence']} (matched with: {m['reference']})")
