# ai_detector.py
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Download NLTK resources quietly
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Load sentence-transformer model (small & fast)
EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load Hugging Face zero-shot classifier for AI detection
AI_CLASSIFIER = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def clean_text(text: str) -> str:
    """
    Minimal text cleaning for embeddings:
    lowercase → tokenize → remove stopwords → keep alphanumeric tokens.
    """
    if not isinstance(text, str):
        text = str(text)

    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w.isalnum() and w not in stopwords.words("english")]
    return " ".join(filtered)


def check_plagiarism(input_text: str, reference_texts: list, threshold: float = 0.3):
    """
    Semantic plagiarism check using sentence-transformer embeddings.

    Args:
        input_text (str): Text to check.
        reference_texts (list): List of reference texts.
        threshold (float): Minimum cosine similarity (0–1) to consider as a match.

    Returns:
        dict: {
            "plagiarized": bool,
            "matches": [
                {"reference": "...", "similarity": float_percent}
            ]
        }
    """
    if not input_text or not reference_texts:
        return {"plagiarized": False, "matches": []}

    # Generate embeddings
    input_emb = EMBED_MODEL.encode(input_text, convert_to_tensor=True)
    ref_embs = EMBED_MODEL.encode(reference_texts, convert_to_tensor=True)

    # Compute cosine similarities
    cos_scores = util.cos_sim(input_emb, ref_embs)[0]
    matches = []
    for idx, score in enumerate(cos_scores):
        similarity = float(score.item())
        if similarity >= threshold:
            matches.append({
                "reference": reference_texts[idx],
                "similarity": round(similarity * 100, 2)  # as percentage
            })

    return {
        "plagiarized": len(matches) > 0,
        "matches": matches
    }


def check_ai_generated(text: str) -> str:
    """
    Uses zero-shot classification to label text as AI-generated or Human-written.

    Returns:
        str: A user-friendly result string.
    """
    if not text or not text.strip():
        return "No text provided for AI analysis."

    candidate_labels = ["AI-generated", "Human-written"]
    try:
        result = AI_CLASSIFIER(text, candidate_labels=candidate_labels)
        label = result["labels"][0]
        confidence = result["scores"][0]
        return f"{label} (confidence: {confidence:.2f})"
    except Exception as e:
        return f"Error running AI detection: {str(e)}"