import re

def clean_text(text: str) -> str:
    """
    Cleans and normalizes input text.
    - Converts to lowercase
    - Removes special characters (except ., and ,)
    - Normalizes whitespace
    """

    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove unwanted characters (keep letters, numbers, space, dot, comma)
    text = re.sub(r"[^a-z0-9\s.,]", "", text)

    # Normalize multiple spaces → single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing spaces
    return text.strip()


# ✅ Quick test (optional)
if __name__ == "__main__":
    sample = "Hello!!!   This   is a    SAMPLE --- Text, for Cleaning..."
    print("Before:", sample)
    print("After :", clean_text(sample))
