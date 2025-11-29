FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project
COPY . .

# Install NLTK data
RUN python3 - <<EOF
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
EOF

EXPOSE 8000

# Run Flask directly
CMD ["python", "app.py"]
