# PDF Question-Answer Generator

This project generates questions and answers from a given PDF file. It provides a FastAPI-based API for uploading PDF files and extracting relevant questions and answers.

## Features

- Extract text content from uploaded PDF files
- Generate questions and answers based on the PDF content
- Ensure answer factuality and provide source citations using [Instructor](https://jxnl.github.io/instructor/why/) library for pydantic validation
- Evaluate LLM output using [Uptrain](https://docs.uptrain.ai/getting-started/why-we-are-building-uptrain), checking for:
  - Context Utilization: Completeness of the generated response given the provided context
  - Factual Accuracy: Correctness of the response based on the context
  - Language Features: Quality and effectiveness of language (clarity, coherence, conciseness)
  - Tonality: Matching the generated response to the required persona's tone

## Installation

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/rishabgit/pdf-qna.git
   cd pdf-qna
   ```

2. Set up your OpenAI API key in a `.env` file:

   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key" > .env
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Upload a PDF file using cURL:

```bash
curl -X POST -F "pdf_file=@data/The-Emperors-New-Clothes.pdf" http://127.0.0.1:8000/extract_text
```

### Script

Run the script directly:

```python
from src.main import run
run("data/The-Emperors-New-Clothes.pdf")
```

## Development

1. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

2. Run pre-commit checks:

   ```bash
   pre-commit run --all-files
   ```

3. Run tests:

   ```bash
   pytest tests/
   ```
