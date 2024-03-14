# PDF Question-Answer Generator

Generates questions and answers from a given PDF file. Provides an API for uploading PDF files and extracting relevant questions and answers with cited sources. Includes LLM evaluation sources with reasoning for assessing the quality of the generated output.

## Features

- Extract text content from uploaded PDF files
- Generate questions and answers based on the PDF content
- Ensure answer factuality and provide source citations using Instructor library for pydantic data validation
- Evaluate LLM output using Uptrain, checking for:
  - Context Utilization: Completeness of the generated response given the provided context
  - Factual Accuracy: Correctness of the response based on the context
  - Language Features: Quality and effectiveness of language (clarity, coherence, conciseness)
  - Tonality: Matching the generated response to the required persona's tone

## Quick Setup with Docker

Create a `.env` file in the project root directory with your OpenAI API key:

```bash
echo "OPENAI_API_KEY=your_openai_api_key" > .env
```

To quickly set up the API using Docker, follow these steps:

```bash
docker build -t pdf-qna .
docker run -p 8000:8000 --env-file .env -v ./data:/app/data pdf-qna
   ```

The API will be accessible at `http://localhost:8000`, check out cURL example for uploading PDF in [API](#api) section below.

## Usage

If you're not using Docker, set up your OpenAI API key and install the required dependencies:

```bash
echo "OPENAI_API_KEY=your_openai_api_key" > .env
pip install -r requirements.txt
```

### API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Upload a PDF file using cURL (replace data/The-Emperors-New-Clothes.pdf with your PDF file path):

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

## TODOs

- Integrate support for images, charts, and illustrations in PDFs using GPT-4V. Refer to this [comparison](https://www.reddit.com/r/LocalLLaMA/comments/192eg2s/comparing_imagechartillustration_descriptions/) between GPT-4V, LLaVa, Qwen-VL, and CogVLM for more information.
  - Update the PDF reader to handle these additional modalities.
- Allows users to input questions and receive corresponding answers with supporting metadata.
- Perform an extensive evaluation of the LLM using OpenAI's [evals](https://github.com/openai/evals) framework across [diverse datasets](https://github.com/openai/evals/tree/main/evals/registry/data). This assessment will help identify niche areas where the LLM's output accuracy may be suboptimal, allowing us to determine beforehand if the model might not excel on a PDF belonging to that specific field.
- Find the final prompts sent by Uptrain and Instructor to OpenAI using [mitmproxy](https://mitmproxy.org/). Refer to this [blog post](https://hamel.dev/blog/posts/prompt/) by Hamel Husain for further context.
