# pdf-qna

for running:

```bash
echo "OPENAI_API_KEY=your_openai_api_key" > .env
pip install -r requirements.txt
uvicorn app:app --reload
curl -X POST -F "pdf_file=@data/The-Emperors-New-Clothes.pdf" http://127.0.0.1:8000/extract_text
```

```python
from src.main import run
run("data/The-Emperors-New-Clothes.pdf")
```

for dev:

```bash
pre-commit install
pre-commit run --all-files
pytest tests/
```
