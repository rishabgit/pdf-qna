# pdf-qna

for running:

```bash
echo "OPENAI_API_KEY=your_openai_api_key" > .env
pip install -r requirements.txt
python src/main.py
```

for dev:

```bash
pre-commit install
pre-commit run --all-files
pytest tests/
```
