import os

from fastapi import FastAPI, File, UploadFile

from src.main import run

app = FastAPI()

os.makedirs("data", exist_ok=True)


@app.post("/extract_text")
async def extract_text(pdf_file: UploadFile = File(...)):
    file_path = os.path.join("data", pdf_file.filename)
    with open(file_path, "wb") as file:
        file.write(await pdf_file.read())
    documents = run(file_path)
    return documents
