# Local AI Restaurant Review Assistant

## Project Overview

This project is a simple local AI assistant that uses LangChain and Ollama embeddings to answer questions about restaurant reviews. It loads a CSV dataset of realistic restaurant reviews, indexes the review text using a local Chroma vector store, and lets you query the indexed reviews interactively from the command line.

## Setup Instructions

1. Clone or download the project into a local folder.
2. Create and activate a Python virtual environment.
   - Windows PowerShell:
     ```powershell
     python -m venv env
     .\env\Scripts\Activate.ps1
     ```
3. Install the required Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Ensure the dataset file `realistic_restaurant_reviews.csv` is present in the project root.

## Required Environment Variables

This project does not depend on any custom environment variables out of the box.

However, the Ollama model integration may require that an Ollama instance is installed and running locally if you are using the Ollama client. If your Ollama setup requires environment variables, configure them accordingly.

## How to Run the Project Locally

From the project root, with the virtual environment activated:

```powershell
python main.py
```

The script starts an interactive prompt and asks for questions. Type your question and press Enter. Enter `q` to quit.


## Additional Notes and Dependencies

- Python dependencies are listed in `requirements.txt`:
  - `langchain`
  - `langchain-ollama`
  - `langchain-chroma`
  - `pandas`
- The `vector.py` module builds or loads a Chroma vector store in `./chroma_langchain_db`.
- If the database folder does not exist, `vector.py` creates the vector store and persists document embeddings.
- The project assumes local access to an Ollama embedding/model service via `langchain_ollama`.
- If you want to customize the model or dataset, update `main.py` and `vector.py` as needed.
