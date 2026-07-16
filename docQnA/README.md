# Document QnA Chatbot

A simple Streamlit-based document question-answering app that loads a PDF, splits text into chunks, generates embeddings with Google Gemini, and allows conversational querying over the uploaded document.

## Project overview

This project is a lightweight document Q&A chatbot built with Streamlit and LangChain. Users upload a PDF file, the app processes the text into vector embeddings, and then answers questions using the context retrieved from the uploaded document.

## Setup instructions

1. Clone or copy the project to your local machine.
2. Create a Python virtual environment if you do not already have one:

```bash
python -m venv env
```

3. Activate the virtual environment:

- Windows PowerShell:
  ```powershell
  .\env\Scripts\Activate.ps1
  ```
- Windows Command Prompt:
  ```cmd
  .\env\Scripts\activate.bat
  ```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Required environment variables

This app imports `load_dotenv()` but does not currently reference any environment variables directly in `app.py`.

If you plan to use Google Generative AI credentials or other secrets, add them to a `.env` file in the project root. Example variables may include:

```env
GOOGLE_API_KEY=your_api_key_here
```

> Note: Configure any provider-specific credentials required by `langchain_google_genai` according to your chosen deployment.

## How to run the project locally

From the project root, start the Streamlit app:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).


## Additional notes

- The app uploads the selected PDF to `uploaded_document.pdf` in the project root.
- If you close the browser and reopen the app, you may need to re-upload the PDF to reinitialize the session.
- The app uses an in-memory vector store, so all document data is lost when the app stops.

## Dependencies

The main dependencies are listed in `requirements.txt` and include:

- `streamlit`
- `langchain`
- `langchain-google-genai`
- `langchain-community`
- `langchain-text-splitters`
- `python-dotenv`
- `pypdf`

If you add more packages later, update `requirements.txt` accordingly.
