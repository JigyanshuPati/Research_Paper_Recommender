# ArxivScout-Research_Paper_Recommender

ArxivScout is an interactive semantic search and question-answering dashboard for arXiv scientific papers. It leverages state-of-the-art language models and vector search to help you discover, preview, and ask questions about research papers.

---

## Features

- **Semantic Search:** Find relevant arXiv papers using natural language queries.
- **Paper Preview:** Instantly view the first page of the top-matching paper.
- **Similarity Visualization:** See a bar plot of the top-k papers most similar to your query.
- **Question Answering:** Ask questions about the top paper and get answers powered by Google Gemini (via LangChain).
- **arXiv Integration:** Direct links to arXiv abstracts for further reading.

---

## Setup

### 1. Clone the repository

```sh
git clone <your-repo-url>
cd semantic_research_paper
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

### 3. Set up your environment variables

Create a `.env` file in the project root with your Google API key:

```
GOOGLE_API_KEY=your-google-api-key-here
```

You can get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 4. Prepare your data

- Ensure `arXiv_scientific_dataset_cleaned.csv` is in the project directory.
- (Optional) Prepare your ChromaDB vector store using the provided scripts.

---

## Usage

Run the Gradio dashboard:

```sh
python gradio-dashboard.py
```

The app will launch in your browser.  
- Enter a search query to find papers.
- Preview the first page and see similarity scores.
- Ask a question about the top paper and get an AI-generated answer.

---

## Requirements

See `requirements.txt` for all dependencies.

---

## Notes

- This project uses [LangChain](https://github.com/langchain-ai/langchain), [ChromaDB](https://www.trychroma.com/), [Sentence Transformers](https://www.sbert.net/), and [Google Gemini](https://aistudio.google.com/app/apikey).
- Make sure your Python version is compatible with all dependencies (Python 3.8–3.11 recommended).

---


## Acknowledgements

- arXiv for open access to scientific papers.
- The open-source AI and NLP community.

---

