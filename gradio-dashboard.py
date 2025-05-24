import os
import tempfile
import requests
import pandas as pd
import fitz 

 # PyMuPDF
import gradio as gr
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import os

from prompt import (
    GoogleGenerativeAIEmbeddings,
    answer_question_with_llm,
    build_vectorstore_from_pdf
)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

CSV_PATH = "/Users/jigyanshupati/semantic_research_paper/arXiv_scientific_dataset_cleaned.csv"
COLLECTION_NAME = "tagged_summary_collection"
PERSIST_PATH = "/Users/jigyanshupati/semantic_research_paper/chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"


model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=PERSIST_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def render_first_page(arxiv_id: str) -> str:
    try:
        prefix, num_version = arxiv_id.split("-")
        number = num_version.split("v")[0]
        url_id = f"{prefix}/{number}"
    except Exception as e:
        raise ValueError("Invalid arXiv ID format.") from e

    pdf_url = f"https://arxiv.org/pdf/{url_id}.pdf"

    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, f"{arxiv_id}.pdf")
    img_path = os.path.join(temp_dir, f"{arxiv_id}_page1.png")

    if not os.path.exists(pdf_path):
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(response.content)

    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200)
    pix.save(img_path)
    return img_path

def get_arxiv_abs_url(arxiv_id):
    if '-' in arxiv_id:
        category, rest = arxiv_id.split('-', 1)
        number = rest.split('v')[0]
        url_id = f"{category}/{number}"
    else:
        url_id = arxiv_id.split('v')[0]
    return f"https://arxiv.org/abs/{url_id}"

def get_recommendations(query: str, k: int = 5):
    if not query.strip():
        return None, "Please enter a search query.", None, None

    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k
    )

    doc_ids = [doc.split()[0].strip() for doc in results['documents'][0]]

    df = pd.read_csv(CSV_PATH)
    matched_df = df[df["id"].isin(doc_ids)]

    if matched_df.empty:
        return None, "No matching papers found.", None, None

    matched_df.loc[:, "id"] = pd.Categorical(
        matched_df["id"],
        categories=list(dict.fromkeys(doc_ids)),
        ordered=True
    )
    matched_df = matched_df.sort_values("id")

    arxiv_id = matched_df.iloc[0]['id']
    img_path = render_first_page(arxiv_id)
    arxiv_url = get_arxiv_abs_url(arxiv_id)

    paper_titles = matched_df["title"].tolist()
    distances = results["distances"][0]
    similarity_scores = [1 - d for d in distances]

    min_len = min(len(paper_titles), len(similarity_scores))
    paper_titles = paper_titles[:min_len]
    similarity_scores = similarity_scores[:min_len]

    plot_df = pd.DataFrame({
        "Paper Title": paper_titles,
        "Similarity Score": similarity_scores
    })

    return img_path, f"[View paper on arXiv]({arxiv_url})", plot_df, arxiv_id

def answer_question(prompt, arxiv_id):
    if not arxiv_id:
        return "Please search and select a paper first."
    arxiv_abs_url = get_arxiv_abs_url(arxiv_id)
    try:
        vectorstore = build_vectorstore_from_pdf(arxiv_abs_url, GoogleGenerativeAIEmbeddings, GOOGLE_API_KEY)
        answer = answer_question_with_llm(vectorstore, prompt, GOOGLE_API_KEY)
        return answer.content
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown(
        "# ArxivScout-Research_Paper_Recommender",
        elem_id="centered-heading"
    )
    gr.Markdown("Enter a search query to get top-k similar arXiv papers.")

    with gr.Row():
        with gr.Column(scale=1):
            query = gr.Textbox(label="Search Query")
            prompt_box = gr.Textbox(label="Ask a question about the top paper")
            answer_box = gr.Textbox(label="Response from AI", interactive=False)
        with gr.Column(scale=3):
            img = gr.Image(label="First Page of PDF")
            link = gr.Markdown(label="arXiv Paper Link")
            barplot = gr.BarPlot(
                value=None,
                x="Paper Title",
                y="Similarity Score",
                title="Top-k Paper Similarities to Query"
            )

    state_arxiv_id = gr.State()

    def update(query):
        img_path, arxiv_link, plot_df, arxiv_id = get_recommendations(query)
        return img_path, arxiv_link, plot_df, arxiv_id

    query.submit(
        update,
        inputs=query,
        outputs=[img, link, barplot, state_arxiv_id]
    )
    prompt_box.submit(
        answer_question,
        inputs=[prompt_box, state_arxiv_id],
        outputs=answer_box
    )

demo.launch()