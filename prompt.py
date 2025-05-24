import requests
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import fitz 
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def fetch_arxiv_pdf_text(arxiv_abs_url): 
    arxiv_id = arxiv_abs_url.split('/')[-1]
    if '/' in arxiv_abs_url.replace("https://arxiv.org/abs/", ""):
        arxiv_id = '/'.join(arxiv_abs_url.split('/')[-2:])
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(pdf_url)
    response.raise_for_status()
    pdf_path = f"{arxiv_id.replace('/', '_')}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(response.content)
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def build_vectorstore_from_pdf(arxiv_abs_url, embedding_model, google_api_key):
    text = fetch_arxiv_pdf_text(arxiv_abs_url)
    documents = [Document(page_content=text)]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = embedding_model(
        model="models/embedding-001",
    )
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def build_prompt(relevant_chunks, user_question):
    PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}"""
    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_chunks])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=user_question)
    return prompt

def answer_question_with_llm(vectorstore, user_question, google_api_key):
    retriever = vectorstore.as_retriever()
    relevant_chunks = retriever.invoke(user_question)
    prompt = build_prompt(relevant_chunks, user_question)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key
    )
    response = llm.invoke(prompt)
    return response

