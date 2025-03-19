import numpy as np
import pickle
import faiss
import pypdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def extract_text_from_pdf(pdf_path):
    # Open the PDF File
    pdf_reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


def split_text_into_chuncks(cv_text, chunk_size=1024, chunk_overlap=64):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(cv_text)


def generate_cv_embeddings(cv_path):
    cv_text = extract_text_from_pdf(cv_path)
    # Split text into smaller sections (e.g., paragraphs)
    sections = split_text_into_chuncks(cv_text)  # Splitting chunks of text

    # Convert each section into an embedding
    embeddings = embedding_model.embed_documents(sections)

    return sections, embeddings


def save_faiss_index(cv_sections, cv_embeddings,
                     index_path="data/cv_embeddings.index"):
    cv_embeddings_np = np.array(cv_embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(cv_embeddings_np.shape[1])
    index.add(cv_embeddings_np)
    faiss.write_index(index, index_path)
    with open("data/cv_chunks.pkl", "wb") as f:
        pickle.dump(cv_sections, f)


def retrieve_releveant_cv_sections(query, top_k=3):
    query_embedding = np.array(
        [embedding_model.embed_query(query)],
        dtype=np.float32
    )

    # Load Faiss Index
    index = faiss.read_index("cv_embeddings.index")

    # search for the top_k most similar CV chunks
    distances, indices = index.search(query_embedding, top_k)

    # load CV chunks from file
    with open("cv_chunks.pkl", "rb") as f:
        cv_chunks = pickle.load(f)

    # get the actual text of the chunks
    retrieved_chunks = [cv_chunks[i] for i in indices[0] if i < len(cv_chunks)]

    return "\n".join(retrieved_chunks)


def load_cv():
    cv_path = "/Users/dj/Documents/CLI_CV_Chatbot/CV_Del_Jackson.pdf"

    cv_sections, cv_embeddings = generate_cv_embeddings(cv_path)
    # convert embeddings to numpy array required by FAISS
    cv_embeddings_np = np.array(cv_embeddings, dtype=np.float32)
    # Save cv_sections
    with open("cv_chunks.pkl", "wb") as f:
        pickle.dump(cv_sections, f)

    # create FAISS index
    embedding_dim = cv_embeddings_np.shape[1]  # get size
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance fro similarity
    index.add(cv_embeddings_np)  # Add embeddings to FAISS
    faiss.write_index(index, "cv_embeddings.index")
    print("CV Loaded to Memory")
