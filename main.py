from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import pypdf # used to read pdf documents
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import pickle

#optimize torch for Apple computers
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


#LLM Code
# Load the model & tokenizer
print("Starting Code")
MODEL_NAME = "google/gemma-2-2b-it"
print("Loading Tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Loading Model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True
    )
model.to(DEVICE)
print("Model Loaded")

#Loading CV data from PDF
print("Loading CV pdf to memory.")
def extract_text_from_pdf(pdf_path):
    #Open the PDF File
    pdf_reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small & efficient

# define how to split the PDF document
def split_text_into_chuncks(cv_text, chunk_size=1024, chunk_overlap=64):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(cv_text)
    return chunks


# Generate embeddings from the chunks of data
def generate_cv_embeddings(cv_path):
    cv_text = extract_text_from_pdf(cv_path)
    # Split text into smaller sections (e.g., paragraphs)
    sections = split_text_into_chuncks(cv_text)  # Splitting by double newlines

    # Convert each section into an embedding
    embeddings = embedding_model.encode(sections, convert_to_tensor=True)

    return sections, embeddings

cv_path = "/Users/dj/Documents/CLI_CV_Chatbot/CV_Del_Jackson.pdf"

cv_sections, cv_embeddings = generate_cv_embeddings(cv_path)
# convert embeddings to numpy array required by FAISS
cv_embeddings_np = np.array([emb.cpu().numpy() for emb in cv_embeddings])
# Save cv_sections
with open("cv_chunks.pkl", "wb") as f:
    pickle.dump(cv_sections, f)

#create FAISS index
embedding_dim = cv_embeddings_np.shape[1] # get size
index = faiss.IndexFlatL2(embedding_dim) # L2 distance fro similarity
index.add(cv_embeddings_np) # Add embeddings to FAISS

faiss.write_index(index, "cv_embeddings.index")
print(f"CV Loaded to Memory")

# Function to embed the users prompt
def get_query_embedding(query):
    query_embedding = embedding_model.encode([query]) # convert query to vector
    return np.array(query_embedding)

def retrieve_releveant_cv_sections(query, top_k=3):
        query_embedding = get_query_embedding(query)

        #Load Faiss Index
        index = faiss.read_index("cv_embeddings.index")

        #search for the top_k most similar CV chunks
        distances, indices = index.search(query_embedding, top_k)

        #load CV chunks from file
        with open("cv_chunks.pkl", "rb") as f:
            cv_chunks = pickle.load(f)

        # get the actual text of the chunks
        retrieved_chunks = [cv_chunks[i] for i in indices[0] if i <len(cv_chunks)]

        return "\n".join(retrieved_chunks)

# Adding chat history to the bot
chat_history = []


# Function to generate text
def chat_with_gemma(prompt):
    print("Getting a response from the LLM......")

    # Retreive relevant CV information
    relevant_cv_info = retrieve_releveant_cv_sections(prompt)

    # included prompt engineering to get better responses
    system_prompt = """You are an AI assistant helping users understand a persons CV. Use the 
    provided CV information to answer questions accurately. If the CV does not contain the
    answer, say 'I do not know."
    """

    # keep last 3 exchanges in the prompt to aviod making the prompt too long
    history_text = "\n".join(chat_history[-6:]) # only keeps last 3 exchanges (6 messages user + AI)
    full_prompt = f"{system_prompt}\n\nConversation History:\n{history_text}\n\nRelevant CV Information:\n{relevant_cv_info}\n\nUser: {prompt}\nAI:"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.3,  # controls randomness
            top_p=0.8,  # improves coherence
            repetition_penalty=1.2  # Penalizes word repetition
            )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # extract only the AI's response by splitting on "AI:"
    if "AI:" in response:
        response = response.split("AI:")[-1].strip()
    chat_history.append(f"User: {prompt}")
    chat_history.append(f"AI: {response}")
    print("\nGemma's Response:\n", response)


print("Starting up chat")
print("Type the word quit to end the program")
# Loop to enable the user to continually chat
runLLM = True
while runLLM:
    try:
        query = input("Enter your next message: ").strip()
        if query == "":
            continue
        if query == "quit":
            runLLM = False
        else:
            chat_with_gemma(query)
    except KeyboardInterrupt:
        print("Chatbot Terminated")
        sys.exit(0)
