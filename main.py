from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, sys, pypdf, pickle, faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
# Langchain implementation
from langchain_huggingface import HuggingFaceEmbeddings
# Functionality to call LLM using Langchain model integrations
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser



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

# Model updates with Langchain
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
    max_new_tokens=300,
    do_sample=True,
    temperature = 0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id
    )

# Wrap it in LangChain's LLM class
llm = HuggingFacePipeline(pipeline=llm_pipeline)

#Create LangChain PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["chat_history", "cv_info", "user_input"],
    template="""You are an AI assistant helping users understand a person's CV. 
    Use the provided CV information to answer questions accurately. 
    If the CV does not contain the answer, say 'I do not know.' and do not make up an answer.
     Do not repeat the question.
    Provide detailed, structured responses, explaining key points with useful context.
    Elaborate when possible by adding background information to make the answer more useful
    to a potential employer. Avoid one-word answers unless absolutely necessary, but try to keep answers 
    concise and to the point.

    Here is the conversation history so far:
    {chat_history}

    Additional relevant CV Information:
    {cv_info}

    Now, answer the following question in a professional and structured manner:
    User's Question: {user_input}
    AI's Response:
    """
)



# Define the LLM Chain
llm_chain = prompt_template | llm | StrOutputParser()


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
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
    sections = split_text_into_chuncks(cv_text)  # Splitting chunks of text

    # Convert each section into an embedding
    embeddings = embedding_model.embed_documents(sections)

    return sections, embeddings

cv_path = "/Users/dj/Documents/CLI_CV_Chatbot/CV_Del_Jackson.pdf"

cv_sections, cv_embeddings = generate_cv_embeddings(cv_path)
# convert embeddings to numpy array required by FAISS
cv_embeddings_np = np.array(cv_embeddings, dtype=np.float32)
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
    query_embedding = embedding_model.embed_query(query) # convert query to vector
    return np.array([query_embedding], dtype=np.float32)

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

    # keep last 3 exchanges in the prompt to aviod making the prompt too long
    history_text = "\n".join(chat_history[-6:]) # only keeps last 3 exchanges (6 messages user + AI)

    response = llm_chain.invoke({
        "chat_history": history_text,
        "cv_info": relevant_cv_info,
        "user_input": prompt
    })
    if "AI's Response:" in response:
        response = response.split("AI's Response:")[-1].strip()

    # extract only the AI's response by splitting on "AI:"]
    clean_response = response.strip()
    chat_history.append(f"User: {prompt}")
    chat_history.append(f"AI: {clean_response}")
    print("\nDel's CV Chatbot repsonse:\n", response)


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
