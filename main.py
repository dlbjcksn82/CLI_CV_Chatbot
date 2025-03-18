from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

#optimize torch for Apple computers
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

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


# Function to generate text
def chat_with_gemma(prompt):
    print("Getting a response from the LLM......")
    # included prompt engineering to get better responses
    system_prompt = """You are an AI assistant trained to provide detailed and accurate responses. 
    Always give structured, helpful, and relevant answers to user queries.
    If you don't know something, say you don't know instead of making up information.
    """
    full_prompt = f"{system_prompt}\nUser: {prompt}\nAI:"

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
