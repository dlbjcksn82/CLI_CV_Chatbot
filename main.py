from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

# Load the model & tokenizer
print("Starting Code")
MODEL_NAME = "google/gemma-3-1b-pt"
print("Loading Tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Loading Model")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
print("Model Loaded")
# Function to generate text


def chat_with_gemma(prompt):
    print("Getting a response from the LLM......")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=150,
        do_sample=True,
        temperature=0.5,  # controls randomness
        top_p=0.9,  # nucleus sampling to balance diversity
        repetition_penalty=1.2  # Penalizes word repetition
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGemma's Response:\n", response)

print("Starting up chat")
print("Type the word quit to end the program")
# Example
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
