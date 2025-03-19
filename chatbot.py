from model import load_llm
from prompt import prompt_template
from embeddings import retrieve_releveant_cv_sections, load_cv
from langchain.schema.output_parser import StrOutputParser

print("Starting up CV Chatbot")
llm = load_llm()
print("Loading CV")
load_cv()
llm_chain = prompt_template | llm | StrOutputParser()
chat_history = []


def chat_with_gemma(prompt):
    print("Getting a response from the LLM......")

    # Retreive relevant CV information
    relevant_cv_info = retrieve_releveant_cv_sections(prompt)

    # Keep last 3 exchanges in the prompt to avoid making the prompt too long
    # only keeps last 3 exchanges (6 messages user + AI)
    history_text = "\n".join(chat_history[-6:])

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
