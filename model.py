import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
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
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )
    print("Model Loaded")
    # Wrap it in LangChain's LLM class
    return HuggingFacePipeline(pipeline=llm_pipeline)
