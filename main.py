import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
import pandas as pd
import tqdm

from huggingface_hub import login
from google.colab import userdata



login(token=userdata.get('HF_TOKEN'))

# Load the model
model_name = "microsoft/phi-2"  # this is just an example, make your own choice on the model.
# Note that different models may have different ways to initialize, adapt by yourself.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="cuda:0"  # specify the model hosting on GPU
)


model


# The following codes illustrate how to use the model to generate content based on given inputs.
# NOTE that different model has different way of generating responses, adapt by yourself
prompt = "Hello, how are you?"
formatted_prompt = f"Instruct: {prompt}\nOutput:"
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
output_ids = model.generate(
    **inputs,
    max_length=200,
    do_sample=True,
    temperature=0.7,
    # add/reduce parameters upon your desire
)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)


# Extract only the answer after "Output:"
output_text.split("Output:", 1)[-1].strip()


# Load the Toxicity subset of the DecodingTrust dataset
dataset = load_dataset("AI-Secure/DecodingTrust", "toxicity")  # choose your own subsets (at least 3)
dataset  # check the structure and info of this dataset