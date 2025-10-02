import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login
import os
import dotenv



# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
SAMPLE_SIZE = 50
RANDOM_SEED = 42

# ============================================================================
# Model Setup
# ============================================================================

def setup_model(model_name):
    """Load tokenizer and model."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="cpu"
    )
    return tokenizer, model

# ============================================================================
# Dataset Loading
# ============================================================================

def load_decodingtrust_datasets():
    """Load DecodingTrust benchmark datasets."""
    print("Loading DecodingTrust datasets...")
    toxicityDataset = load_dataset("AI-Secure/DecodingTrust", "toxicity")  # choose your own subsets (at least 3)
    stereotypeDataset = load_dataset("AI-Secure/DecodingTrust", "stereotype")  # choose your own subsets (at least 3
    privacyDataset = load_dataset("AI-Secure/DecodingTrust", "privacy")  # choose your own subsets (at least 3


    
    
    return privacyDataset, toxicityDataset , stereotypeDataset

def sample_datasets(toxicityDataset, stereotypeDataset, privacyDataset, 
                    sample_size=50, seed=42):
    """Sample random subsets from datasets."""
    # Extract specific splits
    stereotypeSplit = stereotypeDataset["stereotype"]
    toxicSplit = toxicityDataset["toxic.gpt4"]
    privacySplit = privacyDataset["enron.five_shot"]

    
    # Random sampling
    toxicGpt4SplitSubRandom = toxicSplit.shuffle(seed=42).select(range(50))
    stereotypeSplitSubRandom = stereotypeSplit.shuffle(seed=42).select(range(50))
    privacySplitSubRandom = privacySplit.shuffle(seed=42).select(range(50))

    # sampled text
    toxicityPromptsRandom = [item['text'] for item in toxicGpt4SplitSubRandom['prompt']]
    print("Randomly sampled prompts:")
    print(toxicityPromptsRandom)

    stereotypeRandomPrompts = [item["text"] for item in stereotypeSplitSubRandom["prompt"]]
    print("Randomly sampled stereotype prompts:")
    print(stereotypeRandomPrompts)

    privacyRandomPrompts = [item for item in privacySplitSubRandom["prompt"]]
    print("Randomly sampled privacy prompts:")
    print(privacyRandomPrompts)
    
    return toxicityPromptsRandom, stereotypeRandomPrompts, privacyRandomPrompts
def create_splits(toxicityDataset, stereotypeDataset, privacyDataset, 
                    sample_size=50, seed=42):
    stereotypeSplit = stereotypeDataset["stereotype"]
    toxicSplit = toxicityDataset["toxic.gpt4"]
    privacySplit = privacyDataset["enron.five_shot"]
    return toxicSplit, stereotypeSplit, privacySplit
def balanced_sampling(toxicSplit, stereotypeSplit, privacySplit):
    toxicGpt4DF = toxicSplit.to_pandas()
    stereotypeDF = stereotypeSplit.to_pandas()
    privacyDF = privacySplit.to_pandas()

    toxicGpt4DF['toxicity_score'] = toxicGpt4DF['prompt'].apply(lambda x: x['toxicity'])
    stereotypeDF["sys_prompt_type_tag"] = stereotypeDF["prompt"].apply(lambda x: x["sys_prompt_type_tag"])

    return toxicGpt4DF, stereotypeDF, privacyDF

def main():
    dotenv.load_dotenv()
    login(token=os.getenv("HF_TOKEN"))
    modelDetails = setup_model(MODEL_NAME)
    print("model setup works")
    datasets = load_decodingtrust_datasets()
    print("datasets loaded")
    splits = create_splits(datasets[1], datasets[2], datasets[0])
    print(balanced_sampling(*splits))

if __name__ == "__main__":
    main()
