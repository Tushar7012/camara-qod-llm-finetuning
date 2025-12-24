"""
Direct Preference Optimization (DPO) Training Script for CAMARA QoD API

This script implements DPO to align the fine-tuned model away from hallucinations
and ensure strict adherence to CAMARA API specifications.

DPO optimizes the model to prefer "chosen" responses over "rejected" ones using
the Bradley-Terry preference model without explicit reward modeling.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

# =============================================================================
# Configuration
# =============================================================================

class DPOConfig:
    """Configuration for DPO training"""
    
    # Model paths
    SFT_MODEL_PATH = "./camara_qod_lora_model"  # Path to SFT checkpoint
    PREFERENCE_DATA_PATH = "preference_dataset.jsonl"
    OUTPUT_DIR = "./camara_qod_dpo_model"
    
    # DPO hyperparameters
    BETA = 0.1  # KL divergence penalty coefficient
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 1
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    MAX_LENGTH = 1024
    MAX_PROMPT_LENGTH = 512
    
    # LoRA configuration (same as SFT for consistency)
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05

# =============================================================================
# Data Processing
# =============================================================================

def format_dpo_prompt(example):
    """
    Format preference dataset for DPO training
    
    Args:
        example: Dict with 'prompt', 'chosen', 'rejected' keys
    
    Returns:
        Formatted dict for DPO trainer
    """
    instruction = "You are an expert assistant for the CAMARA Quality on Demand (QoD) API. Convert user requests into valid API calls."
    
    prompt_template = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{example['prompt']}

### Response:
"""
    
    return {
        "prompt": prompt_template,
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

def load_and_prepare_dataset(data_path):
    """Load and format preference dataset"""
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(format_dpo_prompt)
    return dataset

# =============================================================================
# Model Loading
# =============================================================================

def load_model_and_tokenizer(model_path, config):
    """
    Load SFT model as reference and create trainable copy for DPO
    
    Returns:
        model: Trainable model
        ref_model: Reference model (frozen)
        tokenizer: Tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load trainable model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    return model, ref_model, tokenizer

# =============================================================================
# DPO Training
# =============================================================================

def train_dpo(config=DPOConfig()):
    """
    Main DPO training loop
    
    The DPO objective maximizes:
        J(θ) = -E[(log σ(β * log(π_θ(y_w|x) / π_ref(y_w|x)) - 
                        β * log(π_θ(y_l|x) / π_ref(y_l|x))))]
    
    Where:
        π_θ = policy model (being trained)
        π_ref = reference model (frozen SFT checkpoint)
        y_w = chosen response
        y_l = rejected response
        β = KL penalty coefficient
        σ = sigmoid function
    """
    print("=== DPO Training for CAMARA QoD API ===\n")
    
    # Load dataset
    print("Loading preference dataset...")
    dataset = load_and_prepare_dataset(config.PREFERENCE_DATA_PATH)
    print(f"Loaded {len(dataset)} preference pairs\n")
    
    # Load models
    print("Loading models...")
    model, ref_model, tokenizer = load_model_and_tokenizer(
        config.SFT_MODEL_PATH, 
        config
    )
    print("Models loaded successfully\n")
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.NUM_EPOCHS,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        remove_unused_columns=False,
    )
    
    # Initialize DPO trainer
    print("Initializing DPO trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=config.BETA,  # KL penalty coefficient
        max_length=config.MAX_LENGTH,
        max_prompt_length=config.MAX_PROMPT_LENGTH,
    )
    
    print("\nDPO Trainer Configuration:")
    print(f"  Beta (KL penalty): {config.BETA}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}\n")
    
    # Train
    print("Starting DPO training...\n")
    dpo_trainer.train()
    
    # Save final model
    print("\nSaving DPO-aligned model...")
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    
    print(f"\n✅ DPO training complete!")
    print(f"Model saved to: {config.OUTPUT_DIR}")

# =============================================================================
# Inference & Evaluation
# =============================================================================

def evaluate_dpo_model(model_path, test_queries):
    """
    Evaluate DPO-aligned model on test queries
    
    Args:
        model_path: Path to DPO model
        test_queries: List of test queries
    """
    import json
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("\n=== DPO Model Evaluation ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        
        # Format prompt
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert assistant for the CAMARA Quality on Demand (QoD) API. Convert user requests into valid API calls.

### Input:
{query}

### Response:
"""
        
        # Generate
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.3)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = response.split("### Response:")[-1].strip()
        
        print(f"Response:\n{result}\n")
        
        # Validate JSON
        try:
            json_obj = json.loads(result)
            required_fields = ["device", "applicationServer", "qosProfile", "duration"]
            has_all_fields = all(f in json_obj for f in required_fields)
            
            if has_all_fields:
                print("✅ Valid CAMARA structure\n")
            else:
                print("⚠️ Missing required fields\n")
        except:
            print("❌ Invalid JSON\n")
        
        print("-" * 80 + "\n")

# =============================================================================
# Pseudocode for DPO Algorithm
# =============================================================================

DPO_PSEUDOCODE = """
# Direct Preference Optimization (DPO) Algorithm

## Key Concept
DPO directly optimizes the language model to prefer "chosen" responses over 
"rejected" responses without requiring a separate reward model.

## Algorithm Steps

1. INITIALIZE:
   - Load SFT checkpoint as reference model (π_ref) - FROZEN
   - Clone SFT checkpoint as policy model (π_θ) - TRAINABLE
   - Load preference dataset: {(x, y_chosen, y_rejected)}

2. FOR EACH TRAINING BATCH:
   
   a. Forward Pass:
      - Compute log probabilities for chosen response:
        log π_θ(y_chosen | x)
        log π_ref(y_chosen | x)
      
      - Compute log probabilities for rejected response:
        log π_θ(y_rejected | x)
        log π_ref(y_rejected | x)
   
   b. Compute Implicit Reward:
      - reward_chosen = β * [log π_θ(y_chosen|x) - log π_ref(y_chosen|x)]
      - reward_rejected = β * [log π_θ(y_rejected|x) - log π_ref(y_rejected|x)]
      
      Where β controls KL divergence from reference model
   
   c. Compute DPO Loss (Bradley-Terry Model):
      - loss = -log(σ(reward_chosen - reward_rejected))
      
      Where σ is the sigmoid function
      
      This encourages:
        π_θ(y_chosen | x) > π_θ(y_rejected | x)
      
      While penalizing deviation from π_ref via β coefficient
   
   d. Backward Pass:
      - Compute gradients: ∇_θ loss
      - Update only policy model parameters
      - Reference model remains frozen

3. REPEAT until convergence

## Why DPO Works

- Directly optimizes preference probability without reward model
- Implicitly learns reward via preference comparisons
- KL penalty (β) prevents model drift from SFT checkpoint
- More stable than RLHF (no unstable RL training)

## Expected Outcome

The model will:
✅ Generate responses matching "chosen" examples (CAMARA spec)
❌ Avoid generating responses like "rejected" examples (hallucinations)
🎯 Stay close to SFT checkpoint capabilities (via β penalty)
"""

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print(DPO_PSEUDOCODE)
    print("\n" + "="*80 + "\n")
    
    # Train DPO
    train_dpo()
    
    # Evaluate on test queries
    test_queries = [
        "I need better network for gaming. IP 192.168.1.50, server 203.0.113.100, 2 hours.",
        "4K streaming from phone +14155551234 to server 198.51.100.50 for 90 minutes.",
        "IoT sensor upload from phone +12025551111 to cloud 10.0.0.100, 15 minutes.",
    ]
    
    evaluate_dpo_model("./camara_qod_dpo_model", test_queries)
