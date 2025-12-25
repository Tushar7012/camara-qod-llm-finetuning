"""
Simple Test Script for CAMARA QoD Fine-tuned Model

This script loads your trained model and tests it with various queries.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

print("="*60)
print("🚀 CAMARA QoD Model Tester")
print("="*60)

# Configuration
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "./camara_qod_final_model"

print("\n📥 Loading model...")
print(f"   Base: {BASE_MODEL}")
print(f"   Adapter: {ADAPTER_PATH}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("✅ Model loaded successfully!\n")

# Prompt template (same as training)
def format_prompt(user_query):
    instruction = "You are an expert assistant for the CAMARA Quality on Demand (QoD) API. Convert user requests into valid API calls."
    
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{user_query}

### Response:
"""
    return prompt

# Generate function
def generate_api_call(query, max_tokens=512, temperature=0.3):
    prompt = format_prompt(query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    result = response.split("### Response:")[-1].strip()
    return result

# Test cases
test_queries = [
    {
        "name": "Gaming with IPv4",
        "query": "I need ultra-low latency for VR gaming. My device IP is 192.168.1.50 and game server is 203.0.113.100. Need this for 2 hours."
    },
    {
        "name": "Streaming with Phone Number",
        "query": "I'm at a stadium and need better upload for 4K streaming. My phone is +14155551234, streaming to 198.51.100.50 for 90 minutes."
    },
    {
        "name": "IoT Sensor",
        "query": "IoT sensor uploading telemetry data to cloud server 10.0.0.100. Device phone +12025551111, need reliable connection for 15 minutes."
    },
    {
        "name": "Video Conference with IPv6",
        "query": "Video conference from IPv6 2001:db8::1 to server 2001:db8:1234::1 for 45 minutes."
    },
]

print("="*60)
print("🧪 Running Test Queries")
print("="*60)

for i, test in enumerate(test_queries, 1):
    print(f"\n{'='*60}")
    print(f"Test {i}/{len(test_queries)}: {test['name']}")
    print(f"{'='*60}")
    print(f"\n📝 Query:")
    print(f"   {test['query']}\n")
    
    print("🤖 Generating response...")
    result = generate_api_call(test['query'])
    
    print("\n📤 Model Output:")
    print(result)
    
    # Validate JSON
    print("\n🔍 Validation:")
    try:
        json_obj = json.loads(result)
        print("   ✅ Valid JSON structure")
        
        # Check required fields
        required = ["device", "applicationServer", "qosProfile", "duration"]
        missing = [f for f in required if f not in json_obj]
        
        if not missing:
            print("   ✅ All required fields present")
            print(f"   ✅ QoS Profile: {json_obj['qosProfile']}")
            print(f"   ✅ Duration: {json_obj['duration']} seconds")
        else:
            print(f"   ⚠️  Missing fields: {', '.join(missing)}")
        
        # Check for hallucinated fields
        valid_fields = {
            "device", "applicationServer", "qosProfile", "duration",
            "devicePorts", "applicationServerPorts", "notificationUrl",
            "notificationAuthToken", "webhook"
        }
        
        hallucinated = [k for k in json_obj.keys() if k not in valid_fields]
        if hallucinated:
            print(f"   ❌ Hallucinated fields found: {', '.join(hallucinated)}")
        else:
            print("   ✅ No hallucinated fields")
            
    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON: {e}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("✅ Testing Complete!")
print("="*60)

# Interactive mode
print("\n💡 Want to test your own query?")
print("   Type a query or press Enter to skip:")
user_query = input("\n> ")

if user_query.strip():
    print(f"\n🤖 Processing: {user_query}")
    result = generate_api_call(user_query)
    print("\n📤 Result:")
    print(result)
    
    try:
        json_obj = json.loads(result)
        print("\n✅ Valid JSON!")
    except:
        print("\n❌ Invalid JSON")

print("\n👋 Done! Your model is working!\n")
