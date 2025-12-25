# 🚀 Quick Model Test - Google Colab Version

## Instructions:
1. Open this notebook in Google Colab
2. Upload your `camara_qod_final_model` folder to Colab
3. Run all cells below

---

## Step 1: Install Dependencies
```python
!pip install -q peft transformers accelerate
```

## Step 2: Load the Model
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

print("Loading model...")

# Configuration
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "./camara_qod_final_model"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load your trained adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("✅ Model loaded!")
```

## Step 3: Test Function
```python
def test_query(query):
    """Test the model with a query"""
    
    # Format prompt
    instruction = "You are an expert assistant for the CAMARA Quality on Demand (QoD) API. Convert user requests into valid API calls."
    
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context.

### Instruction:
{instruction}

### Input:
{query}

### Response:
"""
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = response.split("### Response:")[-1].strip()
    
    # Display results
    print(f"Query: {query}\n")
    print(f"Response:\n{result}\n")
    
    # Validate
    try:
        json_obj = json.loads(result)
        print("✅ Valid JSON!")
        
        required = ["device", "applicationServer", "qosProfile", "duration"]
        if all(f in json_obj for f in required):
            print("✅ All required fields present")
            print(f"✅ QoS Profile: {json_obj['qosProfile']}")
            print(f"✅ Duration: {json_obj['duration']} seconds")
        else:
            print("⚠️ Missing required fields")
    except:
        print("❌ Invalid JSON")
    
    print("-" * 60)
    return result
```

## Step 4: Run Tests
```python
# Test Case 1: Gaming
test_query("Need ultra-low latency for VR gaming. Device IP 192.168.1.50, server 203.0.113.100, 2 hours")

# Test Case 2: Streaming
test_query("4K streaming from phone +14155551234 to server 198.51.100.50 for 90 minutes")

# Test Case 3: IoT
test_query("IoT sensor uploading to 10.0.0.100. Phone +12025551111, 15 minutes")
```

## Step 5: Test Your Own Query
```python
# Try your own!
my_query = "YOUR QUERY HERE"
test_query(my_query)
```

---

## 📊 Expected Results

For "Gaming, IP 192.168.1.50, 2 hours", you should see:

```json
{
  "device": {
    "ipv4Address": {
      "publicAddress": "192.168.1.50"
    }
  },
  "applicationServer": {
    "ipv4Address": "203.0.113.100"
  },
  "qosProfile": "QOS_E",
  "duration": 7200
}
```

✅ Valid JSON  
✅ All required fields  
✅ No hallucinated fields

---

## 🎯 What to Look For

**Good Signs:**
- ✅ Valid JSON structure
- ✅ Correct field names: `device`, `applicationServer`, `qosProfile`, `duration`
- ✅ Proper nesting (IPv4 inside `ipv4Address`, etc.)
- ✅ QoS profile matches use case (QOS_E for gaming, QOS_S for streaming)
- ✅ Duration in seconds

**Bad Signs:**
- ❌ Flat structure like `{"device_ip": "..."}`
- ❌ Fake fields like `bandwidth`, `quality_level`, `connection_type`
- ❌ Wrong duration format (minutes/hours instead of seconds)

---

## 📝 Notes

If you see 100% valid JSON with no hallucinations, your model is working perfectly! 🎉
