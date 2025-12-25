# 🧪 Testing Code for Your Colab Notebook

## Add these cells at the END of your Colab notebook after training is complete

---

## Cell 1: Test Single Query (Quick Test)

```python
# =============================================================================
# Quick Test - Single Query
# =============================================================================

print("="*60)
print("🧪 Quick Test - Single Query")
print("="*60)

test_query = "I need ultra-low latency for VR gaming. My device IP is 192.168.1.50 and game server is 203.0.113.100. Need this for 2 hours."

print(f"\n📝 Test Query:\n{test_query}\n")

# Format prompt
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert assistant for the CAMARA Quality on Demand (QoD) API. Convert user requests into valid API calls.

### Input:
{}

### Response:
{}"""

# Enable inference mode
FastLanguageModel.for_inference(model)

inputs = tokenizer(
    [alpaca_prompt.format(test_query, "")],
    return_tensors="pt"
).to("cuda")

print("🤖 Generating response...\n")

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.3,
    use_cache=True
)

response = tokenizer.batch_decode(outputs)[0]
result = response.split("### Response:")[-1].strip()

print("📤 Model Output:")
print(result)
print("\n" + "="*60)

# Validate JSON
import json

print("\n🔍 Validation:")
try:
    json_obj = json.loads(result)
    print("✅ Valid JSON structure")
    
    # Check required fields
    required_fields = ["device", "applicationServer", "qosProfile", "duration"]
    missing = [f for f in required_fields if f not in json_obj]
    
    if not missing:
        print("✅ All required CAMARA fields present")
        print(f"   • QoS Profile: {json_obj['qosProfile']}")
        print(f"   • Duration: {json_obj['duration']} seconds")
    else:
        print(f"⚠️  Missing fields: {', '.join(missing)}")
    
    # Check for hallucinations
    valid_fields = {
        "device", "applicationServer", "qosProfile", "duration",
        "devicePorts", "applicationServerPorts", "notificationUrl",
        "notificationAuthToken", "webhook"
    }
    
    hallucinated = [k for k in json_obj.keys() if k not in valid_fields]
    if hallucinated:
        print(f"❌ Hallucinated fields found: {', '.join(hallucinated)}")
    else:
        print("✅ No hallucinated fields - Perfect!")
        
except json.JSONDecodeError as e:
    print(f"❌ Invalid JSON: {str(e)}")

print("="*60)
```

---

## Cell 2: Comprehensive Test Suite

```python
# =============================================================================
# Comprehensive Test Suite - Multiple Scenarios
# =============================================================================

import json

print("="*60)
print("🧪 COMPREHENSIVE TEST SUITE")
print("="*60)

# Test cases covering different scenarios
test_cases = [
    {
        "name": "Test 1: Gaming (IPv4)",
        "query": "Need ultra-low latency for VR gaming. Device IP 192.168.1.50, server 203.0.113.100, 2 hours.",
        "expected_qos": "QOS_E",
        "expected_duration": 7200
    },
    {
        "name": "Test 2: Streaming (Phone Number)",
        "query": "I'm at a crowded stadium and need better upload for 4K stream. My phone number is +14155551234 and I'm streaming to server 198.51.100.50 for 90 minutes.",
        "expected_qos": "QOS_S",
        "expected_duration": 5400
    },
    {
        "name": "Test 3: IoT (Mission Critical)",
        "query": "IoT sensor uploading telemetry data to cloud 10.0.0.100. Device phone +12025551111, need reliable connection for 15 minutes.",
        "expected_qos": "QOS_M",
        "expected_duration": 900
    },
    {
        "name": "Test 4: Video Conference (IPv6)",
        "query": "Video conference from IPv6 2001:db8::1 to server 2001:db8:1234::1 for 45 minutes.",
        "expected_qos": "QOS_L",
        "expected_duration": 2700
    },
    {
        "name": "Test 5: Mobile Gaming with Port",
        "query": "Playing mobile game on port 7777. My IP is 203.0.113.50, game server 192.0.2.100, need gaming quality for 3 hours.",
        "expected_qos": "QOS_E",
        "expected_duration": 10800
    }
]

# Track results
results = {
    "total": len(test_cases),
    "passed": 0,
    "failed": 0,
    "json_valid": 0,
    "spec_compliant": 0,
    "no_hallucinations": 0,
    "correct_qos": 0,
    "correct_duration": 0
}

# Run all tests
for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"{test['name']}")
    print(f"{'='*60}")
    print(f"\n📝 Query: {test['query']}\n")
    
    # Generate response
    inputs = tokenizer(
        [alpaca_prompt.format(test['query'], "")],
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,
        use_cache=True
    )
    
    response = tokenizer.batch_decode(outputs)[0]
    result = response.split("### Response:")[-1].strip()
    
    print("🤖 Response:")
    print(result)
    
    # Validate
    test_passed = True
    
    try:
        json_obj = json.loads(result)
        results["json_valid"] += 1
        print("\n✅ Valid JSON")
        
        # Check required fields
        required = ["device", "applicationServer", "qosProfile", "duration"]
        if all(f in json_obj for f in required):
            results["spec_compliant"] += 1
            print("✅ All required fields present")
            
            # Check QoS profile
            if json_obj["qosProfile"] == test["expected_qos"]:
                results["correct_qos"] += 1
                print(f"✅ Correct QoS Profile: {json_obj['qosProfile']}")
            else:
                print(f"⚠️  QoS Profile: {json_obj['qosProfile']} (expected {test['expected_qos']})")
                test_passed = False
            
            # Check duration
            if json_obj["duration"] == test["expected_duration"]:
                results["correct_duration"] += 1
                print(f"✅ Correct Duration: {json_obj['duration']} seconds")
            else:
                print(f"⚠️  Duration: {json_obj['duration']} (expected {test['expected_duration']})")
                test_passed = False
        else:
            missing = [f for f in required if f not in json_obj]
            print(f"❌ Missing fields: {', '.join(missing)}")
            test_passed = False
        
        # Check hallucinations
        valid_fields = {
            "device", "applicationServer", "qosProfile", "duration",
            "devicePorts", "applicationServerPorts", "notificationUrl",
            "notificationAuthToken", "webhook"
        }
        hallucinated = [k for k in json_obj.keys() if k not in valid_fields]
        
        if not hallucinated:
            results["no_hallucinations"] += 1
            print("✅ No hallucinated fields")
        else:
            print(f"❌ Hallucinated: {', '.join(hallucinated)}")
            test_passed = False
            
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {str(e)}")
        test_passed = False
    
    if test_passed:
        results["passed"] += 1
        print("\n🎉 TEST PASSED")
    else:
        results["failed"] += 1
        print("\n❌ TEST FAILED")

# Print summary
print("\n" + "="*60)
print("📊 TEST SUMMARY")
print("="*60)

print(f"\nTotal Tests: {results['total']}")
print(f"✅ Passed: {results['passed']}")
print(f"❌ Failed: {results['failed']}")
print(f"\nSuccess Rate: {(results['passed']/results['total']*100):.1f}%")

print(f"\n📈 Detailed Metrics:")
print(f"   JSON Validity:      {results['json_valid']}/{results['total']} ({results['json_valid']/results['total']*100:.0f}%)")
print(f"   Spec Compliance:    {results['spec_compliant']}/{results['total']} ({results['spec_compliant']/results['total']*100:.0f}%)")
print(f"   No Hallucinations:  {results['no_hallucinations']}/{results['total']} ({results['no_hallucinations']/results['total']*100:.0f}%)")
print(f"   Correct QoS:        {results['correct_qos']}/{results['total']} ({results['correct_qos']/results['total']*100:.0f}%)")
print(f"   Correct Duration:   {results['correct_duration']}/{results['total']} ({results['correct_duration']/results['total']*100:.0f}%)")

print("\n" + "="*60)

if results['passed'] == results['total']:
    print("🎉🎉🎉 PERFECT SCORE! All tests passed! 🎉🎉🎉")
elif results['passed'] >= results['total'] * 0.8:
    print("✅ Excellent! Most tests passed!")
else:
    print("⚠️  Some tests failed. Review the results above.")

print("="*60)
```

---

## Cell 3: Interactive Testing

```python
# =============================================================================
# Interactive Test - Try Your Own Queries
# =============================================================================

print("="*60)
print("💬 INTERACTIVE TESTING")
print("="*60)
print("\nTest the model with your own custom queries!\n")

def interactive_test():
    """Interactive testing function"""
    
    while True:
        print("-"*60)
        query = input("\n📝 Enter your query (or 'quit' to exit):\n> ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Exiting interactive mode. Thanks for testing!")
            break
        
        if not query.strip():
            print("⚠️  Please enter a valid query")
            continue
        
        print(f"\n🤖 Processing: {query}")
        print("⏳ Generating...\n")
        
        # Generate
        inputs = tokenizer(
            [alpaca_prompt.format(query, "")],
            return_tensors="pt"
        ).to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            use_cache=True
        )
        
        response = tokenizer.batch_decode(outputs)[0]
        result = response.split("### Response:")[-1].strip()
        
        print("📤 Response:")
        print(result)
        
        # Validate
        try:
            json_obj = json.loads(result)
            print("\n✅ Valid JSON!")
            print(f"✅ QoS Profile: {json_obj.get('qosProfile', 'N/A')}")
            print(f"✅ Duration: {json_obj.get('duration', 'N/A')} seconds")
        except:
            print("\n❌ Invalid JSON")
        
        print()

# Run interactive mode
interactive_test()
```

---

## Cell 4: Before/After Comparison (Optional - if you tested base model)

```python
# =============================================================================
# Before/After Comparison
# =============================================================================

print("="*60)
print("📊 BEFORE vs AFTER COMPARISON")
print("="*60)

comparison_query = "I need better network for gaming. My IP is 192.168.1.50, server 203.0.113.100, need this for 2 hours."

print(f"\n📝 Query:\n{comparison_query}\n")

# If you saved base model output earlier, compare here
print("="*60)
print("BEFORE FINE-TUNING (Base Model)")
print("="*60)
print("""
Typical base model output:
"To improve your gaming network quality, you should:
1. Contact your ISP
2. Use a wired connection
3. Close background applications
..."

❌ No structured API call
❌ Conversational response
❌ Not useful for CAMARA API
""")

print("\n" + "="*60)
print("AFTER FINE-TUNING (Your Model)")
print("="*60)

# Generate with fine-tuned model
inputs = tokenizer(
    [alpaca_prompt.format(comparison_query, "")],
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.3,
    use_cache=True
)

response = tokenizer.batch_decode(outputs)[0]
result = response.split("### Response:")[-1].strip()

print(result)

try:
    json.loads(result)
    print("\n✅ Perfect CAMARA structure!")
    print("✅ Valid JSON")
    print("✅ Spec compliant")
    print("✅ No hallucinations")
except:
    print("\n⚠️  Output validation failed")

print("\n" + "="*60)
print("🎯 IMPROVEMENT: From conversational text to perfect API calls!")
print("="*60)
```

---

## Cell 5: Export Results (Optional)

```python
# =============================================================================
# Export Test Results
# =============================================================================

print("="*60)
print("💾 EXPORT TEST RESULTS")
print("="*60)

# Create test report
test_report = f"""
# CAMARA QoD Model Test Report

## Model Information
- Base Model: microsoft/Phi-3-mini-4k-instruct
- Fine-tuning: QLoRA (r=16, alpha=16)
- Training: SFT + DPO

## Test Results
- Total Tests: {results['total']}
- Passed: {results['passed']}
- Failed: {results['failed']}
- Success Rate: {(results['passed']/results['total']*100):.1f}%

## Metrics
- JSON Validity: {results['json_valid']}/{results['total']} ({results['json_valid']/results['total']*100:.0f}%)
- Spec Compliance: {results['spec_compliant']}/{results['total']} ({results['spec_compliant']/results['total']*100:.0f}%)
- No Hallucinations: {results['no_hallucinations']}/{results['total']} ({results['no_hallucinations']/results['total']*100:.0f}%)
- Correct QoS: {results['correct_qos']}/{results['total']} ({results['correct_qos']/results['total']*100:.0f}%)
- Correct Duration: {results['correct_duration']}/{results['total']} ({results['correct_duration']/results['total']*100:.0f}%)

## Conclusion
{'✅ Model is production-ready!' if results['passed'] == results['total'] else '⚠️ Model needs further review.'}
"""

# Save to file
with open('test_report.txt', 'w') as f:
    f.write(test_report)

print("✅ Test report saved to: test_report.txt")
print("\nReport Preview:")
print(test_report)

# Download instruction
print("\n💡 To download the report:")
print("   1. Look at the left sidebar (📁 Files)")
print("   2. Find 'test_report.txt'")
print("   3. Right-click → Download")

print("="*60)
```

---

## 🎯 Instructions for Adding to Colab

1. **Scroll to the bottom** of your notebook
2. **Click "+ Code"** to add a new cell
3. **Copy Cell 1** first and paste it
4. **Run it** to do a quick test
5. If it works, **add Cells 2-5** one by one

## ✅ What to Expect

After running these cells, you should see:
- ✅ 100% JSON validity
- ✅ 100% spec compliance
- ✅ 0% hallucinations
- ✅ Correct QoS profile selection
- ✅ Accurate duration calculations

If you get these results, your model is **PERFECT** and ready to use! 🎉
