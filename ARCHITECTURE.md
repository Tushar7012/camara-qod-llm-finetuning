# System Architecture - CAMARA QoD Fine-tuning Project

## High-Level System Overview

This document describes the complete system architecture of the CAMARA QoD API fine-tuning project, built using Supervised Fine-Tuning (SFT) with QLoRA and Direct Preference Optimization (DPO).

---

## System Architecture Diagram

```mermaid
graph TB
 subgraph "Data Layer"
 A1[CAMARA API Spec] --> B1[Dataset Generator]
 B1 --> C1[SFT Dataset<br/>50 examples]
 B1 --> C2[Preference Dataset<br/>30 pairs]
 end
 
 subgraph "Model Layer"
 D1[Phi-3-Mini-4K-Instruct<br/>3.8B params] --> E1[4-bit Quantization<br/>QLoRA]
 E1 --> F1[LoRA Adapters<br/>25M params]
 end
 
 subgraph "Training Pipeline"
 C1 --> G1[SFT Training<br/>3 epochs]
 G1 --> H1[SFT Checkpoint]
 H1 --> G2[DPO Training<br/>1 epoch]
 C2 --> G2
 G2 --> H2[Final Model]
 end
 
 subgraph "Inference Layer"
 H2 --> I1[Model Loading]
 I1 --> I2[Tokenization]
 I2 --> I3[Generation]
 I3 --> I4[JSON Validation]
 end
 
 subgraph "Deployment"
 I4 --> J1[API Endpoint]
 J1 --> J2[Production Usage]
 end
 
 style C1 fill:#e1f5ff
 style C2 fill:#e1f5ff
 style H1 fill:#fff4e1
 style H2 fill:#d4edda
 style J2 fill:#f8d7da
```

---

## Complete Training Pipeline

```mermaid
flowchart LR
 subgraph "Phase 1: Data Preparation"
 A[CAMARA API<br/>Documentation] --> B[Manual Curation]
 B --> C[SFT Dataset<br/>JSONL]
 B --> D[Preference Dataset<br/>JSONL]
 end
 
 subgraph "Phase 2: SFT Training"
 C --> E[Load Base Model<br/>Phi-3-Mini]
 E --> F[Apply QLoRA<br/>4-bit]
 F --> G[SFT Trainer<br/>3 epochs]
 G --> H[SFT Checkpoint]
 end
 
 subgraph "Phase 3: DPO Alignment"
 H --> I[Clone as Reference]
 I --> J[DPO Trainer]
 D --> J
 J --> K[Aligned Model]
 end
 
 subgraph "Phase 4: Evaluation"
 K --> L[Test Queries]
 L --> M[JSON Validation]
 M --> N[Spec Compliance]
 N --> O[Performance Report]
 end
 
 style C fill:#bbdefb
 style D fill:#bbdefb
 style H fill:#fff9c4
 style K fill:#c8e6c9
 style O fill:#ffccbc
```

---

## Component Architecture

### 1. **Data Components**

```mermaid
graph TD
 A[Raw CAMARA Spec] --> B[Data Processing]
 
 B --> C1[SFT Examples]
 B --> C2[Preference Pairs]
 
 C1 --> D1[Instruction]
 C1 --> D2[Input Query]
 C1 --> D3[Response JSON]
 
 C2 --> E1[Prompt]
 C2 --> E2[Chosen Response]
 C2 --> E3[Rejected Response]
 
 style C1 fill:#e3f2fd
 style C2 fill:#f3e5f5
```

**SFT Dataset Structure:**
```json
{
 "instruction": "System prompt",
 "input": "User request",
 "response": "Valid CAMARA JSON"
}
```

**Preference Dataset Structure:**
```json
{
 "prompt": "User request",
 "chosen": "CAMARA-compliant JSON",
 "rejected": "Hallucinated/wrong JSON"
}
```

---

### 2. **Model Architecture**

```mermaid
graph TB
 subgraph "Base Model"
 A[Phi-3-Mini-4K-Instruct<br/>3.8B parameters]
 end
 
 subgraph "Quantization Layer"
 B[BitsAndBytes<br/>4-bit NF4]
 C[Double Quantization]
 end
 
 subgraph "LoRA Adapters"
 D1[Query Projection]
 D2[Key Projection]
 D3[Value Projection]
 D4[Output Projection]
 D5[Gate Projection]
 D6[Up Projection]
 D7[Down Projection]
 end
 
 A --> B
 B --> C
 C --> D1 & D2 & D3 & D4 & D5 & D6 & D7
 
 D1 & D2 & D3 & D4 & D5 & D6 & D7 --> E[~25M Trainable<br/>Parameters]
 
 style A fill:#ffebee
 style B fill:#e8f5e9
 style C fill:#e8f5e9
 style E fill:#fff3e0
```

**Configuration:**
- **Rank (r):** 16
- **Alpha:** 16
- **Dropout:** 5%
- **Target Modules:** QKV, O, Gate, Up, Down projections
- **Trainable %:** 0.66% of total parameters

---

### 3. **Training Architecture**

#### SFT (Supervised Fine-Tuning)

```mermaid
sequenceDiagram
 participant D as Dataset
 participant T as Tokenizer
 participant M as Model
 participant L as Loss
 participant O as Optimizer
 
 D->>T: Formatted Prompts
 T->>M: Token IDs
 M->>M: Forward Pass
 M->>L: Predictions vs Targets
 L->>O: Compute Gradients
 O->>M: Update LoRA Weights
 
 Note over M: Repeat for 3 epochs
```

**Hyperparameters:**
- Learning Rate: 2e-4
- Batch Size: 2
- Gradient Accumulation: 4 steps
- Effective Batch Size: 8
- Optimizer: Paged AdamW 8-bit
- Scheduler: Linear warmup

---

#### DPO (Direct Preference Optimization)

```mermaid
graph TB
 subgraph "Input"
 A[Prompt]
 B[Chosen Response]
 C[Rejected Response]
 end
 
 subgraph "Model Inference"
 D1[Policy Model π_θ<br/>Trainable]
 D2[Reference Model π_ref<br/>Frozen]
 end
 
 subgraph "Probability Computation"
 E1[log π_θ chosen]
 E2[log π_θ rejected]
 E3[log π_ref chosen]
 E4[log π_ref rejected]
 end
 
 subgraph "Reward Calculation"
 F1[reward_chosen = β × log π_θ/π_ref]
 F2[reward_rejected = β × log π_θ/π_ref]
 end
 
 subgraph "Loss"
 G[DPO Loss = -log σ reward_chosen - reward_rejected]
 end
 
 A & B --> D1 & D2
 C --> D1 & D2
 
 D1 --> E1 & E2
 D2 --> E3 & E4
 
 E1 & E3 --> F1
 E2 & E4 --> F2
 
 F1 & F2 --> G
 G --> D1
 
 style D1 fill:#ffccbc
 style D2 fill:#c5cae9
 style G fill:#f8bbd0
```

**DPO Key Components:**
- **β (KL penalty):** 0.1
- **Policy Model:** Trainable (updated)
- **Reference Model:** Frozen (SFT checkpoint)
- **Objective:** Maximize preference probability

---

### 4. **Inference Pipeline**

```mermaid
flowchart LR
 A[User Query] --> B[Prompt Template]
 B --> C[Tokenization]
 C --> D[Model Generation]
 D --> E[Decode Output]
 E --> F{Extract JSON}
 
 F -->|Success| G[Validate Fields]
 F -->|Fail| H[Error: Invalid JSON]
 
 G -->|Valid| I[Return API Call]
 G -->|Invalid| J[Error: Missing Fields]
 
 style A fill:#e1f5fe
 style I fill:#c8e6c9
 style H fill:#ffcdd2
 style J fill:#ffcdd2
```

**Validation Checks:**
1. Valid JSON syntax
2. Required fields: `device`, `applicationServer`, `qosProfile`, `duration`
3. Correct data types
4. No hallucinated fields
5. Proper nesting structure

---

## Technology Stack

```mermaid
graph LR
 subgraph "Core Frameworks"
 A1[PyTorch 2.6]
 A2[Transformers 4.x]
 A3[PEFT LoRA]
 end
 
 subgraph "Training"
 B1[TRL SFTTrainer]
 B2[TRL DPOTrainer]
 B3[BitsAndBytes]
 end
 
 subgraph "Optimization"
 C1[Unsloth 2x Speed]
 C2[Flash Attention 2]
 C3[Gradient Checkpointing]
 end
 
 subgraph "Infrastructure"
 D1[Google Colab T4]
 D2[Kaggle GPU]
 D3[Local NVIDIA GPU]
 end
 
 A1 & A2 & A3 --> B1 & B2 & B3
 B1 & B2 & B3 --> C1 & C2 & C3
 C1 & C2 & C3 --> D1 & D2 & D3
 
 style A1 fill:#ffebee
 style B1 fill:#e8f5e9
 style C1 fill:#e3f2fd
 style D1 fill:#fff3e0
```

---

## Data Flow Diagram

```mermaid
flowchart TD
 subgraph "Input Stage"
 A[User Natural<br/>Language Query]
 end
 
 subgraph "Processing Stage"
 B[System Prompt<br/>Injection]
 C[Alpaca Format<br/>Template]
 D[Tokenizer<br/>BPE Encoding]
 end
 
 subgraph "Model Stage"
 E[Quantized Model<br/>4-bit Weights]
 F[LoRA Adapters<br/>Forward Pass]
 G[Causal LM Head<br/>Next Token Prediction]
 end
 
 subgraph "Output Stage"
 H[Token IDs]
 I[Decoder<br/>Text Generation]
 J[JSON Extractor]
 K[CAMARA API Call]
 end
 
 A --> B --> C --> D
 D --> E --> F --> G
 G --> H --> I --> J --> K
 
 style A fill:#e1f5fe
 style D fill:#f3e5f5
 style F fill:#fff3e0
 style K fill:#c8e6c9
```

---

## Performance Optimization Strategies

### Memory Optimization

```mermaid
graph TD
 A[Full Model<br/>7.6 GB] -->|4-bit Quant| B[Quantized Model<br/>1.9 GB]
 B -->|LoRA| C[+ Adapters<br/>2.1 GB Total]
 C -->|Gradient Checkpoint| D[Training Memory<br/>11.8 GB]
 
 style A fill:#ffcdd2
 style B fill:#fff9c4
 style C fill:#c8e6c9
 style D fill:#bbdefb
```

**Techniques:**
1. **4-bit Quantization:** Reduces model size by 75%
2. **LoRA:** Only train 0.66% of parameters
3. **Gradient Checkpointing:** Trade compute for memory
4. **Paged AdamW:** Efficient optimizer memory usage

---

### Speed Optimization

| Technique | Speedup | Memory Saved |
|-----------|---------|--------------|
| Unsloth Kernels | 2x | 0% |
| Flash Attention 2 | 1.5x | 20% |
| Mixed Precision FP16 | 1.3x | 50% |
| Gradient Accumulation | 1x | Enables larger batch |

---

## Model Versioning

```mermaid
gitGraph
 commit id: "Base Phi-3-Mini"
 branch sft-training
 commit id: "SFT Epoch 1"
 commit id: "SFT Epoch 2"
 commit id: "SFT Epoch 3"
 commit id: "SFT Checkpoint" tag: "v1.0-sft"
 
 branch dpo-training
 commit id: "DPO Setup"
 commit id: "DPO Training"
 commit id: "DPO Complete" tag: "v2.0-final"
 
 checkout main
 merge sft-training
 merge dpo-training tag: "production"
```

**Checkpoints:**
1. **Base Model:** `microsoft/Phi-3-mini-4k-instruct`
2. **SFT Model:** `camara_qod_lora_model/` (after supervised training)
3. **DPO Model:** `camara_qod_dpo_model/` (after preference alignment)
4. **Merged Model:** `camara_qod_final_model/` (production-ready)

---

## Deployment Architecture

```mermaid
graph TB
 subgraph "Storage"
 A1[HuggingFace Hub]
 A2[Google Drive]
 A3[Local Storage]
 end
 
 subgraph "Model Loading"
 B[PEFT Model Loader]
 end
 
 subgraph "Inference Server"
 C1[FastAPI Endpoint]
 C2[Model Cache]
 C3[Request Queue]
 end
 
 subgraph "Client"
 D1[Web Interface]
 D2[API Client]
 D3[Mobile App]
 end
 
 A1 & A2 & A3 --> B
 B --> C1
 C1 --> C2
 C1 --> C3
 C3 --> D1 & D2 & D3
 
 style A1 fill:#e3f2fd
 style C1 fill:#fff3e0
 style D1 fill:#f3e5f5
```

---

## Metrics & Monitoring

```mermaid
graph LR
 subgraph "Training Metrics"
 A1[Loss Curve]
 A2[Perplexity]
 A3[Gradient Norm]
 end
 
 subgraph "Evaluation Metrics"
 B1[JSON Validity %]
 B2[Spec Compliance %]
 B3[Hallucination Rate]
 B4[QoS Accuracy]
 end
 
 subgraph "Inference Metrics"
 C1[Latency ms]
 C2[Throughput req/s]
 C3[Memory Usage GB]
 end
 
 A1 & A2 & A3 --> D[Training Monitor]
 B1 & B2 & B3 & B4 --> E[Quality Monitor]
 C1 & C2 & C3 --> F[Performance Monitor]
 
 D & E & F --> G[Dashboard]
 
 style G fill:#c8e6c9
```

---

## End-to-End Workflow

1. **Dataset Creation** → Manual curation from CAMARA spec
2. **SFT Training** → Learn API structure and field mappings
3. **DPO Training** → Eliminate hallucinations via preference learning
4. **Evaluation** → Validate on test queries
5. **Deployment** → Host model for inference
6. **Monitoring** → Track performance and quality

---

## Development Environment

```yaml
Hardware:
 - GPU: NVIDIA T4 (Colab) / GTX 1650 (Local)
 - VRAM: 16GB (T4) / 4GB (GTX 1650)
 - RAM: 12GB+ recommended

Software:
 - Python: 3.10+
 - CUDA: 11.8 / 12.1
 - PyTorch: 2.6.0+
 - Transformers: 4.36+
 - PEFT: 0.7+
 - TRL: 0.7+

Platforms:
 - Google Colab (Free T4 GPU)
 - Kaggle Notebooks (30hrs/week GPU)
 - Local NVIDIA GPU
 - Lightning AI Studios
```

---

## References

- **Model:** [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- **CAMARA Spec:** [QualityOnDemand API](https://github.com/camaraproject/QualityOnDemand)
- **DPO Paper:** [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)
- **LoRA Paper:** [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- **QLoRA Paper:** [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)
