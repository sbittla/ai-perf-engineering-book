# Part I — Foundations

Companion exercises for **Part I — Foundations** of *AI Systems Performance Engineering*.

Each folder name matches a chapter title exactly as it appears in the book.

## Structure

```
I.Foundations/
├── 1.What_Is_AI_Performance_Engineering/
│   └── roofline_model.py
├── 2.PyTorch_Fundamentals/
│   ├── tensors.py
│   ├── autograd.py
│   ├── nn_modules.py
│   ├── training_loop.py
│   ├── gpu_timing.py
│   └── common_mistakes.py
└── 3.The_AI_Hardware_Stack/
    ├── cpu_and_memory.py
    ├── gpu_memory_and_compute.py
    └── hardware_survey.py
```

## Setup

```bash
git clone https://github.com/your-handle/ai-perf-engineering-book
cd ai-perf-engineering-book

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Running the exercises (in order)

```bash
# Chapter 1
python "I.Foundations/1.What_Is_AI_Performance_Engineering/roofline_model.py"

# Chapter 2
python "I.Foundations/2.PyTorch_Fundamentals/tensors.py"
python "I.Foundations/2.PyTorch_Fundamentals/autograd.py"
python "I.Foundations/2.PyTorch_Fundamentals/nn_modules.py"
python "I.Foundations/2.PyTorch_Fundamentals/training_loop.py"
python "I.Foundations/2.PyTorch_Fundamentals/gpu_timing.py"
python "I.Foundations/2.PyTorch_Fundamentals/common_mistakes.py"

# Chapter 3
python "I.Foundations/3.The_AI_Hardware_Stack/cpu_and_memory.py"
python "I.Foundations/3.The_AI_Hardware_Stack/gpu_memory_and_compute.py"
python "I.Foundations/3.The_AI_Hardware_Stack/hardware_survey.py"
```

Each file prints `✓` for every passing section and stops with a clear error at the first incomplete TODO.

## Exercise map — book section to file

| Book Section | File |
|---|---|
| 1.1–1.6 Roofline model, MFU, economics | `1.What_Is_AI_Performance_Engineering/roofline_model.py` |
| 2.1 Tensors | `2.PyTorch_Fundamentals/tensors.py` |
| 2.2 Autograd | `2.PyTorch_Fundamentals/autograd.py` |
| 2.3 nn.Module | `2.PyTorch_Fundamentals/nn_modules.py` |
| 2.4 Training loop | `2.PyTorch_Fundamentals/training_loop.py` |
| 2.5 GPU timing | `2.PyTorch_Fundamentals/gpu_timing.py` |
| 2.6 Common mistakes | `2.PyTorch_Fundamentals/common_mistakes.py` |
| 3.1–3.2 CPU cache, PCIe | `3.The_AI_Hardware_Stack/cpu_and_memory.py` |
| 3.3–3.4 HBM, Tensor Cores | `3.The_AI_Hardware_Stack/gpu_memory_and_compute.py` |
| 3.5 GPU specs, MFU | `3.The_AI_Hardware_Stack/hardware_survey.py` |

## Hardware requirements

- **Chapters 1–2:** CPU-only is fine. CUDA sections skip gracefully.
- **Chapter 3:** A CUDA GPU is required for most measurement sections (RTX 3060 / 8 GB minimum).
