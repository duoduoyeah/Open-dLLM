# Trace Utils - Model Tracing and Activation Extraction

A comprehensive toolkit for extracting intermediate activations and tracing language model generation with detailed probability tracking.

## Overview

This module provides two main capabilities:

1. **Layer Token Hook**: Extract hidden states from specific layers and token positions during model inference
2. **Generation Tracing**: Track probability distributions and token selections during text generation

## Installation

The utilities are part of the `veomni` package. No additional installation required if you have the main dependencies:

```bash
pip install torch transformers  # Core dependencies
```

## Quick Start

### 1. Extract Layer Activations

```python
from veomni.utils.trace_utils import LayerTokenHook
import torch

# Hook into layer 5, last token position
with LayerTokenHook(model, layer_id=5, token_id=-1) as hook:
    outputs = model(**inputs)
    data = hook.get_data()

print(f"Hidden states shape: {data.hidden_states.shape}")
print(f"Mean activation: {data.hidden_states.mean()}")
```

### 2. Trace Generation

```python
from veomni.utils.trace_utils import decode

trace = decode(
    model=model,
    prompt="The future of AI",
    tokenizer=tokenizer,
    max_new_tokens=20,
    temperature=0.8
)

# Analyze each generation step
for step in trace.steps:
    print(f"Step {step.step}: token={step.selected_token_id}, prob={step.selected_token_prob:.4f}")
```

## API Reference

### Data Structures

#### LayerTokenData

Container for layer activation data.

```python
@dataclass
class LayerTokenData:
    layer_id: int                           # Layer index
    token_id: int                           # Token position index
    hidden_states: torch.Tensor            # Hidden state activations
    attention_output: Optional[torch.Tensor]  # Attention outputs (if available)
    mlp_output: Optional[torch.Tensor]     # MLP outputs (if available)
    layer_name: Optional[str]              # Layer module name
    metadata: Dict[str, Any]               # Additional metadata
```

**Methods:**
- `to(device)`: Move tensors to device
- `cpu()`: Move tensors to CPU
- `detach()`: Detach from computation graph

#### GenerationStep

Information about a single generation step.

```python
@dataclass
class GenerationStep:
    step: int                              # Step index
    logits: torch.Tensor                   # Raw logits
    probs: torch.Tensor                    # Probability distribution
    top_k_probs: Optional[torch.Tensor]    # Top-k probabilities
    top_k_indices: Optional[torch.Tensor]  # Top-k token indices
    selected_token_id: int                 # Selected token ID
    selected_token_prob: float             # Probability of selected token
    temperature: float                     # Temperature used
    metadata: Dict[str, Any]               # Additional metadata
```

#### GenerationTrace

Complete trace of generation process.

```python
@dataclass
class GenerationTrace:
    prompt_text: str                       # Input prompt
    prompt_tokens: List[int]               # Prompt token IDs
    steps: List[GenerationStep]            # Generation steps
    generated_tokens: List[int]            # Generated token IDs
    generated_text: str                    # Decoded generated text
    metadata: Dict[str, Any]               # Additional metadata
```

**Methods:**
- `add_step(step)`: Add a generation step
- `get_token_probabilities()`: Get list of (token_id, probability) pairs
- `to_dict()`: Convert to dictionary for serialization

### Functions

#### layer_token_hook

Extract activation from a specific layer and token position.

```python
def layer_token_hook(
    model: nn.Module,
    layer_id: int,
    token_id: int
) -> LayerTokenHook
```

**Parameters:**
- `model`: The PyTorch model
- `layer_id`: Layer index (0-indexed)
- `token_id`: Token position (-1 for last token)

**Returns:** `LayerTokenHook` context manager

**Example:**
```python
with layer_token_hook(model, layer_id=3, token_id=-1) as hook:
    outputs = model(input_ids)
    data = hook.get_data()
```

#### decode

Trace generation with detailed probability tracking.

```python
def decode(
    model: nn.Module,
    prompt: str,
    tokenizer: Any,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    do_sample: bool = True,
) -> GenerationTrace
```

**Parameters:**
- `model`: The language model
- `prompt`: Input text prompt
- `tokenizer`: Tokenizer for encoding/decoding
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_k`: Top-k sampling parameter
- `top_p`: Nucleus sampling parameter
- `do_sample`: Use sampling vs greedy decoding

**Returns:** `GenerationTrace` with detailed step information

**Example:**
```python
trace = decode(
    model=model,
    prompt="Hello world",
    tokenizer=tokenizer,
    max_new_tokens=10,
    temperature=0.8,
    top_k=50
)

# Access generation details
for step in trace.steps:
    print(f"Token: {step.selected_token_id}, Prob: {step.selected_token_prob}")
```

#### capture_layer_activations

Context manager to capture activations from multiple layers.

```python
@contextmanager
def capture_layer_activations(
    model: nn.Module,
    layer_ids: List[int],
    token_id: int = -1
) -> Dict[int, LayerTokenData]
```

**Parameters:**
- `model`: The PyTorch model
- `layer_ids`: List of layer indices to capture
- `token_id`: Token position to capture (-1 for last)

**Returns:** Dictionary mapping layer_id to LayerTokenData

**Example:**
```python
with capture_layer_activations(model, [0, 5, 10]) as captures:
    outputs = model(input_ids)

for layer_id, data in captures.items():
    print(f"Layer {layer_id}: {data.hidden_states.shape}")
```

## Usage Examples

### Example 1: Analyze Layer Representations

Extract and compare activations across multiple layers:

```python
from veomni.utils.trace_utils import capture_layer_activations
import torch

# Extract from layers 0, 3, 6, 9
layer_ids = [0, 3, 6, 9]

with capture_layer_activations(model, layer_ids, token_id=-1) as captures:
    outputs = model(input_ids)

# Compute similarity between layers
for i in range(len(layer_ids) - 1):
    layer_a = layer_ids[i]
    layer_b = layer_ids[i + 1]
    
    act_a = captures[layer_a].hidden_states
    act_b = captures[layer_b].hidden_states
    
    similarity = torch.nn.functional.cosine_similarity(
        act_a.unsqueeze(0),
        act_b.unsqueeze(0)
    )
    
    print(f"Layer {layer_a} -> {layer_b}: {similarity.item():.4f}")
```

### Example 2: Find Low-Confidence Predictions

Identify tokens where the model has low confidence:

```python
from veomni.utils.trace_utils import decode

trace = decode(
    model=model,
    prompt="The future of artificial intelligence",
    tokenizer=tokenizer,
    max_new_tokens=50,
    do_sample=False  # Greedy
)

# Find low-confidence predictions
threshold = 0.1
low_conf = [
    (step.step, step.selected_token_id, step.selected_token_prob)
    for step in trace.steps
    if step.selected_token_prob < threshold
]

for step_idx, token_id, prob in low_conf:
    token_text = tokenizer.decode([token_id])
    print(f"Step {step_idx}: '{token_text}' (prob={prob:.4f})")
```

### Example 3: Compare Sampling Strategies

Compare different decoding strategies:

```python
from veomni.utils.trace_utils import decode

prompt = "Once upon a time"

# Greedy decoding
trace_greedy = decode(model, prompt, tokenizer, do_sample=False)

# Low temperature sampling
trace_low_temp = decode(model, prompt, tokenizer, temperature=0.5, do_sample=True)

# High temperature sampling
trace_high_temp = decode(model, prompt, tokenizer, temperature=1.5, do_sample=True)

# Compare results
traces = [
    ("Greedy", trace_greedy),
    ("Low Temp", trace_low_temp),
    ("High Temp", trace_high_temp)
]

for name, trace in traces:
    avg_prob = sum(s.selected_token_prob for s in trace.steps) / len(trace.steps)
    print(f"{name}: avg_prob={avg_prob:.4f}, unique_tokens={len(set(trace.generated_tokens))}")
```

### Example 4: Probe Specific Token Representations

Extract representations for specific tokens:

```python
from veomni.utils.trace_utils import LayerTokenHook

text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt")

# Extract representation of "fox" (token position 3)
token_positions = [3]  # Adjust based on tokenization

layer_id = 6
activations = []

for pos in token_positions:
    with LayerTokenHook(model, layer_id, pos) as hook:
        outputs = model(**inputs)
        data = hook.get_data()
        activations.append(data.hidden_states)

# Analyze the extracted activation
activation = activations[0]
print(f"Shape: {activation.shape}")
print(f"L2 norm: {torch.norm(activation).item():.4f}")
```

### Example 5: Generation Analysis Pipeline

Complete pipeline for analyzing generation:

```python
from veomni.utils.trace_utils import decode
import matplotlib.pyplot as plt

# Generate with tracing
trace = decode(
    model=model,
    prompt="Machine learning is",
    tokenizer=tokenizer,
    max_new_tokens=30,
    temperature=0.8
)

# Extract probabilities
probs = [step.selected_token_prob for step in trace.steps]

# Plot probability over time
plt.figure(figsize=(12, 4))
plt.plot(probs, marker='o')
plt.xlabel('Generation Step')
plt.ylabel('Token Probability')
plt.title('Model Confidence During Generation')
plt.grid(True)
plt.savefig('generation_confidence.png')

# Print statistics
print(f"Generated: {trace.generated_text}")
print(f"Avg probability: {sum(probs)/len(probs):.4f}")
print(f"Min probability: {min(probs):.4f}")
print(f"Max probability: {max(probs):.4f}")
```

## Advanced Usage

### Custom Hook Functions

Create custom hooks for specific analysis:

```python
from veomni.utils.trace_utils import LayerTokenHook

class CustomAnalysisHook(LayerTokenHook):
    def _hook_fn(self, module, input, output):
        # Custom analysis logic
        result = super()._hook_fn(module, input, output)
        
        # Add custom metadata
        if self.captured_data:
            self.captured_data.metadata['custom_metric'] = compute_custom_metric(output)
        
        return result

# Use custom hook
with CustomAnalysisHook(model, 5, -1) as hook:
    outputs = model(input_ids)
    data = hook.get_data()
    print(data.metadata['custom_metric'])
```

### Batched Processing

Process multiple inputs:

```python
from veomni.utils.trace_utils import decode

prompts = [
    "The quick brown",
    "Machine learning is",
    "Once upon a time"
]

traces = []
for prompt in prompts:
    trace = decode(model, prompt, tokenizer, max_new_tokens=20)
    traces.append(trace)

# Compare traces
for i, trace in enumerate(traces):
    avg_prob = sum(s.selected_token_prob for s in trace.steps) / len(trace.steps)
    print(f"Prompt {i}: avg_prob={avg_prob:.4f}")
```

## Supported Models

The utilities work with any PyTorch model following standard conventions:

- ✅ Hugging Face Transformers (GPT-2, LLaMA, Qwen, etc.)
- ✅ Custom transformer models
- ✅ Models with standard layer structure

### Model Compatibility Notes

The `LayerTokenHook` tries to find layers using common naming patterns:
- `model.layers.{layer_id}` - Common for many models
- `transformer.h.{layer_id}` - GPT-style
- `model.decoder.layers.{layer_id}` - Decoder models
- `encoder.layer.{layer_id}` - BERT-style

If your model uses different naming, you may need to extend the `_get_layer_module` method.

## Performance Considerations

1. **Memory**: Capturing activations stores tensors in memory. Use `.cpu()` or `.detach()` for large models.

2. **Computation**: Hooks add minimal overhead (~1-2% typically).

3. **Best Practices**:
   - Use `torch.no_grad()` for inference
   - Detach captured tensors if not needed for gradients
   - Capture only necessary layers/tokens

Example for large-scale analysis:
```python
# Memory-efficient capture
with capture_layer_activations(model, [0, 6, 11]) as captures:
    with torch.no_grad():
        outputs = model(input_ids)

# Move to CPU immediately
for data in captures.values():
    data.cpu().detach()
```

## Testing

Run the test suite:

```bash
cd /home/shiyuan/workspace/interestRepo/Open-dLLM
python tests/test_trace_utils.py
```

Run examples:

```bash
python examples/trace_utils_examples.py
```

## Troubleshooting

### Issue: "Could not find layer X in model"

**Solution:** Check your model's layer naming:
```python
# Print all layer names
for name, module in model.named_modules():
    if 'layer' in name.lower():
        print(name)
```

### Issue: "Hook not capturing data"

**Solution:** Ensure the hook is active during forward pass:
```python
# Correct usage
with LayerTokenHook(model, 5, -1) as hook:
    outputs = model(input_ids)  # Forward pass inside context
    data = hook.get_data()

# Incorrect - forward pass outside context
hook = LayerTokenHook(model, 5, -1)
outputs = model(input_ids)  # Won't capture!
```

### Issue: Out of memory with many layers

**Solution:** Capture and process in batches:
```python
layer_groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

all_captures = {}
for group in layer_groups:
    with capture_layer_activations(model, group) as captures:
        outputs = model(input_ids)
    
    # Process immediately and free memory
    for layer_id, data in captures.items():
        all_captures[layer_id] = data.cpu().detach()
```

## Contributing

Contributions welcome! Areas for enhancement:
- Support for more model architectures
- Attention pattern extraction
- Gradient-based analysis
- Visualization utilities

## License

Same as the main Open-dLLM project.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{open_dllm_trace_utils,
  title={Trace Utils for Model Analysis},
  author={Open-dLLM Contributors},
  year={2025},
  url={https://github.com/duoduoyeah/Open-dLLM}
}
```
