# Trace Utils - Quick Reference

## Installation

```bash
# Already included in veomni
from veomni.utils.trace_utils import LayerTokenHook, decode, capture_layer_activations
```

## Quick Examples

### 1. Extract Layer Activation

```python
from veomni.utils.trace_utils import LayerTokenHook

# Hook layer 5, last token
with LayerTokenHook(model, layer_id=5, token_id=-1) as hook:
    outputs = model(input_ids)
    data = hook.get_data()

print(data.hidden_states.shape)  # torch.Size([hidden_dim])
```

### 2. Trace Generation

```python
from veomni.utils.trace_utils import decode

trace = decode(
    model=model,
    prompt="Hello world",
    tokenizer=tokenizer,
    max_new_tokens=10
)

# Access details
for step in trace.steps:
    print(f"Token: {step.selected_token_id}, Prob: {step.selected_token_prob:.4f}")
```

### 3. Multiple Layers

```python
from veomni.utils.trace_utils import capture_layer_activations

with capture_layer_activations(model, [0, 5, 10]) as captures:
    outputs = model(input_ids)

for layer_id, data in captures.items():
    print(f"Layer {layer_id}: {data.hidden_states.mean():.4f}")
```

## API Summary

### Data Structures

| Class | Purpose |
|-------|---------|
| `LayerTokenData` | Stores layer activation data |
| `GenerationStep` | Single generation step info |
| `GenerationTrace` | Complete generation trace |

### Functions

| Function | Purpose |
|----------|---------|
| `layer_token_hook(model, layer_id, token_id)` | Extract from one layer/token |
| `decode(model, prompt, tokenizer, ...)` | Trace generation |
| `capture_layer_activations(model, layer_ids, token_id)` | Extract from multiple layers |

## Common Patterns

### Pattern 1: Analyze Model Confidence

```python
trace = decode(model, prompt, tokenizer, max_new_tokens=20, do_sample=False)

# Find low-confidence predictions
low_conf = [s for s in trace.steps if s.selected_token_prob < 0.1]
print(f"Found {len(low_conf)} low-confidence predictions")
```

### Pattern 2: Compare Layers

```python
with capture_layer_activations(model, [0, 6, 11]) as captures:
    outputs = model(input_ids)

# Compute similarities
layer_ids = sorted(captures.keys())
for i in range(len(layer_ids) - 1):
    similarity = torch.nn.functional.cosine_similarity(
        captures[layer_ids[i]].hidden_states.unsqueeze(0),
        captures[layer_ids[i+1]].hidden_states.unsqueeze(0)
    )
    print(f"Layer {layer_ids[i]} -> {layer_ids[i+1]}: {similarity.item():.4f}")
```

### Pattern 3: Memory-Efficient Capture

```python
# For large models, immediately move to CPU
with capture_layer_activations(model, [0, 5, 10]) as captures:
    with torch.no_grad():
        outputs = model(input_ids)

# Process and free GPU memory
results = {}
for layer_id, data in captures.items():
    results[layer_id] = data.cpu().detach()
```

## Full Documentation

See [docs/trace_utils_guide.md](../docs/trace_utils_guide.md) for complete documentation.

## Tests & Examples

```bash
# Run tests
python tests/test_trace_utils.py

# Run examples
python examples/trace_utils_examples.py
```
