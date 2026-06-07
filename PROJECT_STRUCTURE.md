# Project Structure for Trace Utils

```
Open-dLLM/
│
├── veomni/
│   ├── utils/
│   │   ├── __init__.py                    # ✨ Updated: exports trace utilities
│   │   ├── trace_utils.py                 # 🆕 Core implementation
│   │   └── README_trace_utils.md          # 🆕 Quick reference guide
│   │
│   └── ... (other veomni modules)
│
├── tests/
│   ├── test_trace_utils.py                # 🆕 Comprehensive test suite
│   └── ... (other tests)
│
├── examples/
│   ├── trace_utils_examples.py            # 🆕 Practical examples with real models
│   └── ... (other examples)
│
├── docs/
│   ├── trace_utils_guide.md               # 🆕 Complete documentation
│   └── ... (other docs)
│
└── IMPLEMENTATION_SUMMARY.md              # 🆕 This summary document
```

## Module Organization

### veomni/utils/trace_utils.py (561 lines)

```
┌─────────────────────────────────────────────────┐
│            trace_utils.py                       │
├─────────────────────────────────────────────────┤
│                                                 │
│  Data Structures (Lines 1-167)                  │
│  ├── LayerTokenData                            │
│  ├── GenerationStep                            │
│  └── GenerationTrace                           │
│                                                 │
│  Hook Implementation (Lines 168-312)            │
│  ├── LayerTokenHook (class)                    │
│  │   ├── __init__                              │
│  │   ├── _hook_fn (capture logic)             │
│  │   ├── __enter__ (register hook)            │
│  │   ├── __exit__ (cleanup)                   │
│  │   ├── _get_layer_module (find layer)       │
│  │   └── get_data (retrieve captured)         │
│  │                                              │
│  └── layer_token_hook (convenience function)   │
│                                                 │
│  Generation Tracing (Lines 313-478)            │
│  └── decode (main function)                    │
│      ├── Encode prompt                         │
│      ├── Generation loop                       │
│      │   ├── Forward pass                      │
│      │   ├── Compute probabilities             │
│      │   ├── Sample/select token               │
│      │   └── Create GenerationStep             │
│      └── Return GenerationTrace                │
│                                                 │
│  Utilities (Lines 479-561)                     │
│  └── capture_layer_activations                 │
│      (context manager for multiple layers)     │
│                                                 │
└─────────────────────────────────────────────────┘
```

## API Surface

### Public Exports

```python
# From veomni.utils.trace_utils import:

# Data Structures
LayerTokenData       # Container for layer activations
GenerationStep       # Single generation step info  
GenerationTrace      # Complete generation trace

# Classes
LayerTokenHook       # Hook manager class

# Functions
layer_token_hook()           # Extract from one layer/token
decode()                     # Trace generation process
capture_layer_activations()  # Extract from multiple layers
```

### Usage Flow

```
User Code
    ↓
┌───────────────────────────────────────────┐
│  Import from veomni.utils.trace_utils     │
└───────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────┐
│  Option 1: Extract Layer Activation       │
│  ────────────────────────────────────     │
│  with LayerTokenHook(model, 5, -1) as h:  │
│      outputs = model(inputs)              │
│      data = h.get_data()                  │
│                                           │
│  → Returns: LayerTokenData                │
└───────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────┐
│  Option 2: Trace Generation               │
│  ────────────────────────────────────     │
│  trace = decode(                          │
│      model, prompt, tokenizer,            │
│      max_new_tokens=20                    │
│  )                                        │
│                                           │
│  → Returns: GenerationTrace               │
│     - steps: List[GenerationStep]         │
│     - generated_tokens: List[int]         │
│     - generated_text: str                 │
└───────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────┐
│  Option 3: Multiple Layers                │
│  ────────────────────────────────────     │
│  with capture_layer_activations(          │
│      model, [0, 5, 10]                    │
│  ) as captures:                           │
│      outputs = model(inputs)              │
│                                           │
│  → Returns: Dict[int, LayerTokenData]     │
└───────────────────────────────────────────┘
```

## Data Flow Diagrams

### 1. Layer Token Hook Flow

```
┌──────────────┐
│ User creates │
│ LayerTokenHook│
│ (model, 5, -1)│
└──────┬───────┘
       ↓
┌──────────────────────┐
│ Enter context:       │
│ - Find layer module  │
│ - Register hook      │
└──────┬───────────────┘
       ↓
┌──────────────────────┐
│ User runs forward:   │
│ outputs = model(...) │
└──────┬───────────────┘
       ↓
┌──────────────────────────────┐
│ Hook fires on layer output:  │
│ - Extract hidden_states      │
│ - Select token position      │
│ - Store in LayerTokenData    │
└──────┬───────────────────────┘
       ↓
┌──────────────────────┐
│ User gets data:      │
│ data = hook.get_data()│
└──────┬───────────────┘
       ↓
┌──────────────────────┐
│ Exit context:        │
│ - Remove hook        │
│ - Cleanup            │
└──────────────────────┘
```

### 2. Generation Trace Flow

```
┌──────────────────┐
│ Call decode()    │
│ with prompt      │
└────────┬─────────┘
         ↓
┌──────────────────────┐
│ Encode prompt        │
│ → token IDs          │
└────────┬─────────────┘
         ↓
┌────────────────────────────────┐
│ Generation Loop (each step):   │
│                                │
│  1. Forward pass               │
│     model(input_ids)           │
│     → logits                   │
│        ↓                       │
│  2. Get next token logits      │
│     logits[:, -1, :]           │
│     → next_token_logits        │
│        ↓                       │
│  3. Compute probabilities      │
│     softmax(logits/temp)       │
│     → probs                    │
│        ↓                       │
│  4. Get top-k                  │
│     topk(probs, k)             │
│     → top_k_probs, indices     │
│        ↓                       │
│  5. Sample/select              │
│     multinomial(probs)         │
│     or argmax(probs)           │
│     → selected_token           │
│        ↓                       │
│  6. Create GenerationStep      │
│     store all info             │
│        ↓                       │
│  7. Append to sequence         │
│     input_ids += token         │
│        ↓                       │
│  8. Check EOS                  │
│     break if done              │
│        ↓                       │
└────────┴─────────────────────┘
         ↓
┌──────────────────────┐
│ Decode tokens        │
│ → generated_text     │
└────────┬─────────────┘
         ↓
┌──────────────────────┐
│ Return               │
│ GenerationTrace      │
└──────────────────────┘
```

## File Dependencies

```
trace_utils.py
├── Imports:
│   ├── dataclasses (dataclass, field)
│   ├── typing (Optional, List, Dict, Callable, Any, Tuple)
│   ├── torch
│   ├── torch.nn
│   └── contextlib (contextmanager)
│
└── Used by:
    ├── tests/test_trace_utils.py
    ├── examples/trace_utils_examples.py
    └── User code
```

## Test Coverage

```
test_trace_utils.py (360+ lines)
├── Helper Classes
│   ├── SimpleTransformerLayer
│   ├── SimpleTransformerModel
│   └── SimpleTokenizer
│
├── Tests (7 total)
│   ├── test_layer_token_hook_basic
│   ├── test_layer_token_hook_multiple_tokens
│   ├── test_capture_layer_activations
│   ├── test_decode_basic
│   ├── test_decode_with_sampling
│   ├── test_layer_token_data_operations
│   └── test_generation_step_operations
│
└── run_all_tests()
    └── Pretty output with summary
```

## Example Coverage

```
trace_utils_examples.py (460+ lines)
├── Example 1: Extract layer activation
│   └── Shows basic LayerTokenHook usage
│
├── Example 2: Compare layers
│   └── Multi-layer extraction & similarity
│
├── Example 3: Trace generation
│   └── Full generation trace with analysis
│
└── Example 4: Sampling analysis
    └── Compare greedy vs sampling strategies
```

## Documentation Structure

```
docs/trace_utils_guide.md (650+ lines)
├── Overview
├── Quick Start
├── API Reference
│   ├── Data Structures
│   ├── Functions
│   └── Parameters
├── Usage Examples (5 detailed)
├── Advanced Usage
├── Supported Models
├── Performance Considerations
├── Testing
└── Troubleshooting
```

## Key Implementation Details

### Hook Registration

```python
# LayerTokenHook.__enter__
def __enter__(self):
    layer_module = self._get_layer_module()
    self.hook_handle = layer_module.register_forward_hook(self._hook_fn)
    return self
```

### Hook Function

```python
def _hook_fn(self, module, input, output):
    # Extract hidden states from output
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output
    
    # Select specific token
    token_idx = self.token_id if self.token_id >= 0 else hidden_states.shape[1] + self.token_id
    token_hidden = hidden_states[0, token_idx, :].detach().clone()
    
    # Store
    self.captured_data = LayerTokenData(...)
    
    return output
```

### Generation Loop Core

```python
for step_idx in range(max_new_tokens):
    # Forward
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :] / temperature
    
    # Probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sample
    next_token = torch.multinomial(probs, 1) if do_sample else torch.argmax(probs)
    
    # Record
    step = GenerationStep(...)
    trace.add_step(step)
    
    # Update
    input_ids = torch.cat([input_ids, next_token], dim=1)
```

## Integration Points

```
trace_utils can be used with:

┌─────────────────────────────────────┐
│ Training Scripts (tasks/*.py)       │
│ → Analyze activations during train │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Evaluation (eval/*.py)              │
│ → Detailed error analysis           │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Inference (tasks/infer.py)          │
│ → Track generation confidence       │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Research Notebooks                  │
│ → Probe model knowledge             │
└─────────────────────────────────────┘
```

## Summary Statistics

```
Files Created:     6
Lines of Code:     ~2,200
Documentation:     ~1,500 lines
Tests:             7 test cases
Examples:          4 practical examples
API Functions:     3 main functions
Data Classes:      3 structures
```

## Next Steps for Users

1. **Read**: `veomni/utils/README_trace_utils.md` (Quick start)
2. **Run**: `python tests/test_trace_utils.py` (Verify installation)
3. **Try**: `python examples/trace_utils_examples.py` (See real usage)
4. **Learn**: `docs/trace_utils_guide.md` (Deep dive)
5. **Use**: Import and use in your code!

```python
from veomni.utils import LayerTokenHook, decode

# Your code here!
```
