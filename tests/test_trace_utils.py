"""
Tests and examples for trace_utils module.

This file demonstrates how to use the layer_token_hook and decode functions
to extract intermediate activations and trace generation.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from typing import Optional

# Import the utilities we created
from veomni.utils.trace_utils import (
    LayerTokenData,
    GenerationTrace,
    GenerationStep,
    LayerTokenHook,
    layer_token_hook,
    decode,
    capture_layer_activations,
)


class SimpleTransformerLayer(nn.Module):
    """A simple transformer layer for testing."""
    
    def __init__(self, hidden_size=768):
        super().__init__()
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states):
        # Simple transformer layer
        attn_output = self.attention(hidden_states)
        hidden_states = self.layer_norm(hidden_states + attn_output)
        mlp_output = self.mlp(hidden_states)
        hidden_states = self.layer_norm(hidden_states + mlp_output)
        return hidden_states


class SimpleTransformerModel(nn.Module):
    """A simple transformer model for testing."""
    
    def __init__(self, vocab_size=1000, hidden_size=768, num_layers=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(hidden_size) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        logits = self.lm_head(hidden_states)
        
        # Return in HuggingFace-style format
        class OutputWithLogits:
            def __init__(self, logits):
                self.logits = logits
        
        return OutputWithLogits(logits)


class SimpleTokenizer:
    """A simple tokenizer for testing."""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
    
    def encode(self, text, return_tensors=None):
        # Simple encoding: convert characters to token IDs
        tokens = [ord(c) % self.vocab_size for c in text]
        if return_tensors == 'pt':
            return torch.tensor([tokens])
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=False):
        # Simple decoding: convert token IDs back to characters
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t != self.eos_token_id]
        return ''.join([chr(t % 128) for t in token_ids if t < 128])
    
    def __call__(self, text, return_tensors=None):
        class TokenizerOutput:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        
        tokens = self.encode(text, return_tensors=return_tensors)
        return TokenizerOutput(tokens)


def test_layer_token_hook_basic():
    """Test basic functionality of LayerTokenHook."""
    print("\n" + "="*80)
    print("TEST: Basic LayerTokenHook")
    print("="*80)
    
    # Create a simple model
    model = SimpleTransformerModel(vocab_size=1000, hidden_size=256, num_layers=6)
    model.eval()
    
    # Create input
    input_ids = torch.randint(0, 1000, (1, 10))  # batch_size=1, seq_len=10
    
    # Test hooking layer 3, last token
    layer_id = 3
    token_id = -1
    
    print(f"\nHooking layer {layer_id}, token position {token_id}")
    
    with LayerTokenHook(model, layer_id, token_id) as hook:
        # Run forward pass
        outputs = model(input_ids)
        
        # Get the captured data
        data = hook.get_data()
    
    if data is not None:
        print(f"✓ Successfully captured data from layer {data.layer_id}")
        print(f"  Hidden states shape: {data.hidden_states.shape}")
        print(f"  Layer name: {data.layer_name}")
        print(f"  Token position: {data.token_id}")
    else:
        print("✗ Failed to capture data")
    
    return data is not None


def test_layer_token_hook_multiple_tokens():
    """Test capturing multiple token positions."""
    print("\n" + "="*80)
    print("TEST: Multiple Token Positions")
    print("="*80)
    
    model = SimpleTransformerModel(vocab_size=1000, hidden_size=256, num_layers=6)
    model.eval()
    
    input_ids = torch.randint(0, 1000, (1, 10))
    layer_id = 2
    
    # Capture different token positions
    token_positions = [0, 4, -1]
    captured_data = []
    
    for token_id in token_positions:
        with LayerTokenHook(model, layer_id, token_id) as hook:
            outputs = model(input_ids)
            data = hook.get_data()
            if data is not None:
                captured_data.append(data)
    
    print(f"\nCaptured {len(captured_data)} token positions:")
    for data in captured_data:
        print(f"  Token {data.token_id}: shape {data.hidden_states.shape}")
    
    return len(captured_data) == len(token_positions)


def test_capture_layer_activations():
    """Test capturing activations from multiple layers at once."""
    print("\n" + "="*80)
    print("TEST: Capture Multiple Layers")
    print("="*80)
    
    model = SimpleTransformerModel(vocab_size=1000, hidden_size=256, num_layers=6)
    model.eval()
    
    input_ids = torch.randint(0, 1000, (1, 10))
    
    # Capture layers 0, 2, 4
    layer_ids = [0, 2, 4]
    
    with capture_layer_activations(model, layer_ids, token_id=-1) as captures:
        outputs = model(input_ids)
    
    print(f"\nCaptured {len(captures)} layers:")
    for layer_id, data in captures.items():
        print(f"  Layer {layer_id}: shape {data.hidden_states.shape}")
    
    return len(captures) == len(layer_ids)


def test_decode_basic():
    """Test basic decode functionality."""
    print("\n" + "="*80)
    print("TEST: Basic Decode with Tracing")
    print("="*80)
    
    model = SimpleTransformerModel(vocab_size=1000, hidden_size=256, num_layers=4)
    model.eval()
    
    tokenizer = SimpleTokenizer(vocab_size=1000)
    
    prompt = "Hello"
    max_new_tokens = 5
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Generating {max_new_tokens} tokens...")
    
    # Trace the generation
    trace = decode(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=50,
        do_sample=False,  # Use greedy for deterministic results
    )
    
    print(f"\n✓ Generation complete!")
    print(f"  Prompt tokens: {trace.prompt_tokens}")
    print(f"  Generated tokens: {trace.generated_tokens}")
    print(f"  Number of steps: {len(trace.steps)}")
    
    # Show details of each step
    print(f"\nGeneration steps:")
    for step in trace.steps[:3]:  # Show first 3 steps
        print(f"  Step {step.step}:")
        print(f"    Selected token: {step.selected_token_id}")
        print(f"    Token probability: {step.selected_token_prob:.4f}")
        print(f"    Logits shape: {step.logits.shape}")
        print(f"    Probs shape: {step.probs.shape}")
        if step.top_k_indices is not None:
            print(f"    Top-k tokens: {step.top_k_indices[:5].tolist()}")
    
    return len(trace.steps) > 0


def test_decode_with_sampling():
    """Test decode with sampling."""
    print("\n" + "="*80)
    print("TEST: Decode with Sampling")
    print("="*80)
    
    model = SimpleTransformerModel(vocab_size=1000, hidden_size=256, num_layers=4)
    model.eval()
    
    tokenizer = SimpleTokenizer(vocab_size=1000)
    
    prompt = "Test"
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Using temperature=1.0, top_k=50, top_p=0.95")
    
    # Trace with sampling
    trace = decode(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=10,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        do_sample=True,
    )
    
    print(f"\n✓ Sampled generation complete!")
    print(f"  Generated {len(trace.generated_tokens)} tokens")
    
    # Show token probabilities
    token_probs = trace.get_token_probabilities()
    print(f"\nToken probabilities:")
    for i, (token_id, prob) in enumerate(token_probs[:5]):
        print(f"  Position {i}: token={token_id}, prob={prob:.4f}")
    
    # Test trace serialization
    trace_dict = trace.to_dict()
    print(f"\nTrace dictionary keys: {list(trace_dict.keys())}")
    
    return len(trace.steps) > 0


def test_layer_token_data_operations():
    """Test LayerTokenData operations."""
    print("\n" + "="*80)
    print("TEST: LayerTokenData Operations")
    print("="*80)
    
    # Create sample data
    hidden_states = torch.randn(768)
    attention_output = torch.randn(768)
    
    data = LayerTokenData(
        layer_id=5,
        token_id=10,
        hidden_states=hidden_states,
        attention_output=attention_output,
        metadata={'test': 'value'}
    )
    
    print(f"\nOriginal device: {data.hidden_states.device}")
    
    # Test detach
    data_detached = data.detach()
    print(f"✓ Detached successfully")
    
    # Test CPU move
    data_cpu = data.cpu()
    print(f"✓ Moved to CPU: {data_cpu.hidden_states.device}")
    
    # Test metadata
    print(f"✓ Metadata: {data.metadata}")
    
    return True


def test_generation_step_operations():
    """Test GenerationStep operations."""
    print("\n" + "="*80)
    print("TEST: GenerationStep Operations")
    print("="*80)
    
    # Create sample step
    logits = torch.randn(1000)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, 50)
    
    step = GenerationStep(
        step=0,
        logits=logits,
        probs=probs,
        top_k_probs=top_k_probs,
        top_k_indices=top_k_indices,
        selected_token_id=42,
        selected_token_prob=0.123,
        temperature=0.8,
    )
    
    print(f"\nStep {step.step}:")
    print(f"  Selected token: {step.selected_token_id}")
    print(f"  Probability: {step.selected_token_prob}")
    print(f"  Temperature: {step.temperature}")
    print(f"  Top-5 token indices: {step.top_k_indices[:5].tolist()}")
    print(f"  Top-5 probabilities: {step.top_k_probs[:5].tolist()}")
    
    # Test device movement
    step_cpu = step.cpu()
    print(f"✓ Moved to CPU")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "TRACE UTILS TEST SUITE" + " "*36 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("Layer Token Hook - Basic", test_layer_token_hook_basic),
        ("Layer Token Hook - Multiple Tokens", test_layer_token_hook_multiple_tokens),
        ("Capture Multiple Layers", test_capture_layer_activations),
        ("Decode - Basic", test_decode_basic),
        ("Decode - Sampling", test_decode_with_sampling),
        ("LayerTokenData Operations", test_layer_token_data_operations),
        ("GenerationStep Operations", test_generation_step_operations),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
    
    # Print summary
    print("\n\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*30 + "TEST SUMMARY" + " "*36 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    passed = 0
    failed = 0
    
    for test_name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} {test_name}")
        if error:
            print(f"           Error: {error}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print()
    print(f"Total: {len(results)} tests, {passed} passed, {failed} failed")
    print()
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
