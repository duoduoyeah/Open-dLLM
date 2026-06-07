"""
Practical example of using trace_utils with real models.

This example shows how to:
1. Extract layer activations during inference
2. Trace generation with probability tracking
3. Analyze model behavior
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from veomni.utils.trace_utils import (
    LayerTokenHook,
    decode,
    capture_layer_activations,
)


def example_1_extract_layer_activation():
    """
    Example 1: Extract activation from a specific layer and token.
    
    Use case: Understanding what representations the model builds
    at different layers for a specific token.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Extract Layer Activation")
    print("="*80)
    
    # Load a small model (you can replace with your model)
    model_name = "gpt2"  # or your model path
    print(f"\nLoading model: {model_name}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        # Prepare input
        text = "The quick brown fox"
        inputs = tokenizer(text, return_tensors="pt")
        
        # Extract from layer 6, last token position
        layer_id = 6
        token_id = -1
        
        print(f"\nExtracting activation from:")
        print(f"  Text: '{text}'")
        print(f"  Layer: {layer_id}")
        print(f"  Token position: {token_id} (last token)")
        
        with LayerTokenHook(model, layer_id, token_id) as hook:
            with torch.no_grad():
                outputs = model(**inputs)
            
            data = hook.get_data()
        
        if data is not None:
            print(f"\n✓ Successfully extracted activation!")
            print(f"  Shape: {data.hidden_states.shape}")
            print(f"  Mean: {data.hidden_states.mean().item():.4f}")
            print(f"  Std: {data.hidden_states.std().item():.4f}")
            print(f"  Layer name: {data.layer_name}")
            
            # You can now use this activation for analysis
            # For example, compute similarity with other activations
            activation_norm = torch.norm(data.hidden_states).item()
            print(f"  L2 norm: {activation_norm:.4f}")
            
        else:
            print("✗ Failed to extract activation")
            
    except Exception as e:
        print(f"Error: {e}")
        print("This example requires the 'transformers' library.")
        print("Install with: pip install transformers")


def example_2_compare_layers():
    """
    Example 2: Compare activations across multiple layers.
    
    Use case: Analyzing how representations evolve through the network.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Compare Activations Across Layers")
    print("="*80)
    
    model_name = "gpt2"
    print(f"\nLoading model: {model_name}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        text = "Machine learning is"
        inputs = tokenizer(text, return_tensors="pt")
        
        # Extract from multiple layers
        layer_ids = [0, 3, 6, 9, 11]  # GPT-2 has 12 layers (0-11)
        
        print(f"\nExtracting from layers: {layer_ids}")
        print(f"Text: '{text}'")
        
        with capture_layer_activations(model, layer_ids, token_id=-1) as captures:
            with torch.no_grad():
                outputs = model(**inputs)
        
        print(f"\n✓ Captured {len(captures)} layers")
        print("\nActivation statistics by layer:")
        print(f"{'Layer':<8} {'Mean':<10} {'Std':<10} {'L2 Norm':<10}")
        print("-" * 40)
        
        for layer_id in sorted(captures.keys()):
            data = captures[layer_id]
            mean = data.hidden_states.mean().item()
            std = data.hidden_states.std().item()
            norm = torch.norm(data.hidden_states).item()
            print(f"{layer_id:<8} {mean:<10.4f} {std:<10.4f} {norm:<10.4f}")
        
        # Compute similarity between layers
        print("\nCosine similarity between consecutive layers:")
        layer_list = sorted(captures.keys())
        for i in range(len(layer_list) - 1):
            layer_a = layer_list[i]
            layer_b = layer_list[i + 1]
            
            act_a = captures[layer_a].hidden_states
            act_b = captures[layer_b].hidden_states
            
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                act_a.unsqueeze(0), 
                act_b.unsqueeze(0)
            ).item()
            
            print(f"  Layer {layer_a} -> {layer_b}: {similarity:.4f}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("This example requires the 'transformers' library.")


def example_3_trace_generation():
    """
    Example 3: Trace generation with probability tracking.
    
    Use case: Understanding model's confidence and decision-making during generation.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Trace Generation with Probabilities")
    print("="*80)
    
    model_name = "gpt2"
    print(f"\nLoading model: {model_name}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        
        prompt = "The future of artificial intelligence"
        max_new_tokens = 20
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Generating {max_new_tokens} tokens with greedy decoding...")
        
        # Trace the generation
        trace = decode(
            model=model,
            prompt=prompt,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=False,  # Greedy for reproducibility
        )
        
        print(f"\n✓ Generation complete!")
        print(f"\nGenerated text:")
        print(f"  {trace.generated_text}")
        
        print(f"\nDetailed generation trace:")
        print(f"{'Step':<6} {'Token':<10} {'Token ID':<10} {'Probability':<12} {'Top-5 Tokens'}")
        print("-" * 80)
        
        for i, step in enumerate(trace.steps[:10]):  # Show first 10 steps
            token_text = tokenizer.decode([step.selected_token_id])
            top5_tokens = []
            if step.top_k_indices is not None:
                for idx in step.top_k_indices[:5]:
                    top5_tokens.append(tokenizer.decode([idx.item()]))
            
            print(f"{step.step:<6} {token_text!r:<10} {step.selected_token_id:<10} "
                  f"{step.selected_token_prob:<12.6f} {top5_tokens}")
        
        # Analyze confidence
        token_probs = trace.get_token_probabilities()
        avg_prob = sum(p for _, p in token_probs) / len(token_probs)
        min_prob = min(p for _, p in token_probs)
        max_prob = max(p for _, p in token_probs)
        
        print(f"\nConfidence statistics:")
        print(f"  Average probability: {avg_prob:.4f}")
        print(f"  Min probability: {min_prob:.4f}")
        print(f"  Max probability: {max_prob:.4f}")
        
        # Find low-confidence predictions
        low_conf_threshold = 0.1
        low_conf_steps = [(i, p) for i, (_, p) in enumerate(token_probs) if p < low_conf_threshold]
        
        if low_conf_steps:
            print(f"\nLow-confidence predictions (prob < {low_conf_threshold}):")
            for step_idx, prob in low_conf_steps:
                token_id = trace.steps[step_idx].selected_token_id
                token_text = tokenizer.decode([token_id])
                print(f"  Step {step_idx}: '{token_text}' (prob={prob:.4f})")
        else:
            print(f"\nNo low-confidence predictions found.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("This example requires the 'transformers' library.")


def example_4_sampling_analysis():
    """
    Example 4: Compare greedy vs sampling generation.
    
    Use case: Understanding how sampling parameters affect generation diversity.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Sampling Analysis")
    print("="*80)
    
    model_name = "gpt2"
    print(f"\nLoading model: {model_name}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        
        prompt = "Once upon a time"
        max_new_tokens = 15
        
        print(f"\nPrompt: '{prompt}'")
        
        # Generate with different settings
        configs = [
            ("Greedy", {"temperature": 1.0, "do_sample": False}),
            ("Low temp", {"temperature": 0.5, "do_sample": True, "top_k": 50}),
            ("High temp", {"temperature": 1.5, "do_sample": True, "top_k": 50}),
        ]
        
        results = []
        for name, config in configs:
            print(f"\n{name} decoding...")
            trace = decode(
                model=model,
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                **config
            )
            results.append((name, trace))
        
        # Compare results
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        
        for name, trace in results:
            avg_prob = sum(step.selected_token_prob for step in trace.steps) / len(trace.steps)
            print(f"\n{name}:")
            print(f"  Text: {trace.generated_text}")
            print(f"  Avg probability: {avg_prob:.4f}")
            print(f"  Unique tokens: {len(set(trace.generated_tokens))}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("This example requires the 'transformers' library.")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "TRACE UTILS EXAMPLES" + " "*38 + "║")
    print("╚" + "="*78 + "╝")
    
    examples = [
        example_1_extract_layer_activation,
        example_2_compare_layers,
        example_3_trace_generation,
        example_4_sampling_analysis,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\nExample failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n")
    print("="*80)
    print("Examples completed!")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
