"""
Utilities for tracing model execution and extracting intermediate activations.

This module provides tools for:
1. Hooking into model layers to extract token activations
2. Tracing model generation to get probability distributions and token sequences
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
from contextlib import contextmanager


@dataclass
class LayerTokenData:
    """
    Data structure for storing activations from a specific layer and token position.
    
    Attributes:
        layer_id: The layer index in the model
        token_id: The token position index in the sequence
        hidden_states: The hidden state tensor at this layer and position
        attention_output: Optional attention output if available
        mlp_output: Optional MLP output if available
        layer_name: Optional name of the layer
        metadata: Additional metadata dictionary
    """
    layer_id: int
    token_id: int
    hidden_states: torch.Tensor
    attention_output: Optional[torch.Tensor] = None
    mlp_output: Optional[torch.Tensor] = None
    layer_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        self.hidden_states = self.hidden_states.to(device)
        if self.attention_output is not None:
            self.attention_output = self.attention_output.to(device)
        if self.mlp_output is not None:
            self.mlp_output = self.mlp_output.to(device)
        return self
    
    def cpu(self):
        """Move all tensors to CPU."""
        return self.to(torch.device('cpu'))
    
    def detach(self):
        """Detach all tensors from the computation graph."""
        self.hidden_states = self.hidden_states.detach()
        if self.attention_output is not None:
            self.attention_output = self.attention_output.detach()
        if self.mlp_output is not None:
            self.mlp_output = self.mlp_output.detach()
        return self


@dataclass
class GenerationStep:
    """
    Data structure for storing information about a single generation step.
    
    Attributes:
        step: The generation step index (0 for first predicted token)
        logits: The logits for all vocabulary tokens at this step
        probs: The probability distribution over vocabulary
        top_k_probs: Top-k probabilities
        top_k_indices: Indices of top-k tokens
        selected_token_id: The token that was actually selected
        selected_token_prob: The probability of the selected token
        temperature: Temperature used for sampling (if applicable)
        metadata: Additional metadata
    """
    step: int
    logits: torch.Tensor
    probs: torch.Tensor
    top_k_probs: Optional[torch.Tensor] = None
    top_k_indices: Optional[torch.Tensor] = None
    selected_token_id: int = -1
    selected_token_prob: float = 0.0
    temperature: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        self.logits = self.logits.to(device)
        self.probs = self.probs.to(device)
        if self.top_k_probs is not None:
            self.top_k_probs = self.top_k_probs.to(device)
        if self.top_k_indices is not None:
            self.top_k_indices = self.top_k_indices.to(device)
        return self
    
    def cpu(self):
        """Move all tensors to CPU."""
        return self.to(torch.device('cpu'))


@dataclass
class GenerationTrace:
    """
    Complete trace of a generation process.
    
    Attributes:
        prompt_text: The input prompt text
        prompt_tokens: Token IDs of the prompt
        steps: List of GenerationStep objects, one per generated token
        generated_tokens: List of generated token IDs
        generated_text: The decoded generated text
        metadata: Additional metadata
    """
    prompt_text: str
    prompt_tokens: List[int]
    steps: List[GenerationStep] = field(default_factory=list)
    generated_tokens: List[int] = field(default_factory=list)
    generated_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: GenerationStep):
        """Add a generation step to the trace."""
        self.steps.append(step)
        if step.selected_token_id >= 0:
            self.generated_tokens.append(step.selected_token_id)
    
    def get_token_probabilities(self) -> List[Tuple[int, float]]:
        """Get list of (token_id, probability) for all generated tokens."""
        return [(step.selected_token_id, step.selected_token_prob) 
                for step in self.steps if step.selected_token_id >= 0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'prompt_text': self.prompt_text,
            'prompt_tokens': self.prompt_tokens,
            'generated_tokens': self.generated_tokens,
            'generated_text': self.generated_text,
            'num_steps': len(self.steps),
            'metadata': self.metadata
        }


class LayerTokenHook:
    """
    Hook manager for extracting activations from a specific layer and token position.
    
    This uses PyTorch's forward hook mechanism to capture intermediate activations.
    """
    
    def __init__(self, model: nn.Module, layer_id: int, token_id: int):
        """
        Initialize the hook.
        
        Args:
            model: The model to hook into
            layer_id: Which layer to extract from (0-indexed)
            token_id: Which token position to extract (-1 for last token)
        """
        self.model = model
        self.layer_id = layer_id
        self.token_id = token_id
        self.captured_data: Optional[LayerTokenData] = None
        self.hook_handle = None
        
    def _hook_fn(self, module: nn.Module, input: Tuple, output: Any):
        """The actual hook function that captures data."""
        # Handle different output formats
        if isinstance(output, tuple):
            # Common format: (hidden_states, attention_weights, ...)
            hidden_states = output[0]
        elif isinstance(output, torch.Tensor):
            hidden_states = output
        else:
            # Try to get hidden_states attribute
            hidden_states = getattr(output, 'hidden_states', None)
            if hidden_states is None:
                hidden_states = getattr(output, 'last_hidden_state', None)
        
        if hidden_states is None:
            return output
        
        # Extract the specific token
        # Shape is typically [batch_size, seq_len, hidden_dim]
        if len(hidden_states.shape) >= 3:
            batch_idx = 0  # Assume first batch item
            token_idx = self.token_id if self.token_id >= 0 else hidden_states.shape[1] + self.token_id
            token_hidden = hidden_states[batch_idx, token_idx, :].detach().clone()
        else:
            token_hidden = hidden_states.detach().clone()
        
        # Store the captured data
        self.captured_data = LayerTokenData(
            layer_id=self.layer_id,
            token_id=self.token_id,
            hidden_states=token_hidden,
            layer_name=module.__class__.__name__
        )
        
        return output
    
    def __enter__(self):
        """Register the hook when entering context."""
        self.captured_data = None
        
        # Find the target layer
        layer_module = self._get_layer_module()
        if layer_module is None:
            raise ValueError(f"Could not find layer {self.layer_id} in model")
        
        # Register the hook
        self.hook_handle = layer_module.register_forward_hook(self._hook_fn)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove the hook when exiting context."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def _get_layer_module(self) -> Optional[nn.Module]:
        """
        Find the module corresponding to the layer_id.
        
        This tries common naming patterns for transformer models.
        """
        # Try common patterns
        patterns = [
            f'model.layers.{self.layer_id}',  # Common for many models
            f'transformer.h.{self.layer_id}',  # GPT-style
            f'model.decoder.layers.{self.layer_id}',  # Decoder models
            f'encoder.layer.{self.layer_id}',  # BERT-style
            f'layers.{self.layer_id}',  # Direct layers attribute
        ]
        
        for pattern in patterns:
            try:
                module = self.model
                for attr in pattern.split('.'):
                    module = getattr(module, attr)
                return module
            except AttributeError:
                continue
        
        # If patterns fail, try to find layers by iterating
        for name, module in self.model.named_modules():
            if 'layer' in name.lower() and str(self.layer_id) in name:
                return module
        
        return None
    
    def get_data(self) -> Optional[LayerTokenData]:
        """Get the captured data."""
        return self.captured_data


def layer_token_hook(model: nn.Module, layer_id: int, token_id: int) -> Optional[LayerTokenData]:
    """
    Extract hidden states from a specific layer and token position.
    
    This function should be used within a model forward pass context.
    
    Args:
        model: The model to extract from
        layer_id: Which layer to extract from (0-indexed)
        token_id: Which token position to extract (-1 for last token)
    
    Returns:
        LayerTokenData containing the extracted activations, or None if extraction failed
    
    Example:
        >>> with LayerTokenHook(model, layer_id=5, token_id=-1) as hook:
        ...     outputs = model(**inputs)
        ...     data = hook.get_data()
    """
    hook = LayerTokenHook(model, layer_id, token_id)
    return hook


@torch.no_grad()
def decode(
    model: nn.Module,
    prompt: str,
    tokenizer: Any,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    do_sample: bool = True,
) -> GenerationTrace:
    """
    Generate text while tracing probability distributions at each step.
    
    Args:
        model: The language model
        prompt: Input text prompt
        tokenizer: Tokenizer for encoding/decoding
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        do_sample: Whether to sample or use greedy decoding
    
    Returns:
        GenerationTrace containing detailed information about each generation step
    
    Example:
        >>> trace = decode(model, "Hello world", tokenizer, max_new_tokens=10)
        >>> for step in trace.steps:
        ...     print(f"Step {step.step}: token={step.selected_token_id}, prob={step.selected_token_prob:.4f}")
    """
    # Encode the prompt
    if hasattr(tokenizer, 'encode'):
        prompt_tokens = tokenizer.encode(prompt, return_tensors='pt')
    else:
        prompt_tokens = tokenizer(prompt, return_tensors='pt').input_ids
    
    device = next(model.parameters()).device
    input_ids = prompt_tokens.to(device)
    
    # Initialize trace
    trace = GenerationTrace(
        prompt_text=prompt,
        prompt_tokens=input_ids[0].cpu().tolist(),
        metadata={
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'do_sample': do_sample,
            'max_new_tokens': max_new_tokens,
        }
    )
    
    # Generation loop
    for step_idx in range(max_new_tokens):
        # Forward pass
        outputs = model(input_ids)
        
        # Get logits for the last token
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        next_token_logits = logits[:, -1, :] / temperature
        
        # Compute probabilities
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        
        # Get top-k
        if top_k is not None and top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
        else:
            top_k_probs, top_k_indices = None, None
        
        # Sample or select greedily
        if do_sample:
            # Apply top-p filtering if specified
            if top_p is not None and top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create mask and zero out filtered probabilities
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs_filtered = probs.clone()
                probs_filtered[indices_to_remove] = 0.0
                probs_filtered = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs_filtered, num_samples=1)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        
        next_token_id = next_token.item()
        next_token_prob = probs[0, next_token_id].item()
        
        # Create generation step
        gen_step = GenerationStep(
            step=step_idx,
            logits=next_token_logits[0].cpu(),
            probs=probs[0].cpu(),
            top_k_probs=top_k_probs[0].cpu() if top_k_probs is not None else None,
            top_k_indices=top_k_indices[0].cpu() if top_k_indices is not None else None,
            selected_token_id=next_token_id,
            selected_token_prob=next_token_prob,
            temperature=temperature,
        )
        
        trace.add_step(gen_step)
        
        # Check for EOS token
        eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        
        # Append to input for next iteration
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    # Decode the generated text
    if hasattr(tokenizer, 'decode'):
        trace.generated_text = tokenizer.decode(trace.generated_tokens, skip_special_tokens=True)
    else:
        trace.generated_text = tokenizer.batch_decode([trace.generated_tokens], skip_special_tokens=True)[0]
    
    return trace


@contextmanager
def capture_layer_activations(
    model: nn.Module,
    layer_ids: List[int],
    token_id: int = -1
):
    """
    Context manager to capture activations from multiple layers.
    
    Args:
        model: The model to hook
        layer_ids: List of layer indices to capture
        token_id: Token position to capture (-1 for last)
    
    Yields:
        Dict mapping layer_id to LayerTokenData
    
    Example:
        >>> with capture_layer_activations(model, [0, 5, 10]) as captures:
        ...     outputs = model(**inputs)
        >>> for layer_id, data in captures.items():
        ...     print(f"Layer {layer_id}: {data.hidden_states.shape}")
    """
    hooks = {layer_id: LayerTokenHook(model, layer_id, token_id) 
             for layer_id in layer_ids}
    captures = {}
    
    try:
        # Enter all hooks
        for layer_id, hook in hooks.items():
            hook.__enter__()
        
        yield captures
        
        # Collect captured data
        for layer_id, hook in hooks.items():
            data = hook.get_data()
            if data is not None:
                captures[layer_id] = data
    finally:
        # Exit all hooks
        for hook in hooks.values():
            hook.__exit__(None, None, None)
