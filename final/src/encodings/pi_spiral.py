"""
π-Spiral Positional Encoding Implementation

This module implements the π-Spiral positional encoding scheme with O(1) global attractor.
Based on the formula: e_n = [cos(2π * frac(n*π)), sin(2π * frac(n*π))]

Key features:
- Non-periodic positional encoding using irrational numbers
- O(1) memory complexity via attractor state
- Incremental state updates
- Support for multiple irrational constants (π, e, √2, φ)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Literal


class PiSpiralPositional(nn.Module):
    """
    π-Spiral Positional Encoding Module
    
    Generates 2D unit vectors on a spiral trajectory using irrational number properties
    to avoid periodic aliasing in long sequences.
    
    Args:
        d_model: Model dimension (must be even for 2D spiral encoding)
        max_len: Maximum sequence length to pre-compute (default: 100000)
        irrational: Choice of irrational constant ('pi', 'e', 'sqrt2', 'phi', 'prng')
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 100000,
        irrational: Literal['pi', 'e', 'sqrt2', 'phi', 'prng'] = 'pi',
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for 2D spiral encoding, got {d_model}")
        
        self.d_model = d_model
        self.max_len = max_len
        self.irrational = irrational
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        
        # Select irrational constant
        self.constant = self._get_irrational_constant(irrational)
        
        # Pre-compute positional encodings for efficiency
        # Shape: (max_len, d_model)
        self.register_buffer('pe', self._compute_positional_encodings(max_len))
    
    def _get_irrational_constant(self, irrational: str) -> float:
        """Get the irrational constant value"""
        constants = {
            'pi': math.pi,
            'e': math.e,
            'sqrt2': math.sqrt(2),
            'phi': (1 + math.sqrt(5)) / 2,  # Golden ratio
            'prng': None,  # Pseudo-random, handled separately
        }
        
        if irrational not in constants:
            raise ValueError(f"Unknown irrational type: {irrational}")
        
        return constants[irrational]
    
    def _compute_positional_encodings(self, length: int) -> torch.Tensor:
        """
        Compute π-Spiral positional encodings for positions 0 to length-1
        
        Formula: e_n = [cos(2π * frac(n * constant)), sin(2π * frac(n * constant))]
        where frac(x) returns the fractional part of x
        
        Returns:
            Tensor of shape (length, d_model)
        """
        # Position indices: 0, 1, 2, ..., length-1
        positions = torch.arange(length, dtype=self.dtype, device=self.device)
        
        # Initialize encoding tensor
        pe = torch.zeros(length, self.d_model, dtype=self.dtype, device=self.device)
        
        if self.irrational == 'prng':
            # Use pseudo-random generator with fixed seed for reproducibility
            torch.manual_seed(42)
            # Generate random phases for each position
            phases = torch.rand(length, dtype=self.dtype, device=self.device) * 2 * math.pi
        else:
            # Compute phases using irrational constant
            # phase = 2π * frac(n * constant)
            # frac(x) = x - floor(x)
            n_times_constant = positions * self.constant
            fractional_part = n_times_constant - torch.floor(n_times_constant)
            phases = 2 * math.pi * fractional_part
        
        # Apply spiral encoding to all dimension pairs
        # Each pair of dimensions gets the same phase (can be extended to different frequencies)
        for i in range(0, self.d_model, 2):
            pe[:, i] = torch.cos(phases)
            pe[:, i + 1] = torch.sin(phases)
        
        return pe
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add π-Spiral positional encoding to input tensor
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            positions: Optional position indices of shape (batch_size, seq_len)
                      If None, uses sequential positions 0, 1, 2, ...
        
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        batch_size, seq_len, d_model = x.shape
        
        if d_model != self.d_model:
            raise ValueError(f"Input d_model {d_model} doesn't match encoding d_model {self.d_model}")
        
        if positions is None:
            # Use sequential positions
            if seq_len > self.max_len:
                # Compute on-the-fly for sequences longer than pre-computed
                positions = torch.arange(seq_len, device=x.device)
                pe = self._compute_positional_encodings_for_positions(positions)
            else:
                pe = self.pe[:seq_len]
        else:
            # Use provided positions
            pe = self._compute_positional_encodings_for_positions(positions.flatten())
            pe = pe.view(batch_size, seq_len, d_model)
        
        return x + pe
    
    def _compute_positional_encodings_for_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute positional encodings for arbitrary position indices
        
        Args:
            positions: Position indices tensor of any shape
        
        Returns:
            Positional encodings of shape (*positions.shape, d_model)
        """
        original_shape = positions.shape
        positions = positions.flatten()
        length = positions.shape[0]
        
        pe = torch.zeros(length, self.d_model, dtype=self.dtype, device=positions.device)
        
        if self.irrational == 'prng':
            # For PRNG, use position as seed modifier
            phases = torch.zeros(length, dtype=self.dtype, device=positions.device)
            for idx, pos in enumerate(positions):
                torch.manual_seed(42 + int(pos.item()))
                phases[idx] = torch.rand(1).item() * 2 * math.pi
        else:
            n_times_constant = positions.float() * self.constant
            fractional_part = n_times_constant - torch.floor(n_times_constant)
            phases = 2 * math.pi * fractional_part
        
        for i in range(0, self.d_model, 2):
            pe[:, i] = torch.cos(phases)
            pe[:, i + 1] = torch.sin(phases)
        
        return pe.view(*original_shape, self.d_model)
    
    def get_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get positional encodings for specific positions without adding to input
        
        Args:
            positions: Position indices of shape (batch_size, seq_len) or (seq_len,)
        
        Returns:
            Positional encodings of shape (*positions.shape, d_model)
        """
        return self._compute_positional_encodings_for_positions(positions)


class AttractorState(nn.Module):
    """
    O(1) Attractor State for Global Context Compression
    
    Maintains a fixed-size state that incrementally accumulates information
    from all past tokens using exponential decay.
    
    Formula: C_t = α * C_{t-1} + e_t ⊗ h_t
    where:
        - C_t: attractor state at time t
        - α: decay factor
        - e_t: positional encoding at time t
        - h_t: hidden state at time t
        - ⊗: outer product
    
    Args:
        d_model: Model dimension
        d_state: Dimension of attractor state (default: d_model)
        alpha_policy: Policy for computing decay factor α
                     ('fixed', 'exp_c_over_N', 'learned')
        alpha_value: Fixed alpha value (used when alpha_policy='fixed')
        c_value: Constant c for exp(-c/N) policy (default: π)
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: Optional[int] = None,
        alpha_policy: Literal['fixed', 'exp_c_over_N', 'learned'] = 'exp_c_over_N',
        alpha_value: float = 0.99,
        c_value: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state or d_model
        self.alpha_policy = alpha_policy
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        
        # Initialize alpha based on policy
        if alpha_policy == 'fixed':
            self.register_buffer('alpha', torch.tensor(alpha_value, dtype=dtype, device=device))
        elif alpha_policy == 'exp_c_over_N':
            # α = exp(-c/N), where c defaults to π
            self.c_value = c_value if c_value is not None else math.pi
            # N will be computed dynamically based on sequence length
            self.alpha = None  # Computed per forward pass
        elif alpha_policy == 'learned':
            # Learnable alpha parameter (constrained to [0, 1])
            self.alpha_raw = nn.Parameter(torch.tensor(0.0, dtype=dtype, device=device))
        else:
            raise ValueError(f"Unknown alpha_policy: {alpha_policy}")
        
        # Projection layers for state compression (optional)
        self.state_projection = nn.Linear(d_model, self.d_state, device=device, dtype=dtype)
        
        # Initialize attractor state
        self.register_buffer(
            'C',
            torch.zeros(1, self.d_state, d_model, dtype=dtype, device=device)
        )
    
    def _compute_alpha(self, seq_len: int) -> torch.Tensor:
        """Compute alpha based on policy"""
        if self.alpha_policy == 'fixed':
            return self.alpha
        elif self.alpha_policy == 'exp_c_over_N':
            alpha = math.exp(-self.c_value / seq_len)
            return torch.tensor(alpha, dtype=self.dtype, device=self.device)
        elif self.alpha_policy == 'learned':
            # Apply sigmoid to constrain to (0, 1)
            return torch.sigmoid(self.alpha_raw)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positional_encodings: torch.Tensor,
        reset_state: bool = False
    ) -> torch.Tensor:
        """
        Update attractor state incrementally
        
        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, d_model)
            positional_encodings: Positional encodings of shape (batch_size, seq_len, d_model)
            reset_state: Whether to reset the attractor state before processing
        
        Returns:
            Updated attractor state of shape (batch_size, d_state, d_model)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        if reset_state:
            self.C = torch.zeros(
                batch_size, self.d_state, d_model,
                dtype=self.dtype, device=self.device
            )
        
        # Compute alpha for this sequence
        alpha = self._compute_alpha(seq_len)
        
        # Expand state to batch size if needed
        if self.C.shape[0] != batch_size:
            self.C = self.C.expand(batch_size, -1, -1).contiguous()
        
        # Incremental update for each position in sequence
        C_current = self.C.clone()
        
        for t in range(seq_len):
            # Get current hidden state and positional encoding
            h_t = hidden_states[:, t, :]  # (batch_size, d_model)
            e_t = positional_encodings[:, t, :]  # (batch_size, d_model)
            
            # Project hidden state to state dimension
            h_t_proj = self.state_projection(h_t)  # (batch_size, d_state)
            
            # Outer product: e_t ⊗ h_t
            # Result shape: (batch_size, d_state, d_model)
            outer_product = torch.bmm(
                h_t_proj.unsqueeze(2),  # (batch_size, d_state, 1)
                e_t.unsqueeze(1)  # (batch_size, 1, d_model)
            )
            
            # Update: C_t = α * C_{t-1} + e_t ⊗ h_t
            C_current = alpha * C_current + outer_product
        
        # Store updated state
        self.C = C_current.detach()
        
        return C_current
    
    def get_state(self) -> torch.Tensor:
        """Get current attractor state"""
        return self.C
    
    def reset(self):
        """Reset attractor state to zeros"""
        self.C.zero_()
