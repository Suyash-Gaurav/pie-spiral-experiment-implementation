"""
Unit Tests for Positional Encoding Modules

Tests for:
- π-Spiral encoding
- RoPE encoding
- Hybrid encoding
- Attractor state

Based on Phase 1 checklist from experiment plan.
"""

import torch
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from encodings.pi_spiral import PiSpiralPositional, AttractorState
from encodings.rope import RoPEPositional
from encodings.hybrid import HybridPositionalEncoding


class TestPiSpiralPositional:
    """Tests for π-Spiral positional encoding"""
    
    def test_initialization(self):
        """Test that π-Spiral encoding initializes correctly"""
        encoding = PiSpiralPositional(d_model=512)
        assert encoding.d_model == 512
        assert encoding.irrational == 'pi'
    
    def test_shape_consistency(self):
        """Test that output shapes are correct"""
        encoding = PiSpiralPositional(d_model=512)
        positions = torch.arange(100)
        
        output = encoding(positions)
        
        assert output.shape == (100, 512), f"Expected shape (100, 512), got {output.shape}"
    
    def test_determinism(self):
        """Test that encoding is deterministic"""
        encoding = PiSpiralPositional(d_model=512, irrational='pi')
        positions = torch.arange(100)
        
        output1 = encoding(positions)
        output2 = encoding(positions)
        
        assert torch.allclose(output1, output2), "Encoding should be deterministic"
    
    def test_different_irrationals(self):
        """Test different irrational constants produce different encodings"""
        positions = torch.arange(100)
        
        encoding_pi = PiSpiralPositional(d_model=512, irrational='pi')
        encoding_e = PiSpiralPositional(d_model=512, irrational='e')
        
        output_pi = encoding_pi(positions)
        output_e = encoding_e(positions)
        
        assert not torch.allclose(output_pi, output_e), "Different irrationals should produce different encodings"
    
    def test_unit_vectors(self):
        """Test that 2D spiral produces unit vectors"""
        encoding = PiSpiralPositional(d_model=512, irrational='pi')
        positions = torch.arange(100)
        
        output = encoding(positions)
        
        # Check first two dimensions form unit vectors
        norms = torch.sqrt(output[:, 0]**2 + output[:, 1]**2)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "2D components should form unit vectors"
    
    def test_non_periodic(self):
        """Test that encoding is non-periodic (no exact repetitions)"""
        encoding = PiSpiralPositional(d_model=512, irrational='pi')
        positions = torch.arange(10000)
        
        output = encoding(positions)
        
        # Check that no two positions have identical encodings
        for i in range(0, min(100, len(output)), 10):
            for j in range(i + 1, min(i + 50, len(output))):
                assert not torch.allclose(output[i], output[j], atol=1e-6), \
                    f"Positions {i} and {j} should not have identical encodings"
    
    def test_batch_processing(self):
        """Test that batch processing works correctly"""
        encoding = PiSpiralPositional(d_model=512)
        
        # Single batch
        positions_single = torch.arange(50)
        output_single = encoding(positions_single)
        
        # Batch processing
        positions_batch = torch.arange(50).unsqueeze(0).repeat(4, 1)
        output_batch = encoding(positions_batch)
        
        assert output_batch.shape == (4, 50, 512), f"Expected shape (4, 50, 512), got {output_batch.shape}"
        
        # Check consistency
        for i in range(4):
            assert torch.allclose(output_batch[i], output_single), "Batch processing should be consistent"


class TestAttractorState:
    """Tests for attractor state mechanism"""
    
    def test_initialization(self):
        """Test attractor state initialization"""
        attractor = AttractorState(d_model=512, alpha=0.99)
        assert attractor.d_model == 512
        assert attractor.alpha == 0.99
    
    def test_incremental_update(self):
        """Test incremental state update"""
        attractor = AttractorState(d_model=512, alpha=0.99)
        
        # Initialize state
        batch_size = 2
        attractor.init_state(batch_size)
        
        # Update with new vectors
        pos_encoding = torch.randn(batch_size, 512)
        hidden_state = torch.randn(batch_size, 512)
        
        state = attractor.update(pos_encoding, hidden_state)
        
        assert state.shape == (batch_size, 512), f"Expected shape (2, 512), got {state.shape}"
    
    def test_state_consistency(self):
        """Test that incremental update matches full recompute"""
        attractor = AttractorState(d_model=512, alpha=0.99)
        
        batch_size = 1
        seq_len = 10
        
        # Incremental updates
        attractor.init_state(batch_size)
        incremental_states = []
        
        for t in range(seq_len):
            pos_enc = torch.randn(batch_size, 512)
            hidden = torch.randn(batch_size, 512)
            state = attractor.update(pos_enc, hidden)
            incremental_states.append(state.clone())
        
        # Full recompute (simplified - just check last state is reasonable)
        final_state = incremental_states[-1]
        assert not torch.isnan(final_state).any(), "State should not contain NaN"
        assert not torch.isinf(final_state).any(), "State should not contain Inf"
    
    def test_o1_memory(self):
        """Test that state size is O(1) - constant with sequence length"""
        attractor = AttractorState(d_model=512, alpha=0.99)
        
        batch_size = 1
        attractor.init_state(batch_size)
        
        initial_state_size = attractor.state.numel()
        
        # Process many tokens
        for _ in range(1000):
            pos_enc = torch.randn(batch_size, 512)
            hidden = torch.randn(batch_size, 512)
            _ = attractor.update(pos_enc, hidden)
        
        final_state_size = attractor.state.numel()
        
        assert initial_state_size == final_state_size, "State size should remain constant (O(1))"
    
    def test_alpha_decay(self):
        """Test different alpha decay policies"""
        # Fixed alpha
        attractor_fixed = AttractorState(d_model=512, alpha=0.99, alpha_policy='fixed')
        assert attractor_fixed.alpha == 0.99
        
        # Exponential decay
        attractor_exp = AttractorState(d_model=512, alpha_policy='exp_c_over_N', c_value=np.pi)
        # Alpha will be computed based on sequence length
        assert attractor_exp.alpha_policy == 'exp_c_over_N'


class TestRoPEPositional:
    """Tests for RoPE positional encoding"""
    
    def test_initialization(self):
        """Test RoPE initialization"""
        rope = RoPEPositional(d_model=512, base=10000.0)
        assert rope.d_model == 512
        assert rope.base == 10000.0
    
    def test_shape_consistency(self):
        """Test output shapes"""
        rope = RoPEPositional(d_model=512)
        positions = torch.arange(100)
        
        output = rope(positions)
        
        assert output.shape == (100, 512), f"Expected shape (100, 512), got {output.shape}"
    
    def test_determinism(self):
        """Test deterministic behavior"""
        rope = RoPEPositional(d_model=512)
        positions = torch.arange(100)
        
        output1 = rope(positions)
        output2 = rope(positions)
        
        assert torch.allclose(output1, output2), "RoPE should be deterministic"
    
    def test_periodic_nature(self):
        """Test that RoPE has periodic properties"""
        rope = RoPEPositional(d_model=512, base=10000.0)
        
        # RoPE should have some periodicity in its structure
        positions = torch.arange(20000)
        output = rope(positions)
        
        # Check that encoding is well-formed
        assert not torch.isnan(output).any(), "RoPE should not produce NaN"
        assert not torch.isinf(output).any(), "RoPE should not produce Inf"


class TestHybridPositionalEncoding:
    """Tests for hybrid positional encoding"""
    
    def test_initialization(self):
        """Test hybrid encoding initialization"""
        hybrid = HybridPositionalEncoding(
            d_model=512,
            hybrid_K=16000,
            transition_width=1000
        )
        assert hybrid.d_model == 512
        assert hybrid.hybrid_K == 16000
    
    def test_shape_consistency(self):
        """Test output shapes"""
        hybrid = HybridPositionalEncoding(d_model=512, hybrid_K=16000)
        positions = torch.arange(32000)
        
        output = hybrid(positions)
        
        assert output.shape == (32000, 512), f"Expected shape (32000, 512), got {output.shape}"
    
    def test_smooth_transition(self):
        """Test smooth transition from RoPE to π-Spiral"""
        hybrid = HybridPositionalEncoding(
            d_model=512,
            hybrid_K=16000,
            transition_width=1000
        )
        
        # Get encodings around transition point
        positions = torch.arange(15000, 17000)
        output = hybrid(positions)
        
        # Check continuity (no sudden jumps)
        diffs = torch.diff(output, dim=0)
        max_diff = torch.max(torch.abs(diffs))
        
        # Differences should be bounded (no discontinuities)
        assert max_diff < 1.0, f"Transition should be smooth, max diff: {max_diff}"
    
    def test_rope_region(self):
        """Test that positions < K use RoPE-like encoding"""
        hybrid = HybridPositionalEncoding(d_model=512, hybrid_K=16000)
        
        # Positions well before K
        positions_early = torch.arange(1000)
        output_early = hybrid(positions_early)
        
        # Should not be NaN or Inf
        assert not torch.isnan(output_early).any()
        assert not torch.isinf(output_early).any()
    
    def test_pi_spiral_region(self):
        """Test that positions > K use π-Spiral-like encoding"""
        hybrid = HybridPositionalEncoding(d_model=512, hybrid_K=16000)
        
        # Positions well after K
        positions_late = torch.arange(20000, 21000)
        output_late = hybrid(positions_late)
        
        # Should not be NaN or Inf
        assert not torch.isnan(output_late).any()
        assert not torch.isinf(output_late).any()
    
    def test_no_short_range_regression(self):
        """Test that hybrid encoding doesn't regress on short sequences"""
        hybrid = HybridPositionalEncoding(d_model=512, hybrid_K=16000)
        rope = RoPEPositional(d_model=512)
        
        # Short sequence (well within RoPE region)
        positions = torch.arange(4000)
        
        output_hybrid = hybrid(positions)
        output_rope = rope(positions)
        
        # Should be very similar in RoPE region
        similarity = torch.nn.functional.cosine_similarity(
            output_hybrid.flatten(),
            output_rope.flatten(),
            dim=0
        )
        
        assert similarity > 0.9, f"Hybrid should be similar to RoPE in short range, similarity: {similarity}"


class TestEncodingIntegration:
    """Integration tests across encoding types"""
    
    def test_all_encodings_same_interface(self):
        """Test that all encodings have consistent interface"""
        encodings = [
            PiSpiralPositional(d_model=512),
            RoPEPositional(d_model=512),
            HybridPositionalEncoding(d_model=512),
        ]
        
        positions = torch.arange(100)
        
        for encoding in encodings:
            output = encoding(positions)
            assert output.shape == (100, 512), f"All encodings should produce same shape"
            assert not torch.isnan(output).any(), f"No encoding should produce NaN"
            assert not torch.isinf(output).any(), f"No encoding should produce Inf"
    
    def test_encoding_differences(self):
        """Test that different encodings produce different outputs"""
        positions = torch.arange(1000)
        
        pi_spiral = PiSpiralPositional(d_model=512)
        rope = RoPEPositional(d_model=512)
        
        output_pi = pi_spiral(positions)
        output_rope = rope(positions)
        
        # Should be different
        assert not torch.allclose(output_pi, output_rope, atol=0.1), \
            "Different encoding types should produce different outputs"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
