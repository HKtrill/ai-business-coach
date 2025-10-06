"""
Python wrapper for C++ RNN implementation.

Usage:
    python cascade_model_cpp_wrapper.py
"""

import numpy as np
import churn_rnn_cpp


def create_toy_weights(input_size=5, hidden_size=10, output_size=1):
    """Create random toy weights for testing."""
    # Xavier initialization
    Wxh = (np.random.randn(hidden_size, input_size) * 
           np.sqrt(2.0 / (input_size + hidden_size))).tolist()
    
    Whh = (np.random.randn(hidden_size, hidden_size) * 
           np.sqrt(2.0 / (hidden_size + hidden_size))).tolist()
    
    Why = (np.random.randn(output_size, hidden_size) * 
           np.sqrt(2.0 / (hidden_size + output_size))).tolist()
    
    bh = np.zeros(hidden_size).tolist()
    by = np.zeros(output_size).tolist()
    
    return Wxh, Whh, Why, bh, by


def create_toy_sequence(seq_length=20, input_size=5):
    """Create a toy input sequence."""
    # Create a simple pattern: alternating high/low values
    sequence = []
    for t in range(seq_length):
        if t % 4 < 2:
            # High values
            x = (np.random.randn(input_size) * 0.5 + 1.0).tolist()
        else:
            # Low values
            x = (np.random.randn(input_size) * 0.5 - 1.0).tolist()
        sequence.append(x)
    
    return sequence


def main():
    print("=" * 60)
    print("C++ RNN Demo - Churn Prediction Pipeline")
    print("=" * 60)
    
    # Configuration
    input_size = 5
    hidden_size = 10
    output_size = 1
    seq_length = 20
    
    print(f"\nNetwork Configuration:")
    print(f"  Input size:  {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Output size: {output_size}")
    print(f"  Sequence length: {seq_length}")
    
    # Create weights
    print("\n[1/4] Creating toy weights...")
    Wxh, Whh, Why, bh, by = create_toy_weights(input_size, hidden_size, output_size)
    print("  ✓ Weights initialized")
    
    # Initialize C++ RNN
    print("\n[2/4] Initializing C++ RNN...")
    rnn = churn_rnn_cpp.RecurrentNetwork(Wxh, Whh, Why, bh, by)
    print(f"  ✓ {rnn}")
    
    # Create toy input sequence
    print("\n[3/4] Creating toy input sequence...")
    inputs = create_toy_sequence(seq_length, input_size)
    print(f"  ✓ Generated sequence of length {len(inputs)}")
    
    # Run forward pass
    print("\n[4/4] Running forward pass...")
    outputs = rnn.forward(inputs)
    print(f"  ✓ Forward pass complete")
    print(f"  ✓ Output shape: [{len(outputs)} x {len(outputs[0])}]")
    
    # Get final prediction
    prediction = rnn.predict(inputs)
    print(f"\n{'='*60}")
    print(f"Final Prediction (churn probability): {prediction[0]:.6f}")
    print(f"{'='*60}")
    
    # Show output evolution
    print("\nOutput Evolution (first 10 timesteps):")
    print("-" * 40)
    for t in range(min(10, len(outputs))):
        print(f"  t={t:2d}: {outputs[t][0]:8.5f}")
    
    if len(outputs) > 10:
        print("  ...")
        print(f"  t={len(outputs)-1:2d}: {outputs[-1][0]:8.5f}")
    
    # Demonstrate state reset
    print("\n" + "="*60)
    print("Testing state reset functionality...")
    print("="*60)
    
    # Run again without reset
    prediction_1 = rnn.predict(inputs)[0]
    prediction_2 = rnn.predict(inputs)[0]
    
    print(f"\nRun 1 (continuing state): {prediction_1:.6f}")
    print(f"Run 2 (continuing state): {prediction_2:.6f}")
    print(f"Difference: {abs(prediction_2 - prediction_1):.6f}")
    
    # Reset and run again
    rnn.reset_state()
    prediction_3 = rnn.predict(inputs)[0]
    
    print(f"\nRun 3 (after reset): {prediction_3:.6f}")
    print(f"Difference from Run 1: {abs(prediction_3 - prediction_1):.6f}")
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()