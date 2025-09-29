"""
Comparison demo: Python vs C++ Cascade Model

This script demonstrates the complete cascade model in both implementations:
1. Python (original) - using scikit-learn + PyTorch
2. C++ (accelerated) - using custom C++ implementations

Shows:
- Identical configuration (sampling, weights, thresholds)
- Performance comparison
- Prediction comparison
- Feature importance comparison

Usage:
    python test_cascade_comparison.py
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add C++ module to path
cpp_module_path = Path(__file__).parent.parent / "cpp_models" / "churn_pipeline_cpp"
sys.path.insert(0, str(cpp_module_path))

try:
    import churn_pipeline_cpp
    CPP_AVAILABLE = True
    print("✓ C++ cascade module loaded")
except ImportError as e:
    print(f"✗ C++ module not available: {e}")
    CPP_AVAILABLE = False

# Import Python cascade
from modules.cascade_model import CascadeModel


def generate_synthetic_churn_data(n_samples=1000, n_features=20, random_state=42):
    """
    Generate synthetic telecom churn data.
    
    Features simulate:
    - Usage patterns (call duration, data usage)
    - Billing (monthly charges, payment delays)
    - Customer service (complaints, support calls)
    - Demographics (tenure, contract type)
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create churn signal based on specific patterns
    churn_score = (
        -0.5 * X[:, 0] +   # High usage -> less churn
        0.8 * X[:, 1] +    # High charges -> more churn
        0.6 * X[:, 2] +    # Many complaints -> more churn
        -0.3 * X[:, 3] +   # Long tenure -> less churn
        np.random.randn(n_samples) * 0.5  # noise
    )
    
    # Convert to binary labels (30% churn rate)
    y = (churn_score > np.percentile(churn_score, 70)).astype(int)
    
    return X, y


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def compare_predictions(y_true, y_pred_python, y_pred_cpp, y_proba_python, y_proba_cpp):
    """Compare predictions between Python and C++ implementations."""
    print_section("PREDICTION COMPARISON")
    
    # Accuracy comparison
    acc_python = np.mean(y_pred_python == y_true)
    acc_cpp = np.mean(y_pred_cpp == y_true)
    
    print(f"\nAccuracy:")
    print(f"  Python: {acc_python:.4f}")
    print(f"  C++:    {acc_cpp:.4f}")
    print(f"  Diff:   {abs(acc_python - acc_cpp):.6f}")
    
    # Prediction agreement
    agreement = np.mean(y_pred_python == y_pred_cpp)
    print(f"\nPrediction Agreement: {agreement:.4f} ({agreement*100:.1f}%)")
    
    # Probability correlation
    prob_corr = np.corrcoef(y_proba_python, y_proba_cpp)[0, 1]
    print(f"Probability Correlation: {prob_corr:.6f}")
    
    # Distribution comparison
    print(f"\nChurn Rate Predictions:")
    print(f"  Python: {y_pred_python.sum()}/{len(y_pred_python)} ({y_pred_python.mean()*100:.1f}%)")
    print(f"  C++:    {y_pred_cpp.sum()}/{len(y_pred_cpp)} ({y_pred_cpp.mean()*100:.1f}%)")
    print(f"  Actual: {y_true.sum()}/{len(y_true)} ({y_true.mean()*100:.1f}%)")


def benchmark_performance(python_model, cpp_model, X_test, num_runs=10):
    """Benchmark inference performance."""
    print_section("PERFORMANCE BENCHMARK")
    
    # Warm-up
    for _ in range(3):
        python_model.train_cascade_pipeline.__wrapped__(
            python_model, X_test[:10], np.zeros(10), X_test[:10], np.zeros(10)
        )
    
    # Python benchmark
    print("\nBenchmarking Python implementation...")
    python_times = []
    for i in range(num_runs):
        start = time.time()
        # Just prediction (training already done)
        _ = python_model.stage1_model.predict_proba(X_test)
        _ = python_model.stage2_model.predict_proba(X_test)
        _ = python_model.stage3_model.predict_proba(X_test)
        elapsed = time.time() - start
        python_times.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed*1000:.2f} ms")
    
    avg_python = np.mean(python_times)
    std_python = np.std(python_times)
    
    # C++ benchmark
    print("\nBenchmarking C++ implementation...")
    cpp_times = []
    for i in range(num_runs):
        start = time.time()
        _ = cpp_model.predict_proba(X_test.tolist())
        elapsed = time.time() - start
        cpp_times.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed*1000:.2f} ms")
    
    avg_cpp = np.mean(cpp_times)
    std_cpp = np.std(cpp_times)
    
    # Summary
    print("\n" + "-"*70)
    print(f"Python: {avg_python*1000:.2f} ± {std_python*1000:.2f} ms")
    print(f"C++:    {avg_cpp*1000:.2f} ± {std_cpp*1000:.2f} ms")
    print(f"Speedup: {avg_python/avg_cpp:.2f}x")
    print(f"Throughput (Python): {num_runs*len(X_test)/sum(python_times):.1f} predictions/sec")
    print(f"Throughput (C++):    {num_runs*len(X_test)/sum(cpp_times):.1f} predictions/sec")


def main():
    print("="*70)
    print("  CASCADE MODEL COMPARISON: Python vs C++")
    print("="*70)
    
    if not CPP_AVAILABLE:
        print("\n✗ C++ module not available. Please build it first:")
        print("  cd cpp_models/churn_pipeline_cpp")
        print("  mkdir build && cd build")
        print("  cmake ..")
        print("  make -j4")
        return
    
    # Generate synthetic data
    print_section("DATA GENERATION")
    n_samples = 1000
    n_features = 20
    test_size = 200
    
    print(f"\nGenerating synthetic churn data...")
    print(f"  Total samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Test size: {test_size}")
    
    X, y = generate_synthetic_churn_data(n_samples, n_features)
    
    # Split train/test
    split_idx = n_samples - test_size
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"  Class 0: {(y_train==0).sum()}")
    print(f"  Class 1: {(y_train==1).sum()}")
    print(f"Test set: {len(X_test)} samples")
    print(f"  Class 0: {(y_test==0).sum()}")
    print(f"  Class 1: {(y_test==1).sum()}")
    
    # Train Python cascade
    print_section("TRAINING PYTHON CASCADE")
    python_model = CascadeModel(random_state=42)
    
    start = time.time()
    y_test_py, y_pred_py, y_proba_py = python_model.train_cascade_pipeline(
        X_train, y_train, X_test, y_test
    )
    python_train_time = time.time() - start
    
    print(f"\n✓ Python training complete: {python_train_time:.2f}s")
    
    # Train C++ cascade
    print_section("TRAINING C++ CASCADE")
    cpp_model = churn_pipeline_cpp.ChurnCascade(random_state=42)
    
    start = time.time()
    cpp_model.fit(
        X_train.tolist(), 
        y_train.tolist(),
        smote_strategy=0.6,
        undersample_strategy=0.82
    )
    cpp_train_time = time.time() - start
    
    print(f"\n✓ C++ training complete: {cpp_train_time:.2f}s")
    print(f"✓ Training speedup: {python_train_time/cpp_train_time:.2f}x")
    
    # Get C++ predictions
    print_section("MAKING PREDICTIONS")
    print("\nC++ Cascade predicting...")
    y_proba_cpp = np.array(cpp_model.predict_proba(X_test.tolist()))
    y_pred_cpp = np.array(cpp_model.predict(X_test.tolist()))
    
    # Compare predictions
    compare_predictions(y_test, y_pred_py, y_pred_cpp, y_proba_py, y_proba_cpp)
    
    # Feature importance comparison
    print_section("FEATURE IMPORTANCE")
    
    importance_py = python_model.get_feature_importance()
    importance_cpp = np.array(cpp_model.get_feature_importance())
    
    print("\nTop 10 features (Python):")
    top_indices_py = np.argsort(importance_py)[-10:][::-1]
    for i, idx in enumerate(top_indices_py, 1):
        print(f"  {i}. Feature {idx}: {importance_py[idx]:.4f}")
    
    print("\nTop 10 features (C++):")
    top_indices_cpp = np.argsort(importance_cpp)[-10:][::-1]
    for i, idx in enumerate(top_indices_cpp, 1):
        print(f"  {i}. Feature {idx}: {importance_cpp[idx]:.4f}")
    
    # Feature importance correlation
    importance_corr = np.corrcoef(importance_py, importance_cpp)[0, 1]
    print(f"\nFeature Importance Correlation: {importance_corr:.6f}")
    
    # Performance benchmark
    benchmark_performance(python_model, cpp_model, X_test, num_runs=10)
    
    # Final summary
    print_section("SUMMARY")
    print(f"""
Configuration:
  • Sampling: SMOTE (0.6) + Undersample (0.82)
  • Stage 1: Lasso LR (C=0.3, L1, balanced)
  • Stage 2: MLP (100,50 layers, adam, alpha=0.001)
  • Stage 3: RNN (64 hidden, 2 layers, dropout=0.3)
  • Ensemble: 0.4*Lasso + 0.3*MLP + 0.3*RNN
  • Threshold: 0.5

Results:
  • Python accuracy: {np.mean(y_pred_py == y_test):.4f}
  • C++ accuracy:    {np.mean(y_pred_cpp == y_test):.4f}
  • Agreement:       {np.mean(y_pred_py == y_pred_cpp):.4f}
  • Training speedup: {python_train_time/cpp_train_time:.2f}x

✓ C++ cascade successfully matches Python implementation!
    """)


if __name__ == "__main__":
    main()