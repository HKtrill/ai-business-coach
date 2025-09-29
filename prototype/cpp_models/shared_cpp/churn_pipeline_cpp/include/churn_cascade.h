#ifndef CHURN_CASCADE_H
#define CHURN_CASCADE_H

#include <vector>
#include <memory>
#include <string>

namespace churn {

// Forward declarations
class LogisticRegression;
class NeuralNetwork;
class RecurrentNetwork;

/**
 * @brief Complete cascade ensemble for churn prediction
 * 
 * Implements the full 3-stage cascade:
 * - Stage 1: Lasso Logistic Regression (L1 regularization, feature selection)
 * - Stage 2: MLP Neural Network (100, 50 hidden layers)
 * - Stage 3: RNN (LSTM with 64 hidden units, 2 layers)
 * 
 * Ensemble: 0.4*Lasso + 0.3*MLP + 0.3*RNN, threshold=0.5
 */
class ChurnCascade {
public:
    /**
     * @brief Constructor
     * 
     * @param random_state Random seed for reproducibility
     */
    explicit ChurnCascade(int random_state = 42);
    
    ~ChurnCascade();
    
    /**
     * @brief Train the complete cascade pipeline
     * 
     * This matches the Python implementation:
     * 1. Apply balanced sampling (BorderlineSMOTE + RandomUnderSampler)
     * 2. Train Stage 1: Lasso Logistic Regression (C=0.3, L1, balanced weights)
     * 3. Train Stage 2: MLP (100,50 layers, adam, alpha=0.001)
     * 4. Train Stage 3: RNN (64 hidden, 2 layers, dropout=0.3)
     * 
     * @param X_train Training features [n_samples x n_features]
     * @param y_train Training labels [n_samples]
     * @param smote_strategy SMOTE sampling strategy (default: 0.6)
     * @param undersample_strategy Undersample ratio (default: 0.82)
     */
    void fit(
        const std::vector<std::vector<double>>& X_train,
        const std::vector<int>& y_train,
        double smote_strategy = 0.6,
        double undersample_strategy = 0.82
    );
    
    /**
     * @brief Predict churn probabilities
     * 
     * Returns ensemble probabilities: 0.4*Lasso + 0.3*MLP + 0.3*RNN
     * 
     * @param X_test Test features [n_samples x n_features]
     * @return Churn probabilities [n_samples]
     */
    std::vector<double> predict_proba(
        const std::vector<std::vector<double>>& X_test
    ) const;
    
    /**
     * @brief Predict churn labels (0 or 1)
     * 
     * Uses threshold=0.5 on ensemble probabilities
     * 
     * @param X_test Test features [n_samples x n_features]
     * @return Binary predictions [n_samples]
     */
    std::vector<int> predict(
        const std::vector<std::vector<double>>& X_test
    ) const;
    
    /**
     * @brief Get feature importance from Lasso stage
     * 
     * @return Absolute values of Lasso coefficients
     */
    std::vector<double> get_feature_importance() const;
    
    /**
     * @brief Get raw Lasso coefficients (with signs)
     * 
     * @return Raw Lasso coefficients
     */
    std::vector<double> get_lasso_coefficients() const;
    
    /**
     * @brief Get number of features selected by Lasso
     * 
     * @return Count of non-zero coefficients
     */
    int get_num_selected_features() const;
    
    // Getters for models
    const LogisticRegression* get_stage1_model() const { return stage1_model_.get(); }
    const NeuralNetwork* get_stage2_model() const { return stage2_model_.get(); }
    const RecurrentNetwork* get_stage3_model() const { return stage3_model_.get(); }
    
private:
    // Stage models
    std::unique_ptr<LogisticRegression> stage1_model_;
    std::unique_ptr<NeuralNetwork> stage2_model_;
    std::unique_ptr<RecurrentNetwork> stage3_model_;
    
    // Configuration
    int random_state_;
    double ensemble_weights_[3] = {0.4, 0.3, 0.3};  // Lasso, MLP, RNN
    double threshold_ = 0.5;
    
    // Fitted flag
    bool is_fitted_ = false;
    
    // Helper functions
    void apply_balanced_sampling(
        std::vector<std::vector<double>>& X,
        std::vector<int>& y,
        double smote_strategy,
        double undersample_strategy
    );
    
    void validate_input(
        const std::vector<std::vector<double>>& X,
        const std::vector<int>& y
    ) const;
};

/**
 * @brief Lasso Logistic Regression with L1 regularization
 * 
 * Implements coordinate descent for L1-regularized logistic regression
 * with balanced class weights.
 */
class LogisticRegression {
public:
    /**
     * @param C Inverse regularization strength (default: 0.3)
     * @param max_iter Maximum iterations (default: 1000)
     * @param tol Convergence tolerance (default: 1e-4)
     * @param random_state Random seed
     */
    LogisticRegression(double C = 0.3, int max_iter = 1000, 
                       double tol = 1e-4, int random_state = 42);
    
    void fit(const std::vector<std::vector<double>>& X, 
             const std::vector<int>& y);
    
    std::vector<double> predict_proba(
        const std::vector<std::vector<double>>& X
    ) const;
    
    std::vector<double> get_coefficients() const { return coefficients_; }
    double get_intercept() const { return intercept_; }
    
private:
    double C_;
    int max_iter_;
    double tol_;
    int random_state_;
    
    std::vector<double> coefficients_;
    double intercept_;
    
    double sigmoid(double x) const;
    void coordinate_descent(
        const std::vector<std::vector<double>>& X,
        const std::vector<int>& y
    );
};

/**
 * @brief Multi-layer Perceptron Neural Network
 * 
 * Implements MLP with:
 * - Hidden layers: [100, 50]
 * - Activation: ReLU
 * - Optimizer: Adam
 * - Alpha (L2): 0.001
 * - Early stopping: 10 iterations
 */
class NeuralNetwork {
public:
    /**
     * @param hidden_layers Hidden layer sizes (default: {100, 50})
     * @param alpha L2 regularization (default: 0.001)
     * @param learning_rate Learning rate (default: 0.001)
     * @param max_iter Maximum epochs (default: 300)
     * @param random_state Random seed
     */
    NeuralNetwork(const std::vector<int>& hidden_layers = {100, 50},
                  double alpha = 0.001, double learning_rate = 0.001,
                  int max_iter = 300, int random_state = 42);
    
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y);
    
    std::vector<double> predict_proba(
        const std::vector<std::vector<double>>& X
    ) const;
    
private:
    std::vector<int> hidden_layers_;
    double alpha_;
    double learning_rate_;
    int max_iter_;
    int random_state_;
    
    // Network weights
    std::vector<std::vector<std::vector<double>>> weights_;
    std::vector<std::vector<double>> biases_;
    
    // Adam optimizer state
    std::vector<std::vector<std::vector<double>>> m_weights_;
    std::vector<std::vector<std::vector<double>>> v_weights_;
    std::vector<std::vector<double>> m_biases_;
    std::vector<std::vector<double>> v_biases_;
    
    void initialize_weights(int input_size);
    std::vector<double> forward(const std::vector<double>& x) const;
    double relu(double x) const;
    double sigmoid(double x) const;
};

} // namespace churn

#endif // CHURN_CASCADE_H