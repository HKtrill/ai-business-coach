#include "recurrent_network.h"
#include <cmath>

namespace churn {

// =============================================================================
// RecurrentNetwork Implementation
// =============================================================================

RecurrentNetwork::RecurrentNetwork(
    const std::vector<std::vector<double>>& Wxh,
    const std::vector<std::vector<double>>& Whh,
    const std::vector<std::vector<double>>& Why,
    const std::vector<double>& bh,
    const std::vector<double>& by
) : Wxh_(Wxh), Whh_(Whh), Why_(Why), bh_(bh), by_(by) {}

std::vector<double> RecurrentNetwork::predict(
    const std::vector<std::vector<double>>& sequence
) const {
    int hidden_size = bh_.size();
    std::vector<double> h(hidden_size, 0.0);
    
    for (const auto& x : sequence) {
        std::vector<double> h_new(hidden_size, 0.0);
        
        for (size_t i = 0; i < hidden_size; ++i) {
            double sum = bh_[i];
            for (size_t j = 0; j < x.size(); ++j) {
                sum += Wxh_[i][j] * x[j];
            }
            for (size_t j = 0; j < hidden_size; ++j) {
                sum += Whh_[i][j] * h[j];
            }
            h_new[i] = std::tanh(sum);
        }
        h = h_new;
    }
    
    std::vector<double> output(by_.size());
    for (size_t i = 0; i < by_.size(); ++i) {
        double sum = by_[i];
        for (size_t j = 0; j < hidden_size; ++j) {
            sum += Why_[i][j] * h[j];
        }
        output[i] = sum;
    }
    
    return output;
}

} // namespace churn