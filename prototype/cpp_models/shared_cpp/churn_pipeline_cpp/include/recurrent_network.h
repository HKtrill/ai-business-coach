#pragma once
#include <vector>

namespace churn {

class RecurrentNetwork {
public:
    RecurrentNetwork(
        const std::vector<std::vector<double>>& Wxh,
        const std::vector<std::vector<double>>& Whh,
        const std::vector<std::vector<double>>& Why,
        const std::vector<double>& bh,
        const std::vector<double>& by
    );
    
    std::vector<double> predict(const std::vector<std::vector<double>>& sequence) const;

private:
    std::vector<std::vector<double>> Wxh_, Whh_, Why_;
    std::vector<double> bh_, by_;
};

} // namespace churn