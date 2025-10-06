#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "churn_cascade.h"

namespace py = pybind11;

PYBIND11_MODULE(churn_cascade_cpp, m) {
    m.doc() = "C++ Churn Cascade Model";

    py::class_<churn::ChurnCascade>(m, "ChurnCascade")
        .def(py::init<int>(), py::arg("random_state") = 42)
        .def("fit", &churn::ChurnCascade::fit,
             py::arg("X_train"),
             py::arg("y_train"),
             py::arg("smote_strategy") = 0.6,
             py::arg("undersample_strategy") = 0.82,
             "Train the cascade model")
        .def("predict", &churn::ChurnCascade::predict,
             py::arg("X_test"),
             "Make predictions")
        .def("predict_proba", &churn::ChurnCascade::predict_proba,
             py::arg("X_test"),
             "Predict probabilities")
        .def("get_feature_importance", &churn::ChurnCascade::get_feature_importance,
             "Get feature importance from Lasso stage");
}