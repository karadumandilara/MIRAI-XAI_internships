# Model Interpretability

## Overview
This repository contains implementations of SHAP (SHapley Additive Explanations) and LIME (Local Interpretable Model-agnostic Explanations) for model interpretability.
## Explanation Techniques

### 1. SHAP (SHapley Additive Explanations)
SHAP is an interpretability method based on Shapley values. It quantifies the contribution of each feature to a model’s predictions by fairly distributing the prediction differences among features.

- **Advantages**:
  - Provides global and local interpretability.
  - Considers feature interactions.
  - Theoretically solid due to Shapley values.

- **Example Usage**:
  ```python
  import shap
  explainer = shap.Explainer(model, data)
  shap_values = explainer(data)
  shap.summary_plot(shap_values, data)
  ```

### 2. LIME (Local Interpretable Model-agnostic Explanations)
LIME explains individual predictions by perturbing input features and training a local surrogate model.

- **Advantages**:
  - Model-agnostic and easy to implement.
  - Provides intuitive local explanations.

- **Example Usage**:
  ```python
  from lime.lime_tabular import LimeTabularExplainer
  explainer = LimeTabularExplainer(data, feature_names=features, class_names=classes, mode='classification')
  explanation = explainer.explain_instance(instance, model.predict_proba)
  explanation.show_in_notebook()
  ```

## References
- Interpretable Machine Learning by Christoph Molnar: [Read here](https://christophm.github.io/interpretable-ml-book/)
