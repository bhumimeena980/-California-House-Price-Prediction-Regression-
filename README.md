California House Price Prediction (Regression)

This project predicts median house prices in California using features like population, income, and housing attributes. It uses a **Neural Network** (built with TensorFlow/Keras) trained on the popular `fetch_california_housing` dataset from Scikit-learn.

---

## Project Overview

- **Goal**: Predict house prices (in $100,000s) using real estate and demographic features.
- **Model Type**: Regression (Neural Network)
- **Dataset**: Built-in Scikit-learn `fetch_california_housing` dataset
- **Libraries**: `TensorFlow`, `scikit-learn`, `matplotlib`,`seaborn`

 
 ## 📈 Model Summary

- **Architecture**:
  - Input Layer: 8 features
  - Hidden Layer 1: 64 neurons (ReLU)
  - Hidden Layer 2: 32 neurons (ReLU)
  - Output Layer: 1 neuron (linear)

- **Performance**:
  - **Mean Squared Error (MSE)**: _(example)_ `0.42`
  - **R² Score**: _(example)_ `0.78`
`
