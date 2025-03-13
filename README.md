# -Linear-Regression-Health-Costs-Calculator
# Health Costs Prediction using Linear Regression

## Overview
This project predicts healthcare costs using a **Linear Regression model** trained on the **insurance dataset**. It preprocesses the data, trains the model, and evaluates performance using **Mean Absolute Error (MAE)**.

## Dataset
The dataset includes:
- **age**: Age of the individual
- **sex**: Gender (Male/Female)
- **bmi**: Body Mass Index
- **children**: Number of dependents
- **smoker**: Whether the individual smokes or not
- **region**: Geographical location
- **expenses**: Medical expenses (Target Variable)

## Steps
1. **Load & Preprocess Data**
   - Convert categorical columns to numeric values using one-hot encoding.
   - Normalize numerical values using **StandardScaler**.
   
2. **Split Data**
   - 80% for training
   - 20% for testing

3. **Build and Train the Model**
   - **Neural Network Model (TensorFlow/Keras)**
   - 2 Hidden layers (ReLU activation)
   - Optimizer: **Adam**
   - Loss function: **Mean Squared Error (MSE)**

4. **Evaluate Model Performance**
   - The model should achieve **MAE < 3500**.

5. **Visualize Predictions**
   - Compares actual vs. predicted costs using a **scatter plot**.

## Requirements
- Python 3.x
- TensorFlow/Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Running the Project
1. Install dependencies:
   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib
   ```
2. Run the script in **Google Colaboratory** or locally.
3. Check the **MAE score** and **visualization plot**.

## Expected Output
- **MAE < 3500** ensures accurate predictions.
- Scatter plot showing **actual vs. predicted** healthcare costs.

## Conclusion
This project demonstrates **regression modeling** for healthcare cost prediction using **neural networks**. It highlights **data preprocessing, feature engineering, and model evaluation** techniques.

