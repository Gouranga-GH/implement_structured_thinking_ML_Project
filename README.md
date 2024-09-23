
# California House Prices Prediction using Machine Learning Frameworks

This repository contains a Jupyter notebook that demonstrates the end-to-end data science lifecycle using the **California House Prices** dataset. The project follows a structured approach, incorporating various **data science frameworks** at each phase to ensure clarity and efficiency.

## Project Overview

The main objective of this project is to predict housing prices based on features such as the number of rooms, age of the house, and other factors. The notebook outlines each phase of the data science lifecycle, starting from problem identification to model deployment. In addition to model building, the project also includes **hypothesis testing** and **statistical analysis**.

### Key Phases and Frameworks

1. **Problem Identification & Business Understanding**  
   - Framework: **5 Whys**
   - We investigate the root cause of inaccurate house price predictions using the 5 Whys technique.

2. **Hypothesis Testing**  
   - Perform hypothesis testing using **Pearson correlation** to check the significance of features like the number of rooms (`RM`) on house prices.

3. **Goal Setting**  
   - Framework: **SMART Goals**
   - Set specific, measurable, and time-bound goals for improving model performance, with an R² target above 0.8.

4. **Framing the Analysis**  
   - Framework: **SCQA (Situation, Complication, Question, Answer)**
   - We structure the analysis by asking the right questions and investigating feature correlations using heatmaps.

5. **Data Collection & Preparation**  
   - Framework: **ETL Process**
   - Implement data cleaning and preprocessing, including feature scaling using standardization.

6. **Modeling**  
   - Framework: **Cross-Validation**
   - We build a regression model using **Linear Regression** and evaluate it using **5-fold cross-validation** to ensure model generalization.

7. **Deployment & Monitoring**  
   - Framework: **MLOps Framework**
   - Simulate model deployment and discuss monitoring strategies. The trained model is saved for future use.

## Getting Started

### Prerequisites

You will need the following tools and libraries to run the notebook:

- **Python 3.x**
- **Jupyter Notebook** or **Google Colab**
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `joblib`

To install the required Python packages, you can use the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

### Running the Notebook

1. Download or clone the repository.
2. Open the `california_house_prices_project.ipynb` notebook in Jupyter or Google Colab.
3. Execute each cell in order to run the code and see the analysis.

### Files

- `california_house_prices_project.ipynb` — The main Jupyter notebook containing the code and explanations for the project.

## Project Highlights

- **Structured Thinking**: Each phase of the data science lifecycle is approached using well-established frameworks to enhance clarity and decision-making.
- **Hypothesis Testing**: Statistical tests are used to validate assumptions about the relationships between features and target variables.
- **Machine Learning**: A regression model is built using **Linear Regression** and evaluated with **cross-validation**.
- **Deployment**: Model deployment is simulated with discussions on best practices using MLOps techniques.

## Acknowledgements

- The **California House Prices Dataset** is provided by the **scikit-learn** library and is commonly used for regression analysis in machine learning.

## License

This project is open-source and available under the [MIT License](LICENSE).
