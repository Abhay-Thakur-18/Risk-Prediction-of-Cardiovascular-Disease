# Risk Prediction of Cardiovascular Disease

An end-to-end machine learning project to predict the risk of cardiovascular disease based on individual patient demographic indicators, health metrics, and lifestyle behaviors.

---

## ?? Project Overview
Cardiovascular disease (CVD) is one of the leading causes of mortality globally. Early detection and risk assessment can significantly assist healthcare professionals and individuals in adopting preventive measures. 

This project explores a real-world cardiovascular health dataset of **308,854 records** with **19 features**, covering demographic attributes, medical history, and lifestyle habits. The project involves rigorous data cleaning, exploratory data analysis (EDA), outlier detection & treatment using the IQR method, categorical encoding, feature scaling, multicollinearity assessment (VIF), and evaluation of multiple classification algorithms to determine the most effective predictive model.

---

## ?? Dataset

- **Dataset File:** data/CVD_cleaned.csv
- **Total Records:** 308,854 rows
- **Total Features:** 19 columns
- **Target Variable:** Heart_Disease (0 = No, 1 = Yes)

### Features Description
| Category | Features |
| :--- | :--- |
| **Demographics & Physical** | Height_(cm), Weight_(kg), BMI *(analyzed and dropped due to multicollinearity with Weight)*, Sex, Age_Category |
| **Medical History** | General_Health, Checkup, Skin_Cancer, Other_Cancer, Depression, Diabetes, Arthritis |
| **Lifestyle & Habits** | Exercise, Smoking_History, Alcohol_Consumption, Fruit_Consumption, Green_Vegetables_Consumption, FriedPotato_Consumption |

---

## ?? Data Preprocessing & Pipeline

1. **Noise Identification & Cleaning:**
   - Standardized inconsistent categories in General_Health (e.g., 'Poo', 'poo' mapped to 'Poor').
   - Cleaned Diabetes and Sex categorical values.
   - Verified no missing/null values across the dataset.

2. **Exploratory Data Analysis (EDA):**
   - **Univariate Analysis:** Analyzed value distributions and frequency distributions across categorical and numerical features.
   - **Skewness Analysis:** Detected right-skewed dietary consumption features (FriedPotato_Consumption, Green_Vegetables_Consumption).
   - **Bivariate & Multivariate Analysis:** Correlation matrices and heatmaps to inspect relationships between lifestyle factors, body metrics, and cardiovascular disease risk.
   - Dropped redundant high-correlation features (BMI was dropped due to high multicollinearity with Weight_(kg)).

3. **Outlier Detection & Treatment:**
   - Utilized the **IQR (Interquartile Range)** technique to calculate bounds ( - 1.5 \times IQR$ and  + 1.5 \times IQR$).
   - Successfully treated extreme outliers across numerical variables (Height_(cm), Weight_(kg), Alcohol_Consumption, Fruit_Consumption, Green_Vegetables_Consumption).

4. **Feature Encoding:**
   - **Ordinal Encoding:** Applied to ordered categorical attributes (General_Health, Checkup, Age_Category).
   - **Binary Encoding:** Applied using category_encoders.BinaryEncoder on binary attributes (Exercise, Skin_Cancer, Other_Cancer, Depression, Diabetes, Sex, Arthritis, Smoking_History).

5. **Scaling & Multicollinearity Check:**
   - Scaled feature matrix using StandardScaler.
   - Verified feature independence by computing the **Variance Inflation Factor (VIF)**.

6. **Train-Test Split:**
   - Split into **75% training** and **25% testing** sets with andom_state=50.

---

## ?? Machine Learning Models & Results

Four classification algorithms were trained and benchmarked on the dataset:

| Model | Training Accuracy | Testing Accuracy | Generalization Notes |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | 91.38% | **91.75%** | Strong linear baseline, fast inference |
| **Random Forest Classifier** | 99.99% | **91.64%** | Best ensemble performance, captures complex feature interactions |
| **K-Nearest Neighbors (KNN)** | 92.22% | **90.52%** | Good local pattern recognition |
| **Decision Tree Classifier** | 99.99% | **85.95%** | Prone to overfitting on training split |

### Model Selection & Serialization
The **Random Forest Classifier** was selected and saved as the final production model artifact:
- **Model Path:** models/CVD_predictor.pkl
- **Saved Using:** joblib

---

## ??? Technologies Used

- **Language:** Python 3.12+
- **Data Manipulation:** pandas, 
umpy
- **Data Visualization:** seaborn, matplotlib
- **Machine Learning:** scikit-learn
- **Feature Engineering:** category_encoders, statsmodels
- **Model Persistence:** joblib
- **Environment:** Jupyter Notebook

---

## ?? Project Structure

`	ext
Risk-Prediction-of-Cardiovascular-Disease/
¦
+-- Risk_Prediction_of_Cardiovascular_Disease.ipynb  # Main Jupyter notebook with end-to-end analysis & models
+-- README.md                                        # Project documentation and summary
+-- requirements.txt                                 # Python dependencies
+-- .gitignore                                       # Files and folders ignored by Git
+-- data/
¦   +-- CVD_cleaned.csv                              # Cardiovascular dataset (308,854 rows)
+-- models/
    +-- CVD_predictor.pkl                            # Serialized Random Forest predictive model
`

---

## ?? How to Run the Project

### 1. Clone the Repository
`ash
git clone https://github.com/Abhay-Thakur-18/Risk-Prediction-of-Cardiovascular-Disease.git
cd Risk-Prediction-of-Cardiovascular-Disease
`

### 2. Set Up Virtual Environment & Dependencies
`ash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
`

### 3. Run the Jupyter Notebook
`ash
jupyter notebook Risk_Prediction_of_Cardiovascular_Disease.ipynb
`

### 4. Load the Pre-trained Model in Python
`python
import joblib

# Load the trained RandomForest model
model = joblib.load('models/CVD_predictor.pkl')
print(\"Model loaded successfully:\", type(model))
`

---

## ?? Future Scope
- Integration with an interactive web UI (Streamlit / Flask / FastAPI) for real-time risk assessment.
- Addressing class imbalance using techniques like SMOTE or cost-sensitive learning to further improve recall on positive disease cases.
- Hyperparameter tuning and threshold calibration for clinical decision support.
