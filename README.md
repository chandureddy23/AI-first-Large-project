# AI-first-Large-project
# Healthcare Stroke Prediction with Machine Learning

## **Overview**
This project applies machine learning techniques to predict stroke risk based on health-related data from the **Healthcare Stroke Dataset**. By utilizing preprocessing, exploratory data analysis, and multiple classification algorithms, we evaluate models to accurately predict stroke occurrence.

---

## **Dataset**

### **Dataset Highlights**
- **Source**: Healthcare Stroke Dataset
- **Rows**: 5,110
- **Features**: 12, including demographics, health history, and lifestyle factors

| Feature Name          | Description                                 |
|-----------------------|---------------------------------------------|
| `gender`             | Gender of the patient                      |
| `age`                | Age of the patient                         |
| `hypertension`       | History of hypertension (1 = Yes, 0 = No)  |
| `heart_disease`      | History of heart disease (1 = Yes, 0 = No) |
| `ever_married`       | Marital status                             |
| `work_type`          | Type of employment                         |
| `Residence_type`     | Urban or rural residence                   |
| `avg_glucose_level`  | Average glucose level                      |
| `bmi`                | Body mass index                           |
| `smoking_status`     | Smoking habits                             |
| `stroke`             | Target variable (1 = Stroke, 0 = No Stroke)|

---

## **Workflow**

### **1. Data Preprocessing**
- Filled missing values in the `bmi` column with the mean.
- Dropped the `id` column as it was irrelevant for prediction.
- Encoded categorical variables using `LabelEncoder`.
- Standardized numerical features for consistent scaling.

### **2. Exploratory Data Analysis (EDA)**
- Visualized categorical distributions (e.g., `gender`, `smoking_status`).
- Plotted histograms for numerical variables (`age`, `bmi`, `avg_glucose_level`).
- Computed correlations between numerical features and visualized with a heatmap.

### **3. Model Training and Evaluation**
We trained six machine learning classifiers with hyperparameter tuning using `GridSearchCV`:

1. **Decision Tree**
2. **Random Forest**
3. **Support Vector Machine (SVM)**
4. **Logistic Regression**
5. **K-Nearest Neighbors (KNN)**
6. **Naive Bayes**

#### **Training Details**:
- **Cross-validation**: Stratified K-Fold with 5 splits
- **Performance Metrics**:
  - Accuracy
  - Confusion Matrix

---

## **Results**

| Classifier           | Train Accuracy | Test Accuracy |
|----------------------|----------------|---------------|
| Decision Tree        | 97.00%         | 94.70%        |
| Random Forest        | 96.38%         | 95.62%        |
| Support Vector Machine (SVM) | 95.75% | 95.72%        |
| Logistic Regression  | 95.75%         | 95.72%        |
| K-Nearest Neighbors  | 95.85%         | 95.52%        |
| Naive Bayes          | 87.37%         | 88.49%        |

---

## **Prediction for New Patients**
The project allows users to input new patient data to predict the likelihood of a stroke using the best-performing model.

### Example Input
| Feature             | Example Value    |
|---------------------|------------------|
| Age                | 65               |
| Gender             | Male             |
| Hypertension       | 1 (Yes)          |
| Avg Glucose Level  | 105.2            |
| BMI                | 28.3             |
| Smoking Status     | Formerly Smoked  |

### Example Output
> **The model predicts that the new patient is likely to have a stroke.**

---

## **Technologies Used**
- **Python**: Core programming language
- **Pandas**: For data manipulation
- **Seaborn & Matplotlib**: For data visualization
- **Scikit-Learn**: For model development and evaluation

---

## **How to Run the Project**

### Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn
  ```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/healthcare-stroke-prediction.git
   cd healthcare-stroke-prediction
   ```
2. Run the script or notebook:
   ```bash
   python stroke_prediction.py
   ```

---

## **Future Work**
- Include additional features like medication history.
- Implement deep learning models for enhanced performance.
- Create a web interface for real-time stroke prediction.

---
