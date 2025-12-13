# Loan Default Prediction - Machine Learning Classification Project

## 📊 Project Overview
A binary classification project that predicts whether a loan applicant will **default** or **repay** their loan based on financial and personal characteristics. This project compares four different machine learning classifiers to identify the best model for loan risk assessment.

## 🎯 Problem Statement

**Business Question**: Can we predict which loan applicants are likely to default on their loans?

**Scenario**: A bank wants to automate the loan approval process by identifying high-risk applicants who are likely to default versus low-risk applicants who will repay their loans.

**Goal**: Build and compare classification models to predict loan default risk.

**Problem Type**: Binary Classification
- **Class 0**: No Default (Low Risk - will repay)
- **Class 1**: Default (High Risk - will not repay)

**Business Value**: 
- Reduce financial losses from loan defaults
- Automate credit risk assessment
- Make faster, data-driven lending decisions
- Identify key factors that indicate default risk

## 🎓 Academic Requirements Met
✅ Binary classification problem with real-world application  
✅ Real dataset (not randomly generated with Python)  
✅ Minimum 2 algorithms required, implementing 4 different classifiers  
✅ Complete workflow: data exploration → cleaning → modeling → evaluation  
✅ Similar structure to the course classification example (loan approval problem)  

## 🤖 Classification Algorithms Compared

We implement and compare **4 different classifiers**:

### 1. **Random Forest Classifier**
- Ensemble of multiple decision trees
- Excellent for financial data with complex patterns
- Handles non-linear relationships well
- Provides feature importance rankings

### 2. **Gaussian Naive Bayes**
- Probabilistic classifier based on Bayes' theorem
- Fast training and prediction
- Works well with independent features
- Creates smooth decision boundaries

### 3. **Logistic Regression**
- Linear classification model
- Industry standard for credit risk modeling
- Highly interpretable coefficients
- Provides probability estimates for default risk

### 4. **Support Vector Machine (RBF Kernel)**
- Powerful non-linear classifier
- Uses kernel trick for complex decision boundaries
- Effective in high-dimensional feature spaces
- Often achieves high accuracy on structured data

## 📊 Dataset Description

**Source**: Loan Default Prediction Dataset

**Size**: ~50,000 loan applications (approximate)

**Target Variable**:
- `Default` - Binary (0 = No Default, 1 = Default)

**Features**:

**Numerical Features** (9):
- `Age` - Applicant's age
- `Income` - Annual income
- `LoanAmount` - Requested loan amount
- `CreditScore` - Credit score (300-850)
- `MonthsEmployed` - Length of current employment
- `NumCreditLines` - Number of open credit lines
- `InterestRate` - Loan interest rate
- `LoanTerm` - Loan duration in months
- `DTIRatio` - Debt-to-Income ratio

**Categorical Features** (4):
- `Education` - Education level
- `EmploymentType` - Type of employment
- `MaritalStatus` - Marital status
- `LoanPurpose` - Purpose of the loan

**Binary Features** (3):
- `HasMortgage` - Has existing mortgage (Yes/No)
- `HasDependents` - Has dependents (Yes/No)
- `HasCoSigner` - Has loan co-signer (Yes/No)

## 🛠️ Tech Stack
- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and tools
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualizations
- **Jupyter Notebook** - Interactive development environment

## 📂 Project Structure
```
loan-default-prediction/
├── data/
│   └── raw/
│       └── loan_default.csv           # Original dataset
├── loan_default_prediction.ipynb   # Complete analysis notebook
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 📓 Notebook Workflow

The single notebook `loan_default_prediction.ipynb` contains all phases:

### **Section 1: Data Exploration & Understanding**
- Load the loan dataset
- Examine data structure and types
- Check for missing values and duplicates
- Analyze target variable distribution (default rate)
- Explore feature distributions
- Visualize relationships between features and default
- Statistical summary and correlation analysis

**Key Questions Answered**:
- What percentage of loans default?
- Which features are most correlated with default?
- Are there any data quality issues?

---

### **Section 2: Data Cleaning & Preprocessing**
- Handle missing values (imputation or removal)
- Remove duplicates if any
- Fix incorrect data types
- Handle outliers in numerical features
- Validate data ranges (e.g., CreditScore 300-850)

**Output**: Clean dataset ready for feature engineering

---

### **Section 3: Feature Engineering**
- Encode categorical variables:
  - One-hot encoding for Education, EmploymentType, MaritalStatus, LoanPurpose
  - Binary encoding already done for HasMortgage, HasDependents, HasCoSigner
- Scale numerical features:
  - StandardScaler or MinMaxScaler for consistent ranges
- Create new features (optional):
  - Income to Loan ratio
  - Credit utilization metrics
  - Risk score combinations

**Output**: Transformed feature matrix ready for modeling

---

### **Section 4: Model Building & Training** ⭐ (CORE SECTION)
- Split data into training (80%) and testing (20%) sets
- Train four different classifiers:
  1. Random Forest Classifier
  2. Gaussian Naive Bayes
  3. Logistic Regression
  4. Support Vector Machine (RBF)
- Use consistent random_state for reproducibility

**Output**: Four trained models

---

### **Section 5: Model Evaluation & Comparison**
- Evaluate each model on test data
- Calculate performance metrics:
  - Accuracy
  - Precision (how many predicted defaults were actual defaults)
  - Recall (how many actual defaults were caught)
  - F1-Score (balance of precision and recall)
  - Confusion Matrix
- Compare all four models visually
- Identify best performing model

**Key Visualizations**:
- Confusion matrices for each classifier
- Performance comparison bar charts
- Decision boundary plots (if using 2D visualization)
- ROC curves (optional advanced)

**Output**: Performance comparison report and best model selection

---

### **Section 6: Predictions & Interpretation**
- Use best model to predict on new loan applications
- Test with sample applicants
- Interpret feature importance (for tree-based models)
- Provide business insights and recommendations

**Output**: Predictions for new applicants and actionable insights

## 🚀 Getting Started

### Step 1: Setup Environment
```bash
# Create project directory
mkdir loan-default-prediction
cd loan-default-prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Get Dataset
- Download the loan default dataset
- Place `loan_data.csv` in the `data/raw/` folder

### Step 3: Run Notebook
```bash
# Launch Jupyter Notebook
jupyter notebook

# Open loan_default_prediction.ipynb
# Run all cells sequentially
```

## 📋 Requirements (requirements.txt)
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

## 📈 Expected Results

**Model Performance Goals**:
- **Accuracy**: 75-85%
- **Precision**: 70-80% (minimize false positives)
- **Recall**: 75-85% (catch most defaults)

**Expected Best Performers**:
- Random Forest or SVM typically achieve highest accuracy
- Logistic Regression provides best interpretability

**Key Predictive Features** (anticipated):
- CreditScore (most important)
- DTIRatio (debt burden)
- Income and LoanAmount
- EmploymentType and MonthsEmployed

## 🎯 Project Deliverables

1. **Complete Jupyter Notebook** - All analysis in one file
2. **Performance Comparison** - Side-by-side evaluation of 4 classifiers
3. **Best Model Selection** - Identified optimal algorithm with justification
4. **Visualizations** - Confusion matrices, comparison charts, feature importance
5. **Business Insights** - Actionable recommendations for loan approval process

## 🎓 Skills Demonstrated
- Binary classification problem formulation
- End-to-end machine learning workflow
- Multiple algorithm implementation (Random Forest, Naive Bayes, Logistic Regression, SVM)
- Model evaluation and comparison
- Feature engineering and preprocessing
- Data visualization and interpretation
- Business problem solving with ML

## 💡 Key Insights to Discover

Through this analysis, you will answer:
- Which features are most predictive of loan default?
- What credit score threshold indicates high risk?
- How does debt-to-income ratio affect default probability?
- Which classifier works best for this problem and why?
- What recommendations can we give to the bank?

## 📚 Resources
- [Scikit-learn Classification Guide](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Credit Risk Modeling Best Practices](https://www.kaggle.com/learn/intro-to-machine-learning)

## 📄 License
Educational use - Free to use for academic projects

---

**Project Status**: 🚀 Ready to Start  
**Complexity Level**: Intermediate  
**Estimated Time**: 10-15 hours  
**Perfect For**: Machine Learning classification assignments