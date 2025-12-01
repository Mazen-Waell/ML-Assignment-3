# 📌 Assignment 3: Probabilistic Models & Decision Trees

## 👥 Team Members & Responsibilities

### 🔹 Part A — Gaussian Generative Classifier  
**Assigned to:** **Ahmed Gamal**

**Responsibilities:**
- Load and preprocess the digits dataset  
- Apply stratified 70/15/15 split  
- Standardize features  
- Implement Gaussian Generative Classifier:  
  - Class priors πₖ  
  - Class means μₖ  
  - Shared covariance Σ  
  - Regularized covariance Σₗ = Σ + λI  
- Tune the hyperparameter λ  
- Evaluate final model (accuracy, macro precision, recall, F1, confusion matrix)  
- Write Part A analysis section  

---

### 🔹 Part B — Naive Bayes Classifier  
**Assigned to:** **Mohamed Mostafa**

**Responsibilities:**
- Load and preprocess the Adult Income dataset  
- Use only categorical features  
- Handle missing values as separate categories  
- Encode categories as integers  
- Implement Naive Bayes with Laplace smoothing α  
- Tune α ∈ [0.1, 0.5, 1.0, 2.0, 5.0]  
- Compare with sklearn MultinomialNB  
- Perform feature subset analysis  
- Study predicted probability distributions  
- Write Part B analysis and conclusion  

---

### 🔹 Part C & D — Decision Tree & Random Forest  
**Assigned to:** **Mazen Wael**

#### Part C: Decision Tree (from scratch)
- Load breast cancer dataset  
- Stratified 70/15/15 split  
- Implement decision tree with continuous features:  
  - Entropy impurity  
  - Information gain  
  - Best threshold selection  
- Implement stopping rules:  
  - max_depth  
  - min_samples_split  
  - pure nodes  
- Hyperparameter tuning:  
  - max_depth ∈ {2,4,6,8,10}  
  - min_samples_split ∈ {2,5,10}  
- Evaluate final model (accuracy, precision, recall, F1, confusion matrix)  
- Analyze feature importance & overfitting  

#### Part D: Random Forest (Bonus)
- Implement Random Forest using your own tree  
- Bootstrap sampling  
- Random subset of features per split  
- Tune:
  - T ∈ {5,10,30,50}  
  - max_features ∈ {√d, d/2}  
- Final evaluation on test set  
- Compare with Part C (bias–variance analysis)  

---

## 📁 Project Structure
project/
│── partA/ # Gaussian Generative Model (Ahmed)
│── partB/ # Naive Bayes (Mohamed)
│── partC/ # Decision Tree (Mazen)
│── partD/ # Random Forest (Mazen)
│── data/
│── report.pdf
│── README.md


---

✔️ Summary

This assignment implements and compares:

Gaussian Generative Classifier

Naive Bayes

Decision Tree

Random Forest (bonus)

Each team member is responsible for a full ML pipeline for their assigned part, including preprocessing, model implementation, hyperparameter tuning, and evaluation.






## 📁 Project Structure
