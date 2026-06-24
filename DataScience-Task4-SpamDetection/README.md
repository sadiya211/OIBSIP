# 📧 Email Spam Detection with Machine Learning
## Oasis Infobyte Data Science Internship — Task 4

| | |
|---|---|
| **Name** | Sadiya Anmol |
| **Track** | Data Science |
| **Internship** | Oasis Infobyte |
| **Dataset** | SMS Spam Collection (Kaggle) |

---

## 📌 Objective

Build an end-to-end NLP-based Machine Learning system 
that classifies emails and SMS messages as **Spam** or **Ham (Legitimate)**, 
and deploy it as an interactive web application.

---

## 📊 Dataset Information

| Property | Value |
|---|---|
| Source | SMS Spam Collection — Kaggle |
| Total Messages | 5,572 |
| Spam Messages | 747 (~13%) |
| Ham Messages | 4,825 (~87%) |
| Features | Message text only |

---

## 🔧 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| ML Library | Scikit-learn |
| NLP | NLTK, TF-IDF Vectorizer |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Web App | Streamlit |
| Notebook | Jupyter Notebook |

---

## 🤖 Models Trained & Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| ✅ Multinomial Naive Bayes | ~98% | ~97% | ~94% | ~95% |
| Logistic Regression | ~97% | ~96% | ~92% | ~94% |
| Support Vector Machine | ~98% | ~97% | ~93% | ~95% |

**Best Model: Multinomial Naive Bayes**

---

## 🔄 Text Preprocessing Pipeline
Raw Text

↓

Lowercase Conversion

↓

Punctuation Removal

↓

Stopword Removal

↓

Porter Stemming

↓

TF-IDF Vectorization (3000 features)

↓

ML Model → Spam / Ham

---

## 💡 Key Findings

- **Naive Bayes** works best for text classification tasks
- **TF-IDF** captures spam-specific word patterns effectively
- **Recall** is the most critical metric — missing spam is worse than over-filtering
- Common spam words: FREE, WIN, URGENT, CLAIM, PRIZE, CLICK
- Common ham words: call, come, home, later, tomorrow, meeting

---

## 🌐 SpamShield — Web App Features

Built a fully functional **Streamlit** web application:

| Feature | Description |
|---|---|
| 🔑 Login System | Username & password authentication |
| ✨ Sign Up | New user registration with password strength indicator |
| 🔍 Spam Checker | Real-time spam/ham prediction |
| 📊 Probability Score | Shows spam & ham probability % |
| 💡 Example Messages | Pre-loaded spam & ham examples to try |
| 🚪 Logout | Secure session management |

### Run the App:
```bash
python -m streamlit run app.py
```

---

## 📁 Project Structure
DataScience-Task4-SpamDetection/

│

├── spam_detection.ipynb    # Main Jupyter Notebook

├── app.py                  # Streamlit Web App

├── spam.csv                # Dataset

├── README.md               # Project Documentation

│

└── screenshots/

├── distribution.png     # Spam vs Ham distribution

├── confusion_matrix.png # Model confusion matrix

├── model_comparison.png # All models comparison

├── wordcloud.png        # Spam & Ham word clouds

├── app_login.png        # SpamShield login page

└── app_checker.png      # SpamShield checker page

---

## ▶️ How to Run Notebook

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/sadiya211/OIBSIP.git
```

**Step 2 — Install dependencies:**
```bash
pip install pandas numpy scikit-learn nltk 
matplotlib seaborn wordcloud streamlit
```

**Step 3 — Download NLTK data:**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

**Step 4 — Open notebook:**
```bash
jupyter notebook spam_detection.ipynb
```

**Step 5 — Run all cells:**
Kernel → Restart & Run All

---

## 📸 Screenshots

| Notebook Outputs | App Screenshots |
|---|---|
| Class Distribution Chart | Login Page |
| Confusion Matrix | Spam Checker |
| Model Comparison Chart | Result Display |
| WordCloud (Spam vs Ham) | Example Messages |

---

## 🎯 Real World Applications

This model can be deployed in:
- 📧 Email clients (Gmail, Outlook)
- 📱 SMS filtering systems
- 💬 Comment moderation systems
- 🤖 Chat application spam filters

---

*Built with ❤️ by **Sadiya Anmol** | Data Science Intern | Oasis Infobyte*