# 🚗 Car Price Prediction with Machine Learning
## Oasis Infobyte Data Science Internship — Task 3

| | |
|---|---|
| **Name** | Sadiya Anmol |
| **Track** | Data Science |
| **Internship** | Oasis Infobyte |
| **Dataset** | CAR DETAILS FROM CAR DEKHO |

---

## 📌 Objective

Build a Machine Learning regression model that predicts 
the selling price of a used car based on features such as 
brand, age, mileage, fuel type, and transmission type.

---

## 📊 Dataset Information

| Property | Value |
|---|---|
| Source | CAR DETAILS FROM CAR DEKHO |
| Total Records | 4,340 |
| Features | 8 columns |
| Target Variable | selling_price |

### Columns
| Column | Description |
|---|---|
| name | Car name (brand + model) |
| year | Year of purchase |
| selling_price | 🎯 Target — price to predict |
| km_driven | Kilometers driven |
| fuel | Petrol/Diesel/CNG/LPG/Electric |
| seller_type | Individual/Dealer/Trustmark |
| transmission | Manual/Automatic |
| owner | First/Second/Third Owner |

---

## 🔧 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| ML Library | Scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter Notebook |

---

## ⚙️ Feature Engineering

Two new features created:
- **Car Age** = 2024 - Year of Purchase
- **Brand** = First word extracted from car name

---

## 🤖 Models Trained

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | High | High | Low |
| Gradient Boosting | Medium | Medium | Good |
| ✅ Random Forest | Lowest | Lowest | Highest |

**Best Model: Random Forest Regressor**

---

## 📈 Key Findings

- Car Age and KM Driven most impact resale price
- Automatic transmission cars sell for more
- Diesel cars have higher resale value
- Luxury brands have significantly higher resale prices
- Residuals are randomly distributed — model is reliable

---

## 📁 Project Structure
DataScience-Task3-CarPricePrediction/
│
├── car_price_prediction.ipynb   # Main Jupyter Notebook
├── CAR_DETAILS_FROM_CAR_DEKHO.csv  # Dataset
└── README.md                    # Project Documentation
---

## ▶️ How to Run

**Step 1 — Clone the repo:**
```bash
git clone https://github.com/sadiya211/OIBSIP.git
```

**Step 2 — Install dependencies:**
```bash
pip install pandas numpy scikit-learn 
matplotlib seaborn jupyter
```

**Step 3 — Open notebook:**
```bash
jupyter notebook car_price_prediction.ipynb
```

**Step 4 — Run all cells:**
Kernel → Restart & Run All

---

## 🎯 Real World Applications

- 🚗 Car dealerships pricing used cars
- 👤 Individual sellers estimating fair price
- 🏦 Banks for vehicle loan assessment
- 📱 Used car apps like CarDekho, OLX

---

*Built with ❤️ by **Sadiya Anmol** | Data Science Intern | Oasis Infobyte*