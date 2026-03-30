
# ⚡ SmartCharging Analytics: Uncovering EV Behavior Patterns

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-link-here.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Data Mining](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)]()

An interactive Business Intelligence web application built with **Streamlit**. This project applies Advanced Data Mining techniques to analyze a global dataset of EV charging stations to uncover usage patterns, segment locations, discover feature associations, and detect anomalous station behaviors.

---

## 🎓 **Academic Details**
* **Developer:** NISHTHA-PRIYESH-SHAH
* **WACP No:** 1000436
* **CRS Subject:** Artificial Intelligence
* **Course Name:** Data Mining / IBCP (AI)
* **Institution:** Aspen Nutan Academy

---

## 📌 **Brief Project Title and Scope**
**Project Title:** SmartCharging Analytics: Uncovering EV Behavior Patterns.

**Project Scope:** The objective of this project is to study EV charging stations worldwide. By analyzing historical dataset variables (such as Cost, Charger Type, Distance to City, and Usage Stats), this project uncovers hidden consumer charging trends. The scope includes rigorous data cleaning, exploratory data analysis (EDA), customer/station segmentation using machine learning, cross-referencing station features for combo-planning, and identifying anomalous charging patterns. The ultimate goal is to provide actionable intelligence to city planners and network operators to optimize EV infrastructure.

---

## 🚀 **Live Dashboard**
The interactive data mining dashboard is deployed live on Streamlit Community Cloud. 
👉 **[Click here to view the live app](https://idai105-1000436-nishthapriyeshshah-sa.streamlit.app/)**

*(Note to grader: Please click the link above to view the interactive visualizations, maps, and insights).*

---

## 📋 **Project Scope & Objectives (Stage 1)**
We aim to map the "city landscape" for EV charging to determine which areas operate smoothly, where users go most, and what unusual cases stand out.
* **Goal:** Uncover behavior patterns (when, where, how much), group stations into functional clusters (e.g., daily commuters, long-duration users), and detect faulty readings or abnormal charging patterns.
* **Outcome:** Provide data-driven strategic recommendations for dynamic pricing, network expansion, and infrastructure maintenance.

---

## 🧹 **Key Preprocessing Steps, Visualizations, and Findings**

### **Key Preprocessing Steps**
Before analysis, the raw dataset underwent rigorous tuning to ensure optimal machine learning performance:
1. **Duplicate Removal:** Scanned for and dropped duplicate rows based on unique `Station ID`.
2. **Handling Missing Values:** Filled null values appropriately (e.g., `Reviews (Rating)` with the median, `Renewable Energy Source` with the mode, and `Connector Types` with "Unknown").
3. **Categorical Encoding:** Encoded string features like `Charger Type` (AC Level 1, AC Level 2, DC Fast), `Station Operator`, and `Renewable Energy Source` (Yes/No) into numeric formats.
4. **Feature Engineering:** Converted `Availability` (e.g., "24/7") into a continuous `Availability Hours` column.
5. **Normalization:** Applied `StandardScaler` to continuous variables (`Cost`, `Usage Stats`, `Charging Capacity`, `Distance to City`) to ensure equal weighting during clustering.

### **Visualizations & Findings**
* **Histograms & Line Charts:** Usage stats histograms showed a dense concentration of 20-50 daily users, while line charts mapped against `Installation Year` proved steady historical growth in network demand.
* **Boxplots & Heatmaps:** Boxplots revealed that operators like Tesla maintain tight pricing bands, whereas others vary wildly. Heatmaps confirmed that 24/7 DC Fast chargers generate the highest demand density.
* **Scatter Plots:** Visualizing Cost vs. Usage and Ratings vs. Usage proved that highly rated stations attract higher usage, confirming that maintenance directly drives revenue.

---

## 🧠 **Methodology & Advanced Analytics**

### **🎯 Stage 4: Clustering Analysis (Station Segmentation)**
* **Algorithm:** K-Means Clustering (`scikit-learn`) & PCA dimensionality reduction.
* **Process:** Used the **Elbow Method** to mathematically determine the optimal number of clusters. Stations were grouped based on Usage, Capacity, Cost, and Distance to City.
* **Outcome:** The algorithm successfully segmented stations into distinct profiles (e.g., *High Demand Premium*, *Underutilized Rural*, *Standard Commuter*). These clusters were then plotted interactively on a **Folium Geographic Map** using Latitude and Longitude.

### **🔗 Stage 5: Association Rule Mining**
* **Algorithm:** Apriori Algorithm (`mlxtend`).
* **Process:** Discovered hidden connections between station features and user demand.
* **Outcome:** Generated robust rules evaluated by Support, Confidence, and Lift. Key findings proved prompt hints, such as: *"If DC Fast Charger + Renewable Energy = Yes, then Usage = High Demand"*.

### **⚠️ Stage 6: Anomaly Detection (Outliers)**
* **Method:** Interquartile Range (IQR), Z-Score, and Local Outlier Factor (LOF).
* **Process:** Isolated stations experiencing sudden, massive usage spikes, as well as detecting "Bad Stations" (stations with unusually high costs, terrible ratings, and suspicious maintenance frequencies).
* **Outcome:** Flagged a precise list of anomalous stations requiring immediate audits or physical maintenance.

### **📈 Stage 7: Insights & Reporting**
The project successfully answers the core business questions:
1. **Popular Chargers:** DC Fast Chargers paired with renewable energy attract the highest volume of users.
2. **Best Operators:** Operators maintaining strict pricing and achieving 4.0+ star ratings dominate average daily usage.
3. **Peak Demand Locales:** Urban stations (<5km from city centers) see massively higher demand compared to rural outposts.
4. **Anomalies:** Detected specific stations with erratic profiles (e.g., high cost, low usage, 1-star ratings, and excessive weekly maintenance).

---

## 🖥️ **UI Dashboard**

*(Note: Ensure your screenshots are named 1.png, 2.png, etc., and placed in the `SCREEN SHOT` folder in your repository)*

 ![Dashboard 1](./SCREEN%20SHOT/1.png)  ![Dashboard 2](./SCREEN%20SHOT/2.png)  ![Dashboard 3](./SCREEN%20SHOT/3.png) 

 ![Dashboard 4](./SCREEN%20SHOT/4.png)  ![Dashboard 5](./SCREEN%20SHOT/5.png)  ![Dashboard 6](./SCREEN%20SHOT/6.png) 

 ![Dashboard 7](./SCREEN%20SHOT/7.png)  ![Dashboard 8](./SCREEN%20SHOT/8.png)  ![Dashboard 9](./SCREEN%20SHOT/9.png) 

---

## 📂 **Repository Structure Guide**

```text
IDAI105-1000436-NISHTHA-PRIYESH-SHAH/
│
├── app.py                               # Main Streamlit dashboard and ML pipeline
├── detailed_ev_charging_stations.csv    # The EV dataset used for analysis
├── requirements.txt                     # Python library dependencies
├── SCREEN SHOT/                         # Folder containing UI documentation images
│   ├── 1.png
│   ├── 2.png
│   └── ...
└── README.md                            # Detailed project documentation (This file)
```

---

## 🛠️ **Installation & Local Setup**

To run this project on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/zeneanand/IDAI105-1000442-ZENE-SOPHIE-ANAND.git](https://github.com/zeneanand/IDAI105-1000442-ZENE-SOPHIE-ANAND.git)
   cd IDAI105-1000442-ZENE-SOPHIE-ANAND
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit server:**
   ```bash
   streamlit run app.py
   ```

---

## 📚 **References**
* **Scikit-Learn:** K-Means Clustering, Standard Scaler, Local Outlier Factor. (https://scikit-learn.org/)
* **Mlxtend:** Apriori Algorithm and Association Rules. (http://rasbt.github.io/mlxtend/)
* **Plotly Express:** Interactive Graphing Library. (https://plotly.com/python/plotly-express/)
* **Folium / Streamlit-Folium:** Interactive Geographic Mapping. (https://python-visualization.github.io/folium/)
* **Streamlit:** Web application deployment framework. (https://streamlit.io/)
```

