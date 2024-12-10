
# **Iris Clustering Dashboard**

A Streamlit-based web application for visualizing and analyzing clustering techniques on the Iris dataset. This project uses **K-Means** and **Hierarchical Clustering** to group iris flowers based on their features and evaluates clustering performance using metrics like the Adjusted Rand Index (ARI).

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Technologies Used](#technologies-used)
6. [File Structure](#file-structure)
7. [Screenshots](#screenshots)
8. [Future Enhancements](#future-enhancements)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)

---

## **Project Overview**

The Iris dataset is a classic dataset in machine learning and statistics, consisting of measurements of iris flowers of three species. This project applies clustering techniques to group data points and provides an interactive dashboard to visualize the results.

---

## **Features**

- **Dataset Overview**:
  - View the dataset and detect potential outliers.

- **Clustering Methods**:
  - Perform **K-Means Clustering** and **Hierarchical Clustering**.
  - Visualize results in 2D using PCA.

- **Evaluation Metrics**:
  - Compute Adjusted Rand Index (ARI) and Silhouette score to assess clustering performance.
  - View confusion matrices for clustering vs. true labels.

- **Interactive Dashboard**:
  - Explore data insights and clustering results in a user-friendly interface.

---

## **Installation**

Follow these steps to set up the project on your local machine:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Johnnysnipes90/Clustering-Iris-dataset.git
   cd Clustering-Iris-dataset
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   soucrce\venv\Scripts\activate   # For Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

---

## **Usage**

1. Start the Streamlit server by running:
   ```bash
   streamlit run app.py
   ```
2. Open the application in your browser (default: [http://localhost:8501](http://localhost:8501)).
3. Navigate through the sidebar to:
   - View the dataset and outliers.
   - Analyze clustering results.
   - Evaluate clustering performance.

---

## **Technologies Used**

- **Programming Language**: Python
- **Libraries**:
  - Data Analysis: `pandas`, `numpy`
  - Clustering: `scikit-learn`, `scipy`
  - Visualization: `matplotlib`, `seaborn`
  - Web App: `streamlit`

---

## **File Structure**

```
Iris-Clustering-Dashboard/
├── data/                 # Folder for data files (if needed)
├── src/                  # Folder for Python scripts (modular code)
|── screenshots           # Folder for Screenshots
├── app.py                # Streamlit app entry point
├── requirements.txt      # Python dependencies
├── README.md             # Project description
└── .gitignore            # Files to ignore in version control
```

---

## **Screenshots**

### **1. Data Overview**
![Data Overview Screenshot](path/to/your/screenshot1.png)

### **2. PCA Visualization**
![PCA Visualization Screenshot](Clustering-Iris-dataset\screenshots\pca.png)

### **3. Confusion Matrix**
![Confusion Matrix Screenshot](Clustering-Iris-dataset\screenshots\confusion_matrix.png)
> Replace `path/to/your/screenshot` with actual image paths in your repository.

---

## **Future Enhancements**

- Add support for more clustering algorithms (e.g., DBSCAN, Gaussian Mixture Models).
- Integrate additional evaluation metrics like Silhouette Score.
- Allow user uploads for custom datasets.
- Deploy the app using Streamlit Cloud or another hosting platform.

---

## **License**

This project is licensed under the [MIT License](LICENSE).

---

## **Acknowledgments**

- Dataset sourced from the classic **Iris Dataset** by Ronald A. Fisher.
- Libraries and frameworks used in this project:
  - [Streamlit](https://streamlit.io/)
  - [Scikit-learn](https://scikit-learn.org/)
  - [Seaborn](https://seaborn.pydata.org/)

---

### **Contributing**

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

---
