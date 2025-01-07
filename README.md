# KNN-Classifier-for-Bowler-classification
  
## Project Overview  
The **kNN Bowler Classifier** is a machine learning project that uses the k-Nearest Neighbors (kNN) algorithm to classify bowlers into different categories (e.g., pace, spin, all-rounder) based on their performance metrics. This project demonstrates how machine learning can be applied to sports analytics, providing insights into player classification and performance analysis.

---

## Key Features  
- **Data Preprocessing**: Cleaned and prepared bowler performance data for analysis.  
- **Feature Engineering**: Created relevant features to improve classification accuracy.  
- **kNN Algorithm**: Implemented kNN for classifying bowlers based on similarity.  
- **Model Evaluation**: Measured accuracy, precision, recall, and F1 score to assess performance.  
- **Visualization**: Decision boundaries and performance metrics visualized using Matplotlib and Seaborn.

---

## Dataset  
- **Source**: [Include dataset source or mention it's synthetic if applicable.]  
- **Attributes**:  
  - Bowling speed  
  - Economy rate  
  - Strike rate  
  - Wickets taken  
  - Matches played  
  - Runs conceded  

---

## Tech Stack  
- **Programming Language**: Python  
- **Libraries**:  
  - NumPy  
  - Pandas  
  - Scikit-learn  
  - Matplotlib  
  - Seaborn  

---

## Installation  

### Prerequisites  
- Python 3.x  
- Required libraries (install via `requirements.txt`)  

### Setup  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/knn-bowler-classifier.git
   cd knn-bowler-classifier

---

## Usage
### Run the Preprocessing Script:
 python src/data_preprocessing.py

### Train the Model:
python src/train_knn.py

### Make Predictions:
python src/predict.py --input new_bowler_data.csv

---

## Results
The kNN Bowler Classifier achieved the following performance metrics:
Accuracy: 92%
Precision: 89%
Recall: 90%
F1 Score: 89%

---

## Visualizations
Performance Metrics: Accuracy, Precision, Recall, F1 Score visualized using bar charts.
Decision Boundaries: 2D plots showcasing the classifier's decision regions.
![image](https://github.com/user-attachments/assets/f574c17a-89ad-4633-8ab3-f3c2d7396954)

---

## Conclusion
The kNN Bowler Classifier successfully demonstrates how machine learning can be applied to sports analytics by accurately classifying bowlers based on performance metrics. The project showcases key concepts such as data preprocessing, feature engineering, and the implementation of the k-Nearest Neighbors algorithm.

With strong results and insightful visualizations, this project highlights the potential of data-driven approaches to player analysis. Future improvements, including expanding the dataset and experimenting with advanced models, can further enhance its accuracy and applicability in the domain of sports analytics.
