# Customer Segmentation and Targeted Marketing for Telecom Operator

## Project Overview
This project implements advanced customer segmentation analysis for a telecom operator using machine learning techniques. By analyzing customer behavior, usage patterns, and demographics, we identify distinct customer segments to enable targeted marketing strategies and improve customer service.

## Key Features
--  Advanced customer segmentation using clustering algorithms
--  Telecom specific feature engineering
--  Comprehensive data preprocessing pipeline
--  Detailed cluster profiling and interpretation
--  Interactive visualizations of customer segments

## Project Structure
```
├── Data/                    # Data directory (not tracked in git)
│   ├── cleaned_data.csv    # Preprocessed dataset
│   ├── cleaned_data2.csv   # Final processed dataset
│   └── data_telecom.csv    # Raw telecom data
├── models/                  # Trained models
├── Notebooks/              # Jupyter notebooks
│   ├── EDA_and_Preprocessing.ipynb    # Data exploration & cleaning
│   ├── Clustering_analysis.ipynb      # Segmentation analysis
│   └── Cluster_interpretation.ipynb   # Results interpretation
├── src/                    # Source code
│   ├── clustering.py       # Clustering algorithms
│   ├── data_loader.py      # Data loading utilities
│   ├── evaluation.py       # Model evaluation metrics
│   ├── feature_engineering.py  # Feature processing
│   ├── preprocessing.py    # Data preprocessing
│   └── visualization.py    # Visualization utilities
└── visualizations/         # Generated plots and figures
```

## Analysis Pipeline
1. **Data Preprocessing** (`EDA_and_Preprocessing.ipynb`)
   - Data cleaning and validation
   - Feature selection and engineering
   - Handling missing values
   - Exploratory data analysis

2. **Customer Segmentation** (`Clustering_analysis.ipynb`)
   - Feature scaling and transformation
   - Optimal cluster number determination
   - Implementation of clustering algorithms
   - Model evaluation and validation

3. **Segment Analysis** (`Cluster_interpretation.ipynb`)
   - Detailed cluster profiling
   - Customer behavior analysis
   - Segment visualization
   - Marketing recommendations

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages:
  ```
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Abhinavt7/Customer-Segmentation-and-Targeted-Marketing-for-Telecom-Operator-.git
   cd Customer-Segmentation-and-Targeted-Marketing-for-Telecom-Operator-
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Data Setup:
   - Place your telecom dataset in the `Data/` directory
   - The main dataset should be named `data_telecom.csv`

### Usage
1. Run the notebooks in sequence:
   - 01- `EDA_and_Preprocessing.ipynb`
   - 02- `Clustering_analysis.ipynb`
   - 03- `Cluster_interpretation.ipynb`

2. Generated visualizations will be saved in the `visualizations/` directory

## Results
The analysis identifies distinct customer segments based on:
- Usage patterns (voice, data)
- Revenue metrics
- Device characteristics
- Network engagement

Each segment is profiled with:
- Key characteristics
- Behavioral patterns
- Revenue potential
- Marketing opportunities


- Project Link: [https://github.com/Abhinavt7/Customer-Segmentation-and-Targeted-Marketing-for-Telecom-Operator-](https://github.com/Abhinavt7/Customer-Segmentation-and-Targeted-Marketing-for-Telecom-Operator-)