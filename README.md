Fake Job Posting Detection
Overview
This project explores classifying job advertisements as genuine or
fraudulent using machine learning and deep learning models.
Models
The notebook includes experiments with: - TF-IDF + XGBoost - BiLSTM -
BiGRU - TextCNN + BiGRU - Hybrid XGBoost + BiLSTM
Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- Matplotlib and Seaborn
Project File
- Fake_Job_Posting_Detection.ipynb --- notebook containing
  preprocessing, model experiments, and evaluation.
Dataset
The notebook expects a CSV file named fake_job_postings.csv. Update
the dataset path in the notebook to match where you store the file. Make
sure you have permission to use and redistribute the dataset before
publishing it.
How to Run
1. Open the notebook in Google Colab or Jupyter Notebook.
2. Obtain the dataset and update the path in the notebook.
3. Install the required libraries.
4. Run the notebook cells in order to train and evaluate the models.
Install the main dependencies with:
pip install pandas numpy scikit-learn tensorflow xgboost matplotlib seaborn nltk
Evaluation
The notebook includes model evaluation and comparison. Re-run the
notebook to generate results for your environment and data split.
Note
This is an educational project. Model performance depends on the
dataset, preprocessing, and evaluation setup; predictions should not be
treated as definitive proof that a job posting is fraudulent.
