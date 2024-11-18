
# Toxic Comment Classification Project

This project aims to build a machine learning pipeline to classify comments as toxic or non-toxic using logistic regression and natural language processing techniques.

## Files in the Repository

1. **Project.ipynb**:  
   A Jupyter Notebook containing the full workflow for the toxic comment classification task, including:
   - Data preprocessing (removal of stopwords, tokenization, stemming).
   - Text vectorization using TF-IDF.
   - Model training with logistic regression.
   - Evaluation metrics such as precision, recall, and precision-recall curve.
   - Hyperparameter tuning using GridSearchCV.

2. **labeled 2.csv**:  
   A dataset with labeled comments:
   - `comment`: The text of the comment.
   - `toxic`: Binary label (1.0 for toxic comments, 0.0 for non-toxic).

## How to Use

1. **Install Dependencies**:  
   Ensure the following Python libraries are installed:
   ```bash
   pip install pandas scikit-learn nltk matplotlib
   ```

2. **Prepare the Data**:  
   The dataset `labeled 2.csv` is used for training and testing. It contains labeled examples for binary classification.

3. **Run the Notebook**:  
   Open the `Project.ipynb` file in Jupyter Notebook and execute the cells step-by-step to:
   - Process the data.
   - Train the machine learning model.
   - Evaluate performance.

## Key Techniques Used

- **Natural Language Processing (NLP)**:
  - Tokenization with NLTK's `word_tokenize`.
  - Stopword removal using `nltk.corpus.stopwords`.
  - Stemming with `SnowballStemmer`.
- **Text Vectorization**:
  - TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical format.
- **Model Training**:
  - Logistic Regression for binary classification.
  - Hyperparameter tuning with GridSearchCV.
- **Evaluation Metrics**:
  - Precision, Recall, and Precision-Recall Curve.

## Results

The trained model provides insights into the classification performance. Precision and recall curves allow the user to identify the best threshold for making predictions.

## Future Improvements

- Experiment with advanced models such as Random Forests, Gradient Boosting, or Neural Networks.
- Expand the dataset for better generalization.
- Implement additional preprocessing steps like lemmatization or advanced stopword filtering.

## Requirements

- Python 3.7 or above
- Libraries: Pandas, Scikit-learn, NLTK, Matplotlib, NumPy

---

Feel free to contribute or raise issues for further improvements!
