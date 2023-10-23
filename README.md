# Text_Classification-Word2Vec

# Word Embeddings and Toxic Comments Classification

This Python script demonstrates the use of word embeddings, specifically GloVe (Global Vectors for Word Representation), and logistic regression for classifying toxic comments. It utilizes the `gensim` library for working with pre-trained word embeddings and `scikit-learn` for machine learning functionalities.

## Prerequisites

Before running the script, ensure you have the following libraries installed:

- `numpy`
- `matplotlib`
- `pandas`
- `spacy`
- `gensim`
- `scikit-learn`

You can install these libraries using the following command:

```bash
pip install numpy matplotlib pandas spacy gensim scikit-learn
```

## Usage

1. **Download Pre-trained Word Vectors:**
   The script uses the GloVe Twitter word vectors. If you haven't downloaded them yet, you can do so using the following line in the script:
   ```python
   wv = api.load('glove-twitter-50')
   ```

2. **Load and Preprocess Data:**
   The script assumes the presence of a CSV file named "toxic_comments_500.csv" containing comment texts and corresponding toxicity labels. Ensure this file is in the same directory as the script. The script loads the data and preprocesses comment texts using spaCy for tokenization and word embeddings.

3. **Train and Evaluate Classifier:**
   The script trains a logistic regression classifier using the preprocessed word embeddings and evaluates its performance on the test data. Accuracy, precision, and recall scores are printed out to assess the classifier's effectiveness.

4. **Visualize Word Embeddings (Optional):**
   The script also includes code to visualize word embeddings using PCA. This step is optional and can be modified or omitted as per your requirements.

5. **Run the Script:**
   Execute the script in your Python environment.
   ```bash
   python toxic_comments_classifier.py
   ```

## Customization

- **Word Embeddings:**
  The script uses pre-trained word vectors from GloVe. You can experiment with different word embeddings by replacing the GloVe vectors with other embeddings available in the `gensim.downloader` module.

- **Text Preprocessing:**
  The `spacy_tokenizer` function can be customized for text preprocessing based on your specific requirements. You can add more stopwords or perform additional text cleaning operations in this function.

- **Classifier and Hyperparameters:**
  The script uses logistic regression as the classification algorithm. If you want to experiment with different classifiers or adjust hyperparameters, you can modify the `classifier` variable and its initialization parameters.
