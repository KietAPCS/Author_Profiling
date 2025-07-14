# Author Profiling Analytics

## Overview

Author Profiling Analytics is a Streamlit-based web application designed to analyze and compare text samples to determine stylistic similarities between authors. The application leverages natural language processing (NLP) techniques to preprocess text, count n-grams, and calculate similarity scores using cosine similarity.

## Features

- **Text Preprocessing**: Tokenization, stemming, and lemmatization of input text.
- **N-gram Profiling**: Generate n-gram profiles for text samples.
- **Similarity Analysis**: Calculate similarity scores between two text samples using cosine similarity.
- **Interactive UI**: A user-friendly interface built with Streamlit for seamless interaction.

## Project Structure

```
AuthorProfiling/
├── app/
│   └── streamlit.py          # Main Streamlit application
├── source/
│   ├── count_word.py         # N-gram counting logic
│   ├── preprocess.py         # Text preprocessing functions
│   ├── similarity.py         # Similarity calculation logic
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/KietAPCS/Author_Profiling.git
   cd Author_Profiling
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app/streamlit.py
   ```
2. Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

### How to Use

- Enter the original text and test text in the provided text areas.
- Adjust the n-gram size if needed (default is 2).
- Click the "Analyze" button to view the similarity score and other metrics.

## Key Modules

### `preprocess.py`

- **Functions**:
  - `stem_lema(text)`: Preprocesses text by tokenizing, stemming, and lemmatizing.

### `count_word.py`

- **Functions**:
  - `count_word(text_tokens, n)`: Generates n-gram profiles from tokenized text.

### `similarity.py`

- **Functions**:
  - `calculate_similarity(profile1, profile2)`: Computes cosine similarity between two n-gram profiles.

## Dependencies

The project uses the following Python libraries:

- `streamlit`
- `nltk`
- `scikit-learn`
- `numpy`
- `pandas`

For a full list, see `requirements.txt`.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push the branch.
4. Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Inspired by the need for advanced text analysis tools.
- Built with love using Python and Streamlit.
