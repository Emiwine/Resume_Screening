# Resume Screening using ML & NLP 

## Project Overview
This project implements an automated resume screening system using Machine Learning and Natural Language Processing (NLP). It processes resumes, extracts key information, and classifies candidates based on predefined categories. The model can help recruiters filter out relevant resumes efficiently.

## Features
- **Automated Resume Classification**: Uses an ML model to classify resumes into different categories.
- **Text Processing**: Utilizes NLP techniques like TF-IDF vectorization.
- **Model Training & Prediction**: Uses a trained Random Forest classifier.
- **Web Application**: A simple web-based interface to upload and classify resumes.

## Technologies Used
- Python
- Flask (for the web app)
- Scikit-learn (for ML models)
- Pandas & NumPy (for data processing)
- NLTK / SpaCy (for NLP processing)

## Project Structure
```
├── app.py                  # Main application script
├── encoder.pkl             # Encoded label data
├── rfclf.pkl               # Trained Random Forest classifier
├── tfidf.pkl               # TF-IDF vectorizer model
├── output1.png             # Sample output visualization
├── output2.png             # Sample output visualization
├── requirement.txt         # Required dependencies
├── Resume Screening.ipynb  # Jupyter notebook with model training steps
├── UpdatedResumeDataSet.csv # Dataset used for training
└── Readme.md               # Project documentation
```

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirement.txt
   ```
3. Run the Flask application:
   ```sh
   python app.py
   ```
4. Open the web application in a browser at:
   ```sh
   http://127.0.0.1:5000
   ```

## How It Works
1. The user uploads a resume through the web application.
2. The resume text is preprocessed using TF-IDF vectorization.
3. The trained Random Forest model classifies the resume into a relevant category.
4. The result is displayed on the web interface.

## Dataset
The project uses `UpdatedResumeDataSet.csv`, which contains resumes along with their respective categories. The data is preprocessed to remove unnecessary text and format it for model training.

## Model Training
- The text data is vectorized using **TF-IDF**.
- A **Random Forest Classifier** is trained using the processed features.
- The trained model (`rfclf.pkl`) is used for predictions in the web app.

## Future Improvements
- Enhance classification accuracy with deep learning models (e.g., BERT, LSTMs).
- Improve resume parsing using advanced NLP techniques.
- Extend the web app with additional features like job matching.


## License
This project is open-source and available under the MIT License.

