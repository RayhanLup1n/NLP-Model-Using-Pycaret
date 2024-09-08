# Emotion Classification Using NLP

## Project Overview
This project aims to build a machine learning model to classify emotions from text data using natural language processing (NLP) techniques. The goal is to create a model that can accurately classify user comments or text into one of three emotional categories: **anger**, **fear**, or **joy**.

Through this project, I seek to explore the power of text classification in detecting emotions and to see how well common machine learning algorithms perform on this task. The project involves preprocessing text data, creating and training several classification models, and evaluating their performance.

## Objective
The main objective of this project is to build a reliable emotion classification model that can:
- Detect and classify emotions in textual data accurately.
- Compare the performance of various machine learning models and select the best-performing model.
- Provide insights into the text data by visualizing word clouds and emotion distributions.

## Dataset Overview
The dataset used in this project is sourced from Kaggle and contains 5937 entries. Each entry consists of a text comment and the corresponding emotion label (either "anger", "fear", or "joy"). 

- **Number of rows**: 5937
- **Number of unique emotions**: 3 (anger, fear, joy)
- **No missing values or duplicate entries**.

You can find the dataset [here](https://www.kaggle.com/datasets/abdallahwagih/emotion-dataset/data).

## Steps Followed in the Project
1. **Data Loading and Preprocessing**:
   - Loaded the dataset using \`pandas\`.
   - Checked for missing values and duplicates.
   - Visualized the distribution of emotions using bar charts.
   - Generated word clouds for each emotion category to gain insights into the most frequently used words.

2. **Text Preprocessing**:
   - Text preprocessing was handled by PyCaret, using **TF-IDF** for text feature extraction.
   - Converted the text comments into numeric features suitable for machine learning models.

3. **Model Building**:
   - Utilized **PyCaret's classification module** to quickly compare several machine learning models.
   - Evaluated multiple models, including Decision Trees, XGBoost, SVM, and more.
   - The best model based on accuracy was **Decision Tree Classifier**, with an accuracy of **93.94%**.

4. **Model Tuning**:
   - Attempted hyperparameter tuning for the best-performing model, but tuning results showed lower performance than the original model, so the original model was selected.

5. **Evaluation**:
   - Generated a confusion matrix to evaluate the performance of the selected model in classifying the different emotion categories.
   - Achieved the following metrics with the Decision Tree Classifier:
     - **Accuracy**: 93.94%
     - **Precision**: 93.98%
     - **Recall**: 93.94%
     - **F1 Score**: 93.94%

## Project Results
The final model, a **Decision Tree Classifier**, achieved an accuracy of **93.94%**, making it a reliable model for classifying emotions in text data. The confusion matrix shows that the model performs well across all three emotion categories, with minimal misclassification between them.

## Tools and Libraries Used
The following tools and libraries were used to build and evaluate the emotion classification model:
- **Python**: The programming language used for this project.
- **Pandas**: For loading and preprocessing the dataset.
- **Seaborn and Matplotlib**: For data visualization.
- **WordCloud**: For generating word clouds of the most frequently used words in each emotion category.
- **PyCaret**: For comparing, selecting, and tuning machine learning models quickly.
- **Sklearn**: For confusion matrix and other evaluation metrics.

## Conclusion
The project successfully demonstrates how text classification techniques can be applied to detect emotions in textual data. The Decision Tree Classifier performed exceptionally well, achieving high accuracy and demonstrating that simple models can still be very effective in solving NLP problems.

## Acknowledgements
I would like to thank **Abdallah Wagih** for uploading this dataset on Kaggle. You can find the dataset [here](https://www.kaggle.com/datasets/abdallahwagih/emotion-dataset/data). Without this dataset, this project would not have been possible.

## Future Work
While this model performs well, there is always room for improvement. Some potential next steps include:
- Experimenting with different feature extraction techniques such as **Word2Vec** or **BERT**.
- Trying more advanced models like **neural networks** or **transformers**.
- Fine-tuning the model further using different hyperparameter optimization techniques.

Feel free to clone this repository and experiment with the model yourself. If you have any suggestions or improvements, feel free to contribute!
