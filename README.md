# Book Recommendation System

## Overview
This project is a simple implementation of a Book Recommendation System that applies basic collaborative filtering techniques (user-based, item-based, and model-based using SVD) along with a popularity-based recommendation approach. The system utilizes the Book-Crossing dataset.

## Features
- **Collaborative Filtering**
  - User-based filtering
  - Item-based filtering
  - Model-based filtering using SVD
- **Popularity-Based Recommendations**
- **Book-Crossing Dataset** for real-world book rating data
- Considers factors such as the number of ratings, average rating scores, experienced users and the total number of users who have interacted with a book.
- **Flask Backend**
  - Handles user requests and processes recommendation logic

## Dataset
The project uses the [Book-Crossing dataset](https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset)

## Technologies Used
- Python
- Flask (backend)
- Pandas & NumPy
- SciPy (for collaborative filtering & SVD)
- Scikit-learn (for model-based filtering)
