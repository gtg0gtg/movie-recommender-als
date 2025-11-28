# 🎬 Movie Recommender System (ALS + MovieLens 100K)

A complete movie recommendation system built using ALS (Alternating Least Squares) from the `implicit` library and the MovieLens 100K dataset. The goal is to demonstrate how real-world recommenders (like Netflix and Amazon) build user–item models using sparse matrices and matrix factorization.

## 📦 Project Structure
```
movie_recommender/
  data/
    ml-100k/
      u.data
      u.item
  src/
    load_data.py
    train_als.py
  requirements.txt
  README.md
```

## 📚 Dataset
MovieLens 100K contains:
- 100,000 ratings  
- 943 users  
- 1,682 movies  

Files used:
- `u.data` → user_id, movie_id, rating, timestamp  
- `u.item` → movie_id, title  

Dataset link: https://grouplens.org/datasets/movielens/

## ⚙️ How It Works
### 1. load_data.py
Loads:
- ratings  
- movie titles  
- merges them into:
  - user_id  
  - movie_id  
  - title  
  - rating  
  - timestamp  

### 2. train_als.py
Builds a **users × items** sparse matrix using:
- `user_id`
- `movie_id`
- `rating`

ALS requires the matrix in **users x items** shape.

Then:
- Trains ALS  
- Computes user & item embeddings  
- Generates top-N movie recommendations  

Example output:
```
Top-5 recommendations for user_id 196:
When Harry Met Sally... (1989)
Sleepless in Seattle (1993)
In & Out (1997)
Emma (1996)
Sabrina (1995)
```

### 3. Train/Test Split
Simulates real-world evaluation:
- Last user interaction → test  
- All previous interactions → train  

## ▶️ Run the Project
From project root:
```
cd src
python train_als.py
```

## 📥 Installation
```
pip install -r requirements.txt
```

## 🚀 Future Work
- Add Hit-Rate@K / Recall@K / MAP@K evaluation  
- Add similar_movies(title)  
- Create REST API / CLI  
- Build Streamlit dashboard  

## 🧑‍💻 Author
Qusai Ayyad  
AI Engineer 
