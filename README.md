# 🎬 Movie Recommender System (ALS + MovieLens 100K)

A production-style **movie recommendation system** built using **ALS matrix factorization** from the `implicit` library.  
The project demonstrates how real-world platforms (Netflix, Amazon, YouTube) build recommendation engines using **sparse user–item matrices** and **latent factor models**.

---

## 📦 Project Structure
```
movie_recommender/
│
├── data/
│   └── ml-100k/
│       ├── u.data        # user–movie ratings
│       └── u.item        # movie titles
│
├── src/
│   ├── load_data.py      # loads and merges dataset
│   └── train_als.py      # matrix building, ALS training, recommendations
│
├── requirements.txt
└── README.md
```

---

## 📚 Dataset — MovieLens 100K
A classic benchmark dataset:

- **100,000 ratings**
- **943 users**
- **1,682 movies**

Files used:
- `u.data` → `user_id`, `movie_id`, `rating`, `timestamp`
- `u.item` → `movie_id`, `title`

Dataset: https://grouplens.org/datasets/movielens/

---

## ⚙️ System Workflow

### **1. Data Loading**
`load_data.py`:

- Loads ratings from `u.data`
- Loads movie titles from `u.item`
- Merges everything into a clean DataFrame:

```
user_id | movie_id | title | rating | timestamp
```

---

### **2. Interaction Matrix (Core of the System)**
`train_als.py` converts the DataFrame into a **users × items** sparse matrix.

This matrix is the heart of any recommender system.

- Rows = internal user indices  
- Columns = internal movie indices  
- Values = ratings  

ALS *requires* exactly this shape.

---

### **3. ALS Training (Matrix Factorization)**

We train an **Alternating Least Squares (ALS)** model:

- `factors = 50`
- `regularization = 0.01`
- `iterations = 20`

ALS learns:

- **User embeddings (latent preferences)**
- **Item embeddings (movie characteristics)**

Then, similarity between user–movie vectors is used to generate recommendations.

---

### **4. Recommendation Generation**

Given a real `user_id`, the pipeline:

1. Converts to internal user index  
2. Extracts movies already watched  
3. Runs:

```
model.recommend(user_index, user_items)
```

4. Maps internal indices back to real movie IDs  
5. Retrieves movie titles  

Example output:

```
Top-5 recommendations for user_id 196:
When Harry Met Sally... (1989)
Sleepless in Seattle (1993)
In & Out (1997)
Emma (1996)
Sabrina (1995)
```

---

### **5. Temporal Train/Test Split**
To simulate real-world prediction:

- For each user → the **last** rating goes to test  
- Everything before it goes to train  

This mimics “next movie prediction.”

---

## ▶️ Running the Project

From the project root:

```
cd src
python train_als.py
```

You will see:

- Matrix shape
- ALS training progress
- Top-N recommendations
- Train/test summary

---

## 📥 Installation

```
pip install -r requirements.txt
```

---

## 🚀 Roadmap / Future Improvements

- Add evaluation metrics:
  - HitRate@K
  - Recall@K
  - MAP@K
- Add `similar_movies(title, N)`
- Build a Streamlit dashboard
- Convert into a REST API (FastAPI)
- Deploy a small demo on HuggingFace Spaces

---

## 👤 Author
**Qusai Ayyad**  
AI Engineer 
