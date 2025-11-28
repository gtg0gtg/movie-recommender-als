import os
import pandas as pd

DATA_DIR = "../data/ml-100k"

def load_ratings():

    ratings_path = os.path.join(DATA_DIR, "u.data")
    cols = ["user_id","movie_id","rating","timestamp"]
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=cols,
        engine="python",

    )
    return ratings

def load_movies():

    movies_path = os.path.join(DATA_DIR, "u.item")
    movies = pd.read_csv(
        movies_path,
        sep="|",
        header=None,
        usecols=[0,1],
        names=["movie_id","title"],
        encoding="latin-1"
    )
    return movies

def load_merged():

    ratings = load_ratings()
    movies = load_movies()
    df = ratings.merge(movies, on="movie_id", how= "left")

    df = df[["user_id","movie_id","title","rating","timestamp"]]
    return df

if __name__ == "__main__":
    df = load_merged()

    print(df.head())
    print("\nShape:", df.shape)
    print("\nNumber of users:",df["user_id"].nunique())
    print("Number of movies:", df["movie_id"].nunique())

