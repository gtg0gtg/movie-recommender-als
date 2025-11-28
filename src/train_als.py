
from scipy.sparse import coo_matrix
import implicit

from load_data import load_merged, load_movies

def build_interaction_matrix(df):

    user_cats = df["user_id"].astype("category")
    item_cats = df["movie_id"].astype("category")

    user_idx = user_cats.cat.codes
    item_idx = item_cats.cat.codes

    df = df.copy()
    df["user_idx"] = user_idx
    df["item_idx"] = item_idx

    num_users = user_idx.nunique()
    num_items = item_idx.nunique()

    data = df["rating"].astype(float).values

    rows = df["user_idx"].values
    cols = df["item_idx"].values

    interactions = coo_matrix(
        (data, (rows, cols)),
        shape=(num_users, num_items)
    ).tocsr()

    idx_to_user_id = dict(enumerate(user_cats.cat.categories))
    idx_to_movie_id = dict(enumerate(item_cats.cat.categories))

    print(df[["user_id","movie_id","user_idx","item_idx","rating"]].head())
    return interactions, idx_to_user_id, idx_to_movie_id

def train_als(interactions, factors=50, regularization=0.01, iterations=20):

    model = implicit.als.AlternatingLeastSquares(
        factors= factors,
        regularization=regularization,
        iterations=iterations,
    )

    model.fit(interactions)

    return model

def recommend_for_user(user_id, model, interactions,
                       idx_to_movie_id, movie_id_to_title,
                       user_id_to_idx, N=5):

    if user_id not in user_id_to_idx:
        print(f"user_id={user_id} not found in data")
        return

    user_index = user_id_to_idx[user_id]

    user_items_all = interactions.tocsr()

    user_items = user_items_all[user_index]

    item_indices, scores = model.recommend(
        user_index,
        user_items,
        N=N,
        filter_already_liked_items=True

    )

    print(f"\nTop-{N} recommendations for user_id {user_id}:")
    for item_idx, score in zip(item_indices, scores):
        movie_id = idx_to_movie_id[item_idx]
        title = movie_id_to_title.get(movie_id, "UNKNOWN_TITLE")
        print(f"movie_id={movie_id}, title={title}, score={score:.3f}")




def train_test_split_per_user(df):

    df_sorted = df.sort_values(by=["user_id", "timestamp"])
    last_row = df_sorted.groupby("user_id").tail(1)
    test_idx = last_row.index


    df_test = df.loc[test_idx]
    df_train = df.drop(test_idx)

    #print("df_train.shape:", df_train.shape)
    #print("df_test.shape:", df_test.shape)
    #print(df_train.head())
    return df_train, df_test



if __name__ == "__main__":

    df = load_merged()
    df_small = df[["user_id","movie_id","rating"]]

    interactions, idx_to_user_id, idx_to_movie_id = build_interaction_matrix(df_small)

    user_id_to_idx = {uid: idx for idx, uid in idx_to_user_id.items()}

    print("interactions shape (user x items):", interactions.shape)

    model = train_als(interactions)


    print("training finished")

    movies_df = load_movies()
    movie_id_to_title = dict(zip(movies_df["movie_id"], movies_df["title"]))


    recommend_for_user(
        user_id=196,
        model=model,
        interactions=interactions,
        idx_to_movie_id=idx_to_movie_id,
        movie_id_to_title=movie_id_to_title,
        user_id_to_idx=user_id_to_idx,
        N=5
    )

    df_train, df_test = train_test_split_per_user(df)
    print(f"\ndf_train.shape: {df_train.shape}\n df_test.shape: {df_test.shape}")
    print(f"\n{df_train.head()}")
    print(f"\n{df_test.head()}")

    print("nNum users:", df["user_id"].nunique())
    print("num users in df_train:", df_train["user_id"].nunique())
    print("num users in df_test:", df_test["user_id"].nunique())
