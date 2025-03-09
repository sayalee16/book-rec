from flask import Flask, render_template,request
import pickle 
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


popular = pickle.load(open('popular.pkl','rb'))
popular = popular.head(50)

df_filled = pickle.load(open('df_filled.pkl','rb'))

df_filled_user = df_filled.T
df_matrix = df_filled_user.to_numpy(dtype=np.float64)  
# Perform SVD
U, sigma, Vt = svds(df_matrix, k=50)
# Convert sigma into a diagonal matrix
sigma = np.diag(sigma)

# Reconstruct the predicted ratings matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
# Convert back to DataFrame with original index & columns
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=df_filled_user.index, columns=df_filled_user.columns)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                            book_name = list(popular['Book-Title'].values),
                            author = list(popular['Book-Author'].values),
                            img = list(popular['Image-URL-S'].values),
                            votes = list(popular['total-count'].values),
                            rating = list(popular['avg-rating'].values)                           
                           )

@app.route('/user-based', methods=['POST','GET'])
def userbased():
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(df_filled_user)

    user_index = int(request.form.get("query_index", 0))
    distances, indices = model_knn.kneighbors(df_filled_user.iloc[user_index, :].values.reshape(1, -1), n_neighbors=6)
    user_id = df_filled_user.index[user_index]

    print(f"Users similar to {user_id}:\n")
    for i in range(1, len(indices.flatten())):  # Skip the first one (self)
        users = df_filled_user.index[indices.flatten()[i]]
        # print(f"{i}: {df_filled_user.index[indices.flatten()[i]]}, Similarity: {1 - distances.flatten()[i]:.4f}")

    # similar users
    similar_users = indices.flatten()[1:]

    # books rated by similar users
    rec_books = df_filled_user.iloc[similar_users].mean(axis=0).sort_values(ascending=False).index

    # print(f"\nTop 5 Recommended books for User {user_id}:\n")
    # for i in recommended_books[:5]:
    #     print(i + "\n")
    return render_template('user-rec.html', rec = rec_books[:5])

@app.route('/item-based', methods=['POST','GET'])
def itembased():
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(df_filled)
    query_index = int(request.form.get("query_index", 0))
    distances, indices = model_knn.kneighbors([df_filled.iloc[query_index]], n_neighbors=6)

    rec = []
    for i in range(1, len(indices.flatten())):
        rec.append(df_filled.index[indices.flatten()[i]])

    return render_template('item-based.html',rec = rec)


@app.route('/model-based', methods=['POST','GET'])
def modelbased():

    user_id = int(request.form.get("query_index"),254)

    user_ratings = predicted_ratings_df.loc[user_id]

    rec = user_ratings.sort_values(ascending=False).index

    return render_template('model-based.html', rec= rec[:5])

if __name__ == '__main__':
    app.run(debug=True)
