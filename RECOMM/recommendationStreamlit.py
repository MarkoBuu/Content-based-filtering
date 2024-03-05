import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


df = pd.read_csv("steamDESC.csv")

def preprocess_data(df):
    def create_score(row):
        pos_count = row['positive_ratings']
        neg_count = row['negative_ratings']
        total_count = pos_count + neg_count
        average = pos_count / total_count
        return round(average, 2)

    df['score'] = df.apply(create_score, axis=1)

    df['combined_features'] = df['steamspy_tags'] + ' ' + df['developer']

    df['combined_features'] = df['combined_features'].str.replace(' ', '-')  
    df['combined_features'] = df['combined_features'].fillna('')  

    df.dropna(inplace=True)

    return df

def matching_score(a,b):
   return fuzz.ratio(a,b)

def get_title_from_index(index):
   return df[df.index == index]['name'].values[0]

def find_closest_title(title):

   leven_scores = list(enumerate(df['name'].apply(matching_score, b=title)))
   sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)
   closest_title = get_title_from_index(sorted_leven_scores[0][0])
   distance_score = sorted_leven_scores[0][1]
   return closest_title, distance_score


def create_model(df):
    vectorizer_combined = TfidfVectorizer(min_df=2, max_df=0.1)
    vectorized_combined = vectorizer_combined.fit_transform(df['combined_features'])

    tfidf_df = pd.DataFrame(vectorized_combined.toarray(), columns=vectorizer_combined.get_feature_names_out())

    tfidf_df.index = df['name']


    cosine_similarity_array = cosine_similarity(tfidf_df)
    cosine_similarity_df = pd.DataFrame(cosine_similarity_array, index=tfidf_df.index, columns=tfidf_df.index)

    return cosine_similarity_df

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def recommend_games(cosine_similarity_df, title, keyword, min_year=None, min_score=None):
    closest_title, distance_score = find_closest_title(title)

    st.write('Recommended Games:\n')

    list_of_games_enjoyed = [closest_title]
    games_enjoyed_df = cosine_similarity_df.loc[~cosine_similarity_df.index.duplicated()].reindex(list_of_games_enjoyed)

    user_prof = games_enjoyed_df.mean()

    # linija koda dodana da igrica samu sebe ne recommenda
    tfidf_subset_df = cosine_similarity_df.drop([closest_title], axis=0)

    similarity_array = cosine_similarity(user_prof.values.reshape(1, -1), tfidf_subset_df)
    similarity_df = pd.DataFrame(similarity_array.T, index=tfidf_subset_df.index, columns=["similarity_score"])

    sorted_similarity_df = similarity_df.sort_values(by="similarity_score", ascending=False)

    recommended_games_info = []

    rank = 1

    for n in sorted_similarity_df.index:
        if rank <= 5:
            if keyword.lower() in n.lower():  #provjera za postojanje keyworda
                release_date_str = df.loc[df['name'] == n, 'release_date'].values[0]
                release_year = None
                if release_date_str:
                    try:
                        release_year = int(release_date_str.split('/')[-1]) 
                    except ValueError:
                        pass
                game_score = df.loc[df['name'] == n, 'score'].values[0] * 100

                if (min_year is None or (release_year is not None and release_year >= min_year)) and (min_score is None or (game_score is not None and game_score >= float(min_score))):
                    recommended_games_info.append((n, sorted_similarity_df.loc[n, "similarity_score"]))

                    st.write("#" + str(rank) + ": " + n + ", " + str(round(sorted_similarity_df.loc[n, "similarity_score"]*100, 2)) + "% match")
                    st.write("    Short Description:", df.loc[df['name'] == n, 'short_description'].values[0])
                    st.write("    Developer:", df.loc[df['name'] == n, 'developer'].values[0])
                    st.write("    Price:", df.loc[df['name'] == n, 'price'].values[0])
                    platforms = ', '.join(df.loc[df['name'] == n, 'platforms'].values)
                    st.write("    Platforms:", platforms)
                    genres = ', '.join(df.loc[df['name'] == n, 'genres'].values)
                    st.write("    Genres:", genres)
                    st.write("    Game rating:", game_score)
                    st.write("    Release date:", release_date_str)

                    rank += 1
            else:
                continue

    return recommended_games_info




def main():
    df_processed = preprocess_data(df)

    cosine_similarity_df = create_model(df_processed)

    save_model(cosine_similarity_df, 'cosine_similarity_model.pkl')
    st.write("Model saved successfully.")

    loaded_model = load_model('cosine_similarity_model.pkl')
    st.write("Model loaded successfully.")

    title = st.text_input("Enter the game title:")
    keyword = st.text_input("Enter the keyword (optional):")
    min_year = st.number_input("Enter the minimum year of release (optional, 0 if not specified):", value=0)
    min_score = st.number_input("Enter the minimum game score (optional, 0 if not specified):")

    if st.button("Recommend Games"):
        recommended_games_info = recommend_games(loaded_model, title, keyword, min_year, min_score)

if __name__ == "__main__":
    main()
