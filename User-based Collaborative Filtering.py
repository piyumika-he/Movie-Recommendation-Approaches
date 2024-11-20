import pandas as pd
import numpy as np


def calculate_pearson_correlation(user_id, ratings_matrix):
    """
    calculates the pearson correlation between the selected user and all the
    other users in the dataset
    :param user_id: int, user ID of the selected user
    :param ratings_matrix: matrix that contains all the ratings of the users
    :return: correlation between the given user and the other users
    """
    correlations = {}

    # Extracting the user ratings of the given user
    user_ratings = ratings_matrix.loc[user_id]

    for other_user_id, other_user_ratings in ratings_matrix.iterrows():
        # Find the common set of movies that both users have rated (not NaN)
        common_rated_movies = user_ratings.notna() & other_user_ratings.notna()

        # Select the users who have at least 2 commonly rated movies
        if common_rated_movies.sum() > 2:
            # Extract ratings for commonly rated movies
            user_ratings_common = user_ratings[common_rated_movies]
            other_user_ratings_common = other_user_ratings[common_rated_movies]

            # Check for zero variance
            if np.var(user_ratings_common) == 0 or np.var(
                    other_user_ratings_common) == 0:
                correlation = 0.0
            else:
                # Calculate the Pearson correlation
                correlation = user_ratings_common.corr(
                    other_user_ratings_common)

            correlations[other_user_id] = correlation
        else:
            # If there are less than 2 common rated movies,
            # set the correlation to 0
            correlations[other_user_id] = 0.0

    return correlations


def calculate_NHSM_similarity(user_id, ratings_matrix):
    """
    calculates the NHSM_similarity between the selected user and all the
    other users in the dataset
    :param user_id: int, user ID of the selected user
    :param ratings_matrix: matrix that contains all the ratings of the users
    :return: similarity between the given user and the other users
    """
    similarities = {}

    # Extracting the user ratings of the given user
    user_ratings = ratings_matrix.loc[user_id]
    # Here we assume median value of our ratings is 3, as ratings varies
    # from 1 to 5
    med = 3
    # extracting the mean rating of each movie
    mean_movie_ratings = ratings_matrix.mean(axis=0)

    # Calculating the Improved Heuristic Similarity
    for other_user_id, other_user_ratings in ratings_matrix.iterrows():
        proximity = 1 - (1 / (
                1 + np.exp(-(np.abs(user_ratings - other_user_ratings)))))

        significance = 1 / (1 + np.exp(
            -(np.abs(user_ratings - med) * np.abs(other_user_ratings - med))))

        singularity = 1 - (1 / (1 + np.exp(-np.abs(
            ((user_ratings + other_user_ratings) / 2) - mean_movie_ratings))))

        PSS_similarity = np.sum(proximity * significance * singularity)

        # Remove missing values (NaN) from the sets
        vector1 = {x for x in user_ratings if not np.isnan(x)}
        vector2 = {x for x in other_user_ratings if not np.isnan(x)}

        # Calculate the intersection
        intersection = len(vector1.intersection(vector2))

        # Calculate the union
        union = len(vector1.union(vector2))

        # Calculate the Jaccard similarity coefficient
        jaccard_similarity = intersection / union

        # Calculate user rating preference
        user_rating_preference = 1 - 1 / (1 + np.exp(-(np.abs(
            np.mean(user_ratings) - np.mean(other_user_ratings)) * np.abs(
            np.std(user_ratings) - np.std(other_user_ratings)))))

        JPSS_similarity = PSS_similarity * jaccard_similarity

        NHSM_similarity = JPSS_similarity * user_rating_preference

        similarities[other_user_id] = NHSM_similarity

    return similarities


def similar_users_top10(user_id, user_similarity):
    """
    Selects the top 10 similar users to the provided user ID
    :param user_id: int, user ID of the selected user
    :param user_similarity: double, a dictionary of similarity scores between
    the provided user and the other users
    :return: top 10 similar users
    """
    # Sort the dictionary by values
    sorted_users = sorted(user_similarity.items(), key=lambda x: x[1],
                          reverse=True)

    top10_similar_users = [user for user in sorted_users if
                           user[0] != user_id][:10]
    # Extract only the user IDs from the list
    top10_user_ids = [user[0] for user in top10_similar_users]

    return top10_user_ids


def similar_users(user_id, user_similarity, ratings_matrix, movie_id):
    """
    Selects the top n similar users to the provided user ID
    :param user_id: user-ID of the selected user
    :param user_similarity: double, a dictionary of similarity scores between
    the provided user and the other users
    :param ratings_matrix: matrix that contains all the ratings of the users
    :param movie_id: int: movie ID
    :return: top similar users
    """
    # Sort the dictionary by values
    sorted_users = sorted(user_similarity.items(), key=lambda x: x[1],
                          reverse=True)

    # Exclude the specified user and get the top n similar user IDs who have
    # rated the given movie
    top_similar_user_ids = []
    # Initialize th variable n. Assumption : Top 50 users will give the best
    # results in recommending
    n = 50
    for user in sorted_users:
        other_user_id = user[0]
        if other_user_id != user_id and not np.isnan(
                ratings_matrix.at[other_user_id, movie_id]):
            top_similar_user_ids.append(other_user_id)
        if len(top_similar_user_ids) >= n:
            break

    return top_similar_user_ids


def movie_recommendation(user_id, movie_id, user_similarity, ratings_matrix,
                         top_similar_user_ids):
    """
    Predicts the score for a given movie for the provided user
    :param user_id: user-ID of the selected user
    :param movie_id: int: movie ID
    :param user_similarity: double, a dictionary of similarity scores between
    the provided user and the other users
    :param ratings_matrix: matrix that contains all the ratings of the users
    :param top_similar_user_ids: int: list of top similar users
    :return: prediction
    """

    # Calculate the mean of the selected user
    mean_of_selected_user = ratings_matrix.loc[user_id].mean()

    # Initialize the numerator
    numerator = 0

    for similar_user_id in top_similar_user_ids:
        similarity_score = user_similarity.get(similar_user_id)
        # Extract the user ratings
        user_ratings = ratings_matrix.loc[similar_user_id]
        # Calculate the mean of other users
        user_mean = ratings_matrix.loc[similar_user_id].mean()
        # Update the numerator
        numerator += similarity_score * (user_ratings[movie_id] - user_mean)

    denominator = sum(user_similarity.values())

    # Make the prediction
    prediction = mean_of_selected_user + (numerator / denominator)

    return prediction


def find_unrated_movies(user_id, ratings_matrix):
    """
    This finds the movies that the selected user hasn't rated
    :param user_id: user-ID of the selected user
    :param ratings_matrix: matrix that contains all the ratings of the users
    :return: list of movies that the selected user hasn't rated
    """
    # Identify the user's row
    user_row = ratings_matrix.loc[user_id]

    # Filter columns where the user has not made any ratings
    movies_with_no_ratings = user_row[user_row.isna()].index

    return movies_with_no_ratings


def main():
    user = int(
        input("Insert a user ID (User ID should be between 1 and 943): "))

    # task a

    data = pd.read_csv('u.data', sep='\t',
                       names=['user_id', 'item_id', 'rating', 'timestamp'])

    print(data.head())

    count_of_ratings = len(data)
    print(
        f"\nRatings count of \"MovieLens 100K\" data set: {count_of_ratings}")

    data = data.drop('timestamp', axis=1)

    ratings_matrix = data.pivot_table(index=['user_id'], columns=['item_id'],

                                      values='rating')

    # task b
    # Calculate Pearson correlations between the given user and the other users

    print()
    print("##################################################################")
    print(
        'Predictions after calculating similarities using Pearson correlation')
    print("##################################################################")
    correlations = calculate_pearson_correlation(user, ratings_matrix)

    # task c
    print(f"\n10 most similar users to user {user} are")
    top_10_users = similar_users_top10(user, correlations)
    print(top_10_users)

    # task d
    # Recommending movies to the user
    # First select the movies that the given user hasn't rated
    unrated_movies = find_unrated_movies(user, ratings_matrix)

    recommended_movies = {}  # Initialize a dictionary

    for movie_id in unrated_movies:
        # Select the top similar users for each scenario
        top_users = similar_users(user, correlations, ratings_matrix,
                                  movie_id)
        # Predicts the movie ratings for the given movie
        predicted_movie_rating = movie_recommendation(user, movie_id,
                                                      correlations,
                                                      ratings_matrix,
                                                      top_users)
        recommended_movies[movie_id] = predicted_movie_rating

    # Load movie data
    movie_data = {}
    with open("u.item", 'r', encoding='latin-1') as file:
        for line in file:
            parts = line.strip().split('|')
            movie_id = int(parts[0])
            movie_name = parts[1]
            movie_data[movie_id] = movie_name

    # Display the top 10 suitable movies with movie names
    sorted_predicted_values = dict(
        sorted(recommended_movies.items(), key=lambda item: item[1],
               reverse=True))

    top_10_indices = list(sorted_predicted_values.keys())[:10]

    print(f"\n10 most suitable movies to user {user} are")
    for movie_id in top_10_indices:
        movie_name = movie_data.get(movie_id, 'Unknown')
        print(f"{movie_id} :  {movie_name} ")

    print()
    print("##################################################################")
    print(
        "Predictions after calculating similarities using Improved Heuristic "
        "Similarity")
    print("##################################################################")

    # task e
    similarity = calculate_NHSM_similarity(user, ratings_matrix)

    print(f"\n10 most similar users to user {user} are")
    top_10_users = similar_users_top10(user, similarity)
    print(top_10_users)


if __name__ == "__main__":
    main()
