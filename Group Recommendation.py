import pandas as pd
import numpy as np
import random


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


def find_unrated_movies_for_group(selected_user_group, ratings_matrix):
    """
    This finds the movies that none of the users from the selected_user_group
    have rated.
    :param selected_user_group: List of user-IDs in the group
    :param ratings_matrix: Matrix that contains all the ratings of the users
    :return: List of movies that none of the users from the group have rated
    """
    # Initialize a set with all movies
    all_movies_set = set(ratings_matrix.columns)

    # Iterate over each user in the selected_user_group
    for user_id in selected_user_group:
        # Identify the user's row
        user_row = ratings_matrix.loc[user_id]
        # Remove rated movies from the set
        all_movies_set -= set(user_row.dropna().index)

    # Convert the set to a list
    unrated_movies_list = list(all_movies_set)

    return unrated_movies_list


def select_users_based_on_correlation(user, ratings_matrix):
    """
    Since similar users increase group similarity, here we select similar users
    to the group
    :param user: int, initially given user
    :param ratings_matrix: Matrix that contains all the ratings of the users
    :return: selected 3 users
    """
    # Calculate Pearson correlations with the given user
    correlations_with_user1 = calculate_pearson_correlation(user,
                                                            ratings_matrix)

    # select 2nd user to the group
    user2 = []

    for i, score in correlations_with_user1.items():
        # Check if the user is not the given user and has a similarity score
        # greater than 0.25
        if i != user and score > 0.25:
            user2.append(i)

    # Randomly select one user from the list
    if user2:
        first_selected_user = random.choice(user2)

        # Calculate Pearson correlations with the selected user
        correlations_with_selected_user = calculate_pearson_correlation(
            first_selected_user, ratings_matrix)

        # select 3rd user to the group
        user3 = []

        for i, score in correlations_with_selected_user.items():
            # Check if the user is not the given user, the first selected
            # user, and has a similarity score greater than 0.25
            if i != user and i != first_selected_user and score > 0.25:
                user3.append(i)

        # Randomly select one user from the second-pass list
        if user3:
            second_selected_user = random.choice(
                user3)

            print("selected users in the group:", user, first_selected_user,
                  second_selected_user)
            # Return the selected users
            return [user, first_selected_user, second_selected_user]

    # If no suitable users found, return an empty list
    return []


def top_10_movie_recommendations(rating_dictionary):
    """
    This function prints the top 10 selected movies in a given scenario
    :param rating_dictionary: dict, this contains the group ratings after each
    method
    :return: top 10 movies
    """
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
        sorted(rating_dictionary.items(), key=lambda item: item[1],
               reverse=True))

    top_10_indices = list(
        sorted_predicted_values.keys())[:10]

    for movie_id in top_10_indices:
        movie_name = movie_data.get(movie_id, 'Unknown')
        print(f"{movie_id} :  {movie_name} ")


def average_aggregation(rating_matrix):
    """
    This method use average aggregation method for the group recommendation
    :param rating_matrix: Matrix that contains all the ratings of the users
    :return: results after the average aggregation method
    """
    # Calculate the average rating for each movie
    average_ratings = rating_matrix.mean(axis=0)

    # Create a dictionary with MovieID as keys and Average Rating as values
    average_ratings_dict = average_ratings.to_dict()
    return average_ratings_dict


def least_misery_aggregation(rating_matrix):
    """
    This method uses the least misery method for the group recommendation
    :param rating_matrix: Matrix that contains all the ratings of the users
    :return: results after the least misery method
    """
    # Calculate the minimum rating for each movie
    min_ratings = rating_matrix.min(axis=0)

    # Create a dictionary with MovieID as keys and Min Rating as values
    min_ratings_dict = min_ratings.to_dict()
    return min_ratings_dict


def consensus_function(rating_matrix):
    """
    This method uses a new method that account the agreements and the
    disagreements of the users for the group recommendation
    :param rating_matrix: Matrix that contains all the ratings of the users
    :return: results after the consensus function
    """
    # Calculate the average rating for each movie
    average_ratings = rating_matrix.mean(axis=0)
    # Calculate the variance for each movie
    variance_ratings = rating_matrix.var(axis=0)

    w1 = 0.2
    w2 = 1 - w1
    consensus_method = w1 * average_ratings + w2 * (1 - variance_ratings)
    consensus_function_dict = consensus_method.to_dict()

    return consensus_function_dict


def main():
    user = int(
        input("Insert a user ID (User ID should be between 1 and 943): "))

    data = pd.read_csv('u.data', sep='\t',
                       names=['user_id', 'item_id', 'rating', 'timestamp'])

    data = data.drop('timestamp', axis=1)

    ratings_matrix = data.pivot_table(index=['user_id'], columns=['item_id'],

                                      values='rating')

    selected_user_group = select_users_based_on_correlation(user,
                                                            ratings_matrix)

    # List to store the results
    result_list = []

    # Find movies that none of the users from the selected group have rated
    unrated_movies = find_unrated_movies_for_group(selected_user_group,
                                                   ratings_matrix)

    # Part A
    # Iterate over each user in the selected_user_group
    for selected_user in selected_user_group:

        correlations = calculate_pearson_correlation(selected_user,
                                                     ratings_matrix)
        recommended_movies = {}  # Initialize a dictionary

        for movie_id in unrated_movies:
            # Select the top similar users for each scenario
            top_users = similar_users(selected_user, correlations,
                                      ratings_matrix, movie_id)

            # Predict the movie ratings for the given movie
            predicted_movie_rating = movie_recommendation(selected_user,
                                                          movie_id,
                                                          correlations,
                                                          ratings_matrix,
                                                          top_users)

            recommended_movies[movie_id] = predicted_movie_rating
            # Append the result to the list
            result_list.append(
                (selected_user, movie_id, predicted_movie_rating))

        # Display the top 10 suitable movies with movie names for individual
        # user
        print(
            f"\n10 most suitable movies to user group with "
            f"user {selected_user} are")
        top_10_movie_recommendations(recommended_movies)

    # Create a DataFrame from the result list
    columns = ['User', 'Movie', 'PredictedRating']
    result_df = pd.DataFrame(result_list, columns=columns)
    result_matrix = pd.pivot_table(result_df, values='PredictedRating',
                                   index='User',
                                   columns='Movie')

    # Create a dictionary with MovieID as keys and Average Ratings as values
    average_ratings_dict = average_aggregation(result_matrix)
    print(f"\n10 most suitable movies to user group with\n average aggregation "
          f"method are")
    top_10_movie_recommendations(average_ratings_dict)

    # Create a dictionary with MovieID as keys and Minimum Ratings as values
    min_ratings_dict = least_misery_aggregation(result_matrix)
    print(f"\n10 most suitable movies to user group with\n least misery "
          f"method are")
    top_10_movie_recommendations(min_ratings_dict)

    # Part B

    # Create a dictionary with MovieID as keys and consensus function
    # outputs as values
    consensus_function_dict = consensus_function(result_matrix)
    print(f"\n10 most suitable movies to user group with\n consensus function "
          f"method are")
    top_10_movie_recommendations(consensus_function_dict)


if __name__ == "__main__":
    main()
