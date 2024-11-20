import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


def dataset_dividing(data_file, instance):
    """
    This function divides the entire dataset into 3 parts. First the dataset is
    sorted according to the date (time stamp). Then data set is divided into 2
    same size datasets. The second data set is again divided into 2 same size
    datasets

    :param data_file: the file that contains the data
    :param instance: int, represents the dataset that should be sent to the
    main function
    :return: Datasets (1 - Initial dataset, 2 - concatenation of initial
    dataset and data1, 3 - concatenation of initial dataset, data1 and data2
    """
    # Loading the datafile
    data = pd.read_csv(data_file, sep='\t',
                       names=['user_id', 'item_id', 'rating', 'timestamp'])

    # Converting timestamp to date & time format
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s', utc=True)

    data['date'] = data['timestamp'].dt.strftime('%Y-%m-%d')
    data['time'] = data['timestamp'].dt.strftime('%H:%M:%S')

    # Sort the DataFrame by the 'date' column in ascending order
    data_sorted = data.sort_values(by='date')
    data_sorted = data_sorted.drop(['timestamp', 'date', 'time'], axis=1)

    # Divide the DataFrame into two halves
    data_initial = data_sorted.iloc[:50000]

    data_second_half = data_sorted.iloc[50000:]

    # Divide the second half into another two equal sets
    data1 = data_second_half.iloc[:len(data_second_half) // 2]

    data2 = data_second_half.iloc[(len(data_second_half) // 2):]

    if instance == 1:
        data_initial = data_initial.pivot_table(index=['user_id'],
                                                columns=['item_id'],
                                                values='rating')

        return data_initial

    elif instance == 2:
        data_time_window1 = pd.concat([data_initial, data1], ignore_index=True)
        data_time_window1 = data_time_window1.pivot_table(index=['user_id'],
                                                          columns=['item_id'],
                                                          values='rating')
        return data_time_window1

    elif instance == 3:
        data_time_window2 = pd.concat([data_initial, data1, data2],
                                      ignore_index=True)
        data_time_window2 = data_time_window2.pivot_table(index=['user_id'],
                                                          columns=['item_id'],
                                                          values='rating')
        return data_time_window2

    else:
        print("Error")


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


def top_10_movies(rating_dictionary):
    """
    This function returns the top 10 selected movies in a given scenario
    :param rating_dictionary: dict, this contains the group ratings after each
        method
    :return: top 10 movies as a dictionary
    """
    # Display the top 10 suitable movies with movie names
    sorted_predicted_values = dict(
        sorted(rating_dictionary.items(), key=lambda item: item[1],
               reverse=True))

    top_10_movies_dict = {index: sorted_predicted_values[index] for index in
                          list(sorted_predicted_values.keys())[:10]}

    return top_10_movies_dict


def print_top_10_movie_recommendations(rating_dictionary):
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


def group_aggregation_function(rating_matrix, alpha):
    """
    This method uses a new method that account the agreements and the
    disagreements of the users for the group recommendation
    :param alpha: float, this gives a weightage to the aggregation function
    :param rating_matrix: Matrix that contains all the ratings of the users
    :return: results after the aggregation function
    """
    # Calculate the average rating for each movie
    average_ratings = rating_matrix.mean(axis=0)

    # Calculate the standard deviation for each movie
    std_ratings = rating_matrix.std(axis=0)

    group_aggregation = (1 - alpha) * average_ratings + alpha * std_ratings
    group_aggregation_dict = group_aggregation.to_dict()

    return group_aggregation_dict


def select_initial_user(data_file):
    """
    This function use to select an initial user to the group of 3 users.
    It should be given as an input to the programme
    :param data_file: the file that contains the data
    :return: the selected user ID
    """
    data = pd.read_csv(data_file, sep='\t',
                       names=['user_id', 'item_id', 'rating', 'timestamp'])

    # Converting timestamp to data & time format
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s', utc=True)

    data['date'] = data['timestamp'].dt.strftime('%Y-%m-%d')
    data['time'] = data['timestamp'].dt.strftime('%H:%M:%S')

    # Sort the DataFrame by the 'date' column in ascending order
    data_sorted = data.sort_values(by='date')

    # Divide the DataFrame into two halves
    data_initial = data_sorted.iloc[:50000]

    data_initial = data_initial.drop(['timestamp', 'date', 'time'],
                                     axis=1)
    # Assuming 'user' is the column containing user information
    unique_users = data_initial['user_id'].unique()

    # Sort unique users in ascending order
    sorted_users = sorted(unique_users)

    print(f"user ids: {sorted_users}")
    while True:
        print("Select a user from the above user list")
        user_input = input("Enter an user ID: ")

        try:
            user_number = int(user_input)

            if user_number in sorted_users:
                print(f"You selected user {user_number}")
                return user_number
            else:
                print(
                    "The inserted user ID is not in the list. Please try "
                    "again.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def select_users_based_on_correlation(user, ratings_matrix):
    """
    T0 check the behaviour of the whole algorithm when there are similar users
    and dissimilar users in the group, Here we select the 3 users in a way
    where one user is similar to the initially selected user and the second
    user is dissimilar to the selected user.
    :param user: int, initially given user
    :param ratings_matrix: Matrix that contains all the ratings of the users
    :return: selected 3 users with their correlation values
    """
    # Calculate Pearson correlations with the given user
    correlations_with_user1 = calculate_pearson_correlation(user,
                                                            ratings_matrix)

    # Select 2nd user to the group with positive correlation
    positive_correlation_users = [(i, score) for i, score in
                                  correlations_with_user1.items() if
                                  i != user and score > 0.25]

    if positive_correlation_users:
        first_selected_user, first_user_correlation = random.choice(
            positive_correlation_users)

        # Calculate Pearson correlations with the selected user
        correlations_with_selected_user = calculate_pearson_correlation(
            first_selected_user, ratings_matrix)

        # Select 3rd user to the group with negative correlation
        negative_correlation_users = [(i, score) for i, score in
                                      correlations_with_selected_user.items()
                                      if i != user and i != first_selected_user
                                      and score < -0.25]

        if negative_correlation_users:
            second_selected_user, second_user_correlation = random.choice(
                negative_correlation_users)

            print("Selected users in the group:")
            print(
                f"User {user} with user {first_selected_user} (Positive "
                f"Correlation: {first_user_correlation})")
            print(
                f"User {user} with user {second_selected_user} (Negative "
                f"Correlation: {second_user_correlation})")

            # Return the selected users with their correlation values
            return [user, first_selected_user, second_selected_user]

    # If no suitable users found, return an empty list
    return []


def calculate_user_satisfaction(top10_movies_individual, top10_movies_group):
    """
    Calculates the user satisfaction
    :param top10_movies_individual: Top 10 recommended movies to the given user
    :param top10_movies_group: Top 10 recommended movies to the group
    :return: user satisfaction
    """
    group_list_satisfaction_user = 0
    user_top_10_movies = top10_movies_individual
    user_list_satisfaction = sum(user_top_10_movies.values())
    # Loop through indices in user_top_10_movies
    for index in user_top_10_movies:
        # Check if the index is present in group_recommendation
        if index in top10_movies_group:
            # Add the rating to the sum
            group_list_satisfaction_user += user_top_10_movies[index]

    user_satisfaction = group_list_satisfaction_user / user_list_satisfaction

    return user_satisfaction


def main():
    # Obtain the initial user as an input to the programme
    initial_user = select_initial_user('u.data')

    initial_ratings_matrix = dataset_dividing('u.data', 1)

    # Select 2 another users to the group based on their correlations.
    # One with a positive correlation with the initial user and the other with
    # a negative correlation to the initial user
    selected_users = select_users_based_on_correlation(initial_user,
                                                       initial_ratings_matrix)

    print(f"Users selected to the group {selected_users}")

    alpha = 0  # Initialize the alpha value

    # Initialize empty lists to store satisfaction values for each user
    satisfaction_user1_list = []
    satisfaction_user2_list = []
    satisfaction_user3_list = []

    # Initialize empty lists to store alpha values in each iteration
    alpha_list = [0]

    # Initialize empty lists to store group satisfaction values in each
    # iteration
    group_satisfaction_list = []

    for i in range(1, 4):
        # List to store the results
        result_list = []

        # Dictionary to store the top 10 movie recommendations for each user
        user_top_10_dict = {}

        ratings_matrix = dataset_dividing('u.data', i)

        # Find movies that none of the users from the selected group have rated
        unrated_movies = find_unrated_movies_for_group(selected_users,
                                                       ratings_matrix)

        # Iterate over each user in the selected_user_group
        for selected_user in selected_users:

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

            # Store the top 10 movie recommendations for the current user
            user_top_10_dict[selected_user] = top_10_movies(recommended_movies)

        # Create a DataFrame from the result list
        columns = ['User', 'Movie', 'PredictedRating']
        result_df = pd.DataFrame(result_list, columns=columns)
        result_matrix = pd.pivot_table(result_df, values='PredictedRating',
                                       index='User',
                                       columns='Movie')

        group_aggregation_dict = group_aggregation_function(result_matrix,
                                                            alpha)

        group_recommendation = top_10_movies(group_aggregation_dict)

        print(
            f"\n10 most suitable movies to user group with aggregated function"
            f" method in iteration {i} are")
        print_top_10_movie_recommendations(group_aggregation_dict)

        # Calculate the satisfaction scores of each user
        satisfaction_user1 = calculate_user_satisfaction(
            user_top_10_dict[selected_users[0]], group_recommendation)
        satisfaction_user1_list.append(satisfaction_user1)

        satisfaction_user2 = calculate_user_satisfaction(
            user_top_10_dict[selected_users[1]], group_recommendation)
        satisfaction_user2_list.append(satisfaction_user2)

        satisfaction_user3 = calculate_user_satisfaction(
            user_top_10_dict[selected_users[2]], group_recommendation)
        satisfaction_user3_list.append(satisfaction_user3)

        # Calculate the group satisfaction score
        group_satisfaction = (satisfaction_user1 + satisfaction_user2 +
                              satisfaction_user3) / 3
        group_satisfaction_list.append(group_satisfaction)
        print(f"Group satisfaction in iteration {i} are {group_satisfaction}")

        # Calculates the alpha value
        alpha = max(satisfaction_user1, satisfaction_user2,
                    satisfaction_user3) - min(satisfaction_user1,
                                              satisfaction_user2,
                                              satisfaction_user3)
        alpha_list.append(alpha)
        print(f"Alpha value in iteration {i} are {alpha}")

    # Number of iterations
    num_iterations = len(satisfaction_user1_list)

    # Generate positions
    positions = np.arange(num_iterations)

    # Plot the grouped bar chart
    plt.bar(positions - 0.2, satisfaction_user1_list, width=0.2,
            label='User 1')
    plt.bar(positions, satisfaction_user2_list, width=0.2, label='User 2')
    plt.bar(positions + 0.2, satisfaction_user3_list, width=0.2,
            label='User 3')

    plt.xlabel('Iteration')
    plt.ylabel('Satisfaction Value')
    plt.title('Satisfaction Values for Each User in Each Iteration')

    plt.legend()

    plt.show()

    # # Plot the line chart to visualize alpha values
    positions = np.arange(len(alpha_list) - 1)

    plt.plot(positions, alpha_list[0:3], marker='o', linestyle='-', color='b',
             label='Alpha Variation')

    plt.xlabel('Data Points')
    plt.ylabel('Alpha Values')
    plt.title('Variation of Alpha List')

    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
