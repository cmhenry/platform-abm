import numpy as np


# Generate random user-term matrix data as binary preferences
num_users = 100
num_terms = 100
user_term_matrix = np.random.randint(2, size=(num_users, num_terms))
print("User-Term Matrix:")
print(user_term_matrix)

# Perform Singular Value Decomposition (SVD) on the user-term matrix
U, S, Vt = np.linalg.svd(user_term_matrix, full_matrices=False)

# Reconstruct the original matrix from the SVD components
# S_diag = np.zeros((num_users, num_terms))
# S_diag[:num_users, :num_users] = np.diag(S)
# reconstructed_matrix = np.dot(U, np.dot(S_diag, Vt))
# print("\nReconstructed Matrix:")
# print(np.round(reconstructed_matrix).astype(int))

# Us = np.dot(U, np.diag(S))
# skV = np.dot(np.diag(S),Vt)
# UsV = np.dot(Us, skV)

# Select three users from the SVD
selected_users_indices = [0, 2, 4]
selected_users_U = U[selected_users_indices, :]
selected_users_preferences = user_term_matrix[selected_users_indices, :]

# Calculate the group-specific latent factors
group_specific_latent_factors = np.dot(selected_users_preferences.T, selected_users_U)

# Modify U, S, and Vt matrices to obtain group-specific latent factors
group_U = np.copy(U)
group_U[selected_users_indices, :] = group_specific_latent_factors.T

# The group-specific S matrix can be obtained by simply taking the diagonal matrix of the group-specific singular values
group_S = np.diag(S)

# The group-specific Vt matrix remains the same since it represents the term-feature matrix, which is independent of user preferences

# Reconstruct the user-term matrix for the group using the modified U, S, and Vt matrices
group_reconstructed_matrix = np.dot(selected_users_U, np.dot(group_S, Vt))

print("\nGroup-Specific Reconstructed Matrix:")
print(np.round(group_reconstructed_matrix).astype(int))

def recommend_items_to_new_user(new_user_preferences, selected_users_preferences, U, Vt, num_recommendations=5):
    # Calculate the group-specific latent factors
    # group_specific_latent_factors = np.linalg.pinv(selected_users_preferences).dot(U)

    # Calculate the new user's latent factors
    new_user_latent_factors = new_user_preferences.dot(group_specific_latent_factors)

    # Obtain the predicted user-item interaction
    predicted_interaction = new_user_latent_factors.dot(Vt)

    # Rank the items based on predicted interaction and recommend top N items
    recommended_item_indices = np.argsort(predicted_interaction)[::-1][:num_recommendations]

    return recommended_item_indices

# Example usage
# Assume you have U, Vt, and selected_users_preferences from previous SVD calculations
num_items = 100

# Simulate a new user's preferences (binary array of length num_items)
new_user_preferences = np.random.randint(2, size=num_items)

# Recommend top 3 items to the new user
recommended_items = recommend_items_to_new_user(new_user_preferences, selected_users_preferences, U, Vt, num_recommendations=3)

print("Recommended Item Indices:", recommended_item_indices)


