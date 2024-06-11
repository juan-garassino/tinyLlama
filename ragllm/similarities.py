# import torch

# def dot_product(vector1, vector2):
#     return torch.dot(vector1, vector2)

# def cosine_similarity(vector1, vector2):
#     dot_product = torch.dot(vector1, vector2)

#     # Get Euclidean/L2 norm of each vector (removes the magnitude, keeps direction)
#     norm_vector1 = torch.sqrt(torch.sum(vector1**2))
#     norm_vector2 = torch.sqrt(torch.sum(vector2**2))

#     return dot_product / (norm_vector1 * norm_vector2)