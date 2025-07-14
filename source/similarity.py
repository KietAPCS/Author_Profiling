from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

def calculate_similarity(profile1, profile2):
    # Check if profiles are empty or None
    if not profile1 or not profile2:
        return 0.0
    
    # Get all unique n-grams from both profiles
    all_ngrams = set(profile1.keys()).union(set(profile2.keys()))
    
    # If no n-grams found, return 0 similarity
    if not all_ngrams:
        return 0.0
    
    # Create vectors for cosine similarity
    vec1 = []
    vec2 = []
    
    for ngram in all_ngrams:
        vec1.append(profile1.get(ngram, 0))
        vec2.append(profile2.get(ngram, 0))
    
    # Convert to numpy arrays
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    
    # Check if vectors have any non-zero values
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    
    try:
        similarity = cosine_similarity(vec1, vec2)[0][0]
        # Handle potential NaN values
        if np.isnan(similarity):
            return 0.0
        return similarity
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0