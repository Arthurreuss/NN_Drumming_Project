# The tokenizer should take a vector as input and return a token
import numpy as np

class SimpleTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            self.vocab = {}
        else:
            self.vocab = vocab

    def in_vocab(self, vector):
        if vector in self.vocab:
            return True
        return False
    
    def tokenize(self, vector):
        if len(self.vocab) != 0:
            next_token = len(self.vocab) + 1
        else:
            next_token = 1
            
        if not self.in_vocab(vector):
            self.vocab[vector] = next_token
            return next_token
        return self.vocab[vector]
    
    
# Example usage
tokenizer = SimpleTokenizer()
print(tokenizer.tokenize((1,0,0)))  # Should assign a new token
print(tokenizer.tokenize((1,0,0)))
print(tokenizer.tokenize((1,0,1)))  # Should assign a new token


# Another option is to use a quantization approach
class QuantizationTokenizer:
    def __init__(self, num_tokens: int, vector_dim: int):
        if num_tokens <= 0:
            raise ValueError("Number of tokens must be positive.")
        if vector_dim <= 0:
            raise ValueError("Vector dimension must be positive.")
            
        self.num_tokens = num_tokens
        self.vector_dim = vector_dim
        self.codebook = np.zeros((num_tokens, vector_dim))
        self.is_fitted = False

    def _calculate_distances(self, vector: np.ndarray) -> np.ndarray:
        """Calculates the L1 distance from a vector to all codewords in the codebook."""
        return np.sum(np.abs(self.codebook - vector), axis=1)
    
    def fit(self, training_vectors: np.ndarray, iterations: int = 100, learning_rate: float = 0.1):
        """Fits the tokenizer to the training vectors using samples from the data."""
        
        # 1. Initialize the codebook with random samples from the training data
        if training_vectors.shape[0] < self.num_tokens:
            raise ValueError("Number of training vectors must be at least equal to the number of tokens.")
        initial_indices = np.random.choice(training_vectors.shape[0], self.num_tokens, replace=False)
        self.codebook = training_vectors[initial_indices].copy().astype(np.float32)

        # 2. Iteratively update the codebook
        for i in range(iterations):
            
            # Shuffle the training data for each iteration
            np.random.shuffle(training_vectors)
            total_distance = 0
            
            for vector in training_vectors:
                # Find the best matching unit (BMU) - the closest codeword
                distances = self._calculate_distances(vector)
                bmu_index = np.argmin(distances)
                total_distance += distances[bmu_index]

                # Update the BMU to be closer to the input vector
                self.codebook[bmu_index] += learning_rate * (vector - self.codebook[bmu_index])
                
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{iterations}, Average Distance: {total_distance / len(training_vectors):.4f}")
                
        self.is_fitted = True
        print("Training complete.")
        
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encodes a set of vectors into a sequence of tokens."""
        if not self.is_fitted:
            raise RuntimeError("The encoder must be fitted with .fit() before encoding.")
        
        tokens = []
        for vector in vectors:
            # Find the index of the closest codeword for each vector
            distances = self._calculate_distances(vector)
            token = np.argmin(distances)
            tokens.append(token)
            
        return np.array(tokens)
        
    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """Decodes a sequence of tokens into vectors."""
        if not self.is_fitted:
            raise RuntimeError("The encoder must be fitted with .fit() before decoding.")
        
        # Simply use the tokens as indices to retrieve vectors from the codebook
        return self.codebook[tokens]
        

# Example usage
tokenizer = QuantizationTokenizer(num_tokens=4, vector_dim=3)
training_data = np.random.rand(100, 3)  # 100 random 3D vectors
tokenizer.fit(training_data, iterations=50, learning_rate=0.2)
encoded = tokenizer.encode(training_data[:5])
print("Encoded tokens:", encoded)
print("Decoded vectors:", tokenizer.decode(encoded))