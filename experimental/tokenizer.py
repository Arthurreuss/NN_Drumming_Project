# the tokenizer should take a vector as input and return a token

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