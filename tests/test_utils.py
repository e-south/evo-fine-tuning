from fine_tuning.utils import custom_tokenizer

def test_custom_tokenizer():
    sequence = "ACGTGCTA"
    tokens = custom_tokenizer(sequence)
    assert isinstance(tokens, list), "Tokenized output should be a list"
    assert all(isinstance(token, int) for token in tokens), "All tokens should be integers"
