"""Compatibility import for the original AutoTCR tokenizer filename."""
from autotcr.tokenizer import BertTokenizer, WordpieceTokenizer, load_vocab, whitespace_tokenize
__all__ = ["BertTokenizer", "WordpieceTokenizer", "load_vocab", "whitespace_tokenize"]
