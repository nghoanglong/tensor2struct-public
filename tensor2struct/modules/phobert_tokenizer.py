from transformers import AutoTokenizer

import logging

logger = logging.getLogger("tensor2struct")


class PhoBERTokenizer:
    sp_nlp = None

    def __init__(self, version):
        
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        
        self.cls_token = self.tokenizer._parameters["cls_token"]
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids(self.cls_token)
        self.sep_token = self.tokenizer._parameters["sep_token"]
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        self.pad_token = self.tokenizer._parameters["pad_token"]
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.pad_token)
    
    def _encode(self, input_):
        if isinstance(input_, list) or isinstance(input_, tuple):
            encodes = self.tokenizer.encode(input_)
        else:
            encodes = self.tokenizer.encode(input_)
        return encodes
    
    def tokenize(self, text):
        encodes = self._encode(text)
        tokens = encodes.tokens[1:-1]
        return tokens


    def tokenize_with_orig(self, text):
        """
        Tokenize but return the original chars, this would be helpful for copying operations.
        """
        # TODO: if text is a list, change accordingly how the offset is computed
        assert isinstance(text, str)
        encodes = self._encode(text)
        orig_tokens = [text[i:j] for i,j in encodes.offsets[1:-1]]
        return orig_tokens
