import torch

import typing as tp

from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

from config import TranslationConfig, CONFIG


class MyTokenizer:
    def __init__(
        self, 
        config: TranslationConfig,
        train_data_file: str, 
        val_data_file: str, 
    ):
        """
        Dataset with texts, supporting BPE tokenizer
        :param data_file: txt file containing texts
        :param sp_model_prefix: path prefix to save tokenizer model
        :param vocab_size: sentencepiece tokenizer vocabulary size
        :param normalization_rule_name: sentencepiece tokenizer normalization rule
        :param model_type: sentencepiece class model type
        """
        self.config = config
        self.train_mode = True
        
        train_texts = []
        with open(train_data_file, encoding="utf-8") as file:
            train_texts.extend(file.readlines())
        self.train_texts = train_texts

        val_texts = []
        with open(val_data_file, encoding="utf-8") as file:
            val_texts.extend(file.readlines())
        self.val_texts = train_texts

        self.tokenizer = get_tokenizer(None)

        self.vocab_transform = build_vocab_from_iterator(
            self._yield_tokens(),
            min_freq=1,
            specials=config.special_symbols,
            special_first=True
        )
        self.vocab_transform.set_default_index(self.config.unk_id)

    def _yield_tokens(self) -> tp.List[str]:
        for data_sample in self.train_texts:
            yield self.tokenizer(data_sample)

    def __getitem__(self, item: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        if self.train_mode:
            seq = self.train_texts[item]
        else:
            seq = self.val_texts[item]    

        tokens = self.tokenizer(seq)
        token_ids = self.vocab_transform(tokens)

        token_ids = torch.cat((
            torch.tensor([self.config.bos_id], dtype=torch.int64),
            torch.tensor(token_ids, dtype=torch.int64),
            torch.tensor([self.config.eos_id], dtype=torch.int64)
        ))

        return token_ids
    
    def train(self):
        self.train_mode = True
    
    def eval(self):
        self.train_mode = False


class TextDataset(Dataset):
    def __init__(
        self, 
        config: TranslationConfig,
    ):
        self.config = config
        self.src_tokenizer = MyTokenizer(
            config=config,
            train_data_file=config.src_files[0],
            val_data_file=config.src_files[1],
        )
        self.tgt_tokenizer = MyTokenizer(
            config=config,
            train_data_file=config.tgt_files[0], 
            val_data_file=config.tgt_files[1], 
        )

    def __len__(self):
        if self.src_tokenizer.train_mode:
            return len(self.src_tokenizer.train_texts)
        return len(self.src_tokenizer.val_texts)

    def __getitem__(self, item: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        return self.src_tokenizer[item], self.tgt_tokenizer[item]

    def text2ids(self, texts: tp.Union[str, tp.List[str]]) -> tp.Union[tp.List[int], tp.List[tp.List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        tokens = self.src_tokenizer.tokenizer(texts)
        token_ids = self.src_tokenizer.vocab_transform(tokens)
        token_ids = torch.cat((
            torch.tensor([self.config.bos_id], dtype=torch.int64),
            torch.tensor(token_ids, dtype=torch.int64),
            torch.tensor([self.config.eos_id], dtype=torch.int64)
        ))
        return torch.tensor(token_ids, dtype=torch.int64)

    def ids2text(self, ids: tp.Union[torch.Tensor, tp.List[int], tp.List[tp.List[int]]]) -> tp.Union[str, tp.List[str]]:
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        return self.tgt_tokenizer.vocab_transform.lookup_tokens(ids)

    def train(self):
        self.src_tokenizer.train()
        self.tgt_tokenizer.train()
    
    def eval(self):
        self.src_tokenizer.eval()
        self.tgt_tokenizer.eval()

def collate_fn(batch):
    """
    Add padding to batch samples
    """
    src_batch = [sample[0] for sample in batch]
    tgt_batch = [sample[1] for sample in batch]
    src_batch = pad_sequence(src_batch, padding_value=CONFIG.pad_id)
    tgt_batch = pad_sequence(tgt_batch, padding_value=CONFIG.pad_id)
    return src_batch, tgt_batch
