import os
import torch
import typing as tp

from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator

from config import TranslationConfig, CONFIG


class MyTokenizer:
    def __init__(
        self, 
        data_file: str, 
        sp_model_prefix: str = None,
        vocab_size: int = 2000,
        normalization_rule_name: str = 'nmt_nfkc_cf',
        model_type: str = 'bpe', 
        max_length: int = 1024
    ):
        """
        Dataset with texts, supporting BPE tokenizer
        :param data_file: txt file containing texts
        :param sp_model_prefix: path prefix to save tokenizer model
        :param vocab_size: sentencepiece tokenizer vocabulary size
        :param normalization_rule_name: sentencepiece tokenizer normalization rule
        :param model_type: sentencepiece class model type
        :param max_length: maximal length of text in tokens
        """
        if not os.path.isfile(sp_model_prefix + '.model'):
            SentencePieceTrainer.train(
                input=data_file, vocab_size=vocab_size,
                model_type=model_type, model_prefix=sp_model_prefix,
                normalization_rule_name=normalization_rule_name, 
                unk_id=CONFIG.unk_id,
                pad_id=CONFIG.pad_id,
                bos_id=CONFIG.bos_id,
                eos_id=CONFIG.eos_id,
            )
        self.sp_model = SentencePieceProcessor(model_file=sp_model_prefix + '.model')

        with open(data_file) as file:
            texts = file.readlines()

        self.texts = texts
        self.indices = self.sp_model.encode(self.texts)

        self.vocab_size = self.sp_model.vocab_size()
        self.max_length = max_length
        self.sp_model.pad_id(), self.sp_model.unk_id(),
        self.sp_model.bos_id(), self.sp_model.eos_id()


        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        # self.vocab_transform = build_vocab_from_iterator(
        #     self.yield_tokens(),
        #     min_freq=1,
        #     specials=special_symbols,
        #     special_first=True
        # )

    def yield_tokens(self) -> tp.List[str]:
        for data_sample in self.indices:
            yield data_sample

    @staticmethod
    def tensor_transform(token_ids: tp.List[int]):
        return torch.cat((torch.tensor([CONFIG.bos_id]),
                        torch.tensor(token_ids),
                        torch.tensor([CONFIG.eos_id])))

    def __getitem__(self, item: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        token_ids = self.indices[item]
        token_ids = self.vocab_transform(token_ids)
        token_ids = MyTokenizer.tensor_transform(token_ids)

        padded_tokens = torch.ones(1, self.max_length) * CONFIG.pad_id
        padded_tokens[: token_ids] = token_ids

        return padded_tokens

    def text2ids(self, texts: tp.Union[str, tp.List[str]]) -> tp.Union[tp.List[int], tp.List[tp.List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        return self.sp_model.encode(texts)

    def ids2text(self, ids: tp.Union[torch.Tensor, tp.List[int], tp.List[tp.List[int]]]) -> tp.Union[str, tp.List[str]]:
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.sp_model.decode(ids)


class TextDataset(Dataset):
    def __init__(
        self, 
        config: TranslationConfig,
        *,
        split
    ):
        split = 0 if split == 'train' else 1
        self.src_tokenizer = MyTokenizer(
            config.src_files[split], 
            config.src_lang, 
            config.src_vocab_size, 
            config.src_normalization_rule_name, 
            config.src_model_type, 
            config.src_max_length
        )
        self.tgt_tokenizer = MyTokenizer(
            config.tgt_files[split], 
            config.tgt_lang, 
            config.tgt_vocab_size, 
            config.tgt_normalization_rule_name, 
            config.tgt_model_type, 
            config.tgt_max_length
        )

    def __len__(self):
        return len(self.src_tokenizer.texts)

    def __getitem__(self, item: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        return self.src_tokenizer[item], self.tgt_tokenizer[item]
