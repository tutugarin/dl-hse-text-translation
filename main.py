# https://pytorch.org/tutorials/beginner/translation_transformer.html

import os
import torch
import dataclasses
import yaml
import sacrebleu

import typing as tp
import torch.nn as nn

from tqdm import tqdm
from timeit import default_timer as timer
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator


 # --- CONFIG ---

@dataclasses.dataclass
class TranslationConfig:

    src_lang: str = 'de'
    tgt_lang: str = 'en'

    src_files: tp.Tuple[str, str] = ('data/train.de-en.de', 'data/val.de-en.de')
    tgt_files: tp.Tuple[str, str] = ('data/train.de-en.en', 'data/val.de-en.en')

    test_file: str = 'data/test1.de-en.de'

    src_vocab_size: int = 8000
    tgt_vocab_size: int = 8000

    src_normalization_rule_name: str = 'nmt_nfkc_cf'
    tgt_normalization_rule_name: str = 'nmt_nfkc_cf'

    src_model_type: str = 'word'
    tgt_model_type: str = 'word'

    src_max_length: int = 96
    tgt_max_length: int = 96

    unk_id: int = 0
    pad_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    batch_size: int = 192

    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    emb_size: int = 512
    nhead: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.1

    num_epochs: int = 15

    @classmethod
    def from_yaml(cls, raw_yaml: tp.Union[str, tp.TextIO]):
        with open(raw_yaml, "rt", encoding="utf8") as stream:
            data = yaml.safe_load(stream)
        data = cls(**data)
        return data


CONFIG = TranslationConfig()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# --- DATASET ---

class MyTokenizer:
    def __init__(
        self, 
        data_file: str, 
        sp_model_prefix: str,
        vocab_size: int,
        normalization_rule_name: str,
        model_type: str, 
        max_length: int
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

    def __getitem__(self, item: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        token_ids = self.indices[item]
        token_ids = torch.cat((
            torch.tensor([CONFIG.bos_id], dtype=torch.int64),
            torch.tensor(token_ids, dtype=torch.int64),
            torch.tensor([CONFIG.eos_id], dtype=torch.int64)
        ))

        padded_tokens = torch.ones(self.max_length, dtype=torch.int64) * CONFIG.pad_id
        padded_tokens[: len(token_ids)] = token_ids

        return padded_tokens


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

    def text2ids(self, texts: tp.Union[str, tp.List[str]]) -> tp.Union[tp.List[int], tp.List[tp.List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        return torch.tensor(self.src_tokenizer.sp_model.encode(texts), dtype=torch.int64)

    def ids2text(self, ids: tp.Union[torch.Tensor, tp.List[int], tp.List[tp.List[int]]]) -> tp.Union[str, tp.List[str]]:
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.tgt_tokenizer.sp_model.decode(ids)


# --- MODEL ---

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* torch.log(torch.tensor(10000)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * torch.sqrt(torch.tensor(self.emb_size))


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int,
                 dropout: float):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor,
                src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == CONFIG.pad_id) # .transpose(0, 1)
    tgt_padding_mask = (tgt == CONFIG.pad_id) # .transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# --- TRAIN ---

def train_epoch(model: Seq2SeqTransformer, optimizer, train_iter: TextDataset):
    model.train()
    losses = 0
    train_dataloader = DataLoader(train_iter, batch_size=CONFIG.batch_size)

    for src, tgt in tqdm(train_dataloader, desc='training'):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        optimizer.zero_grad()
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        loss = nn.CrossEntropyLoss(ignore_index=CONFIG.pad_id)(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        loss.backward()
        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model: Seq2SeqTransformer, val_iter: TextDataset):
    model.eval()
    losses = 0
    val_dataloader = DataLoader(val_iter, batch_size=CONFIG.batch_size)

    for src, tgt in tqdm(val_dataloader, desc='validation'):        
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        loss = nn.CrossEntropyLoss(ignore_index=CONFIG.pad_id)(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


def greedy_decode(model: Seq2SeqTransformer, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(1))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out # .transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == CONFIG.eos_id:
            break
    return ys

def translate(model: torch.nn.Module, src_sentence: str, text2ids, ids2text):
    model.eval()
    src = text2ids(src_sentence).view(1, -1)
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=CONFIG.bos_id).flatten()
    return "".join(ids2text(tgt_tokens)).replace("<bos>", "").replace("<eos>", "")


def main():
    dataset_train = TextDataset(CONFIG, split='train')
    dataset_val = TextDataset(CONFIG, split='val')

    transformer = Seq2SeqTransformer(
    num_encoder_layers=CONFIG.num_encoder_layers,
    num_decoder_layers=CONFIG.num_decoder_layers,
    emb_size=CONFIG.emb_size,
    nhead=CONFIG.nhead,
    src_vocab_size=dataset_train.src_tokenizer.vocab_size,
    tgt_vocab_size=dataset_train.tgt_tokenizer.vocab_size,
    dim_feedforward=CONFIG.dim_feedforward,
    dropout=CONFIG.dropout
    ).to(DEVICE)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)

    for epoch in range(1, CONFIG.num_epochs + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, dataset_train)
        end_time = timer()
        val_loss = evaluate(transformer, dataset_val)

        print(f"""
            ======================================
            Epoch {epoch}: \n
            \t Train loss: {train_loss:.3f}, \n
            \t Val loss: {val_loss:.3f}, \n
            \t Epoch time = {(end_time - start_time):.3f}s, \n\n
        """)

    torch.save(transformer.state_dict(), 'transformer.model')
    torch.save(optimizer.state_dict(), 'optimizer.opt')

    text2ids = dataset_train.text2ids
    ids2text = dataset_train.ids2text
    with open(CONFIG.test_file, 'r') as input:
        with open('solution.txt', 'w') as output:
            for sentence in input.readlines():
                print(translate(transformer, sentence, text2ids, ids2text), file=output)

if __name__ == "__main__":
    main()
