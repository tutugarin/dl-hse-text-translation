import torch
import dataclasses

import typing as tp


@dataclasses.dataclass(init=True)
class TranslationConfig:

    data_dir: str = 'data/'

    special_symbols: tp.List[str] = ('<unk>', '<pad>', '<bos>', '<eos>')

    unk_id: int = 0
    pad_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    batch_size: int = 64

    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    emb_size: int = 512
    nhead: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1

    num_epochs: int = 15
    lr: float = 0.00007

    def __post_init__(self):
        self.src_files: tp.List[str] = [f'{self.data_dir}train.de-en.de', f'{self.data_dir}val.de-en.de']
        self.tgt_files: tp.List[str] = [f'{self.data_dir}train.de-en.en', f'{self.data_dir}val.de-en.en']
        self.test_file: str = f'{self.data_dir}test1.de-en.de'


CONFIG = TranslationConfig()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
