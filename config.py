import torch

import dataclasses
import yaml
import typing as tp


@dataclasses.dataclass
class TranslationConfig:

    src_lang: str = 'de'
    tgt_lang: str = 'en'

    src_files: tp.Tuple[str, str] = ('data/train.de-en.de', 'data/val.de-en.de')
    tgt_files: tp.Tuple[str, str] = ('data/train.de-en.en', 'data/val.de-en.en')

    src_vocab_size: int = 4000
    tgt_vocab_size: int = 4000

    src_normalization_rule_name: str = 'nmt_nfkc_cf'
    tgt_normalization_rule_name: str = 'nmt_nfkc_cf'

    src_model_type: str = 'bpe'
    tgt_model_type: str = 'bpe'

    src_max_length: int = 1024
    tgt_max_length: int = 1024

    unk_id: int = 0
    pad_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    batch_size: int = 64

    @classmethod
    def from_yaml(cls, raw_yaml: tp.Union[str, tp.TextIO]):
        with open(raw_yaml, "rt", encoding="utf8") as stream:
            data = yaml.safe_load(stream)
        data = cls(**data)
        return data


CONFIG = TranslationConfig.from_yaml('config.yaml')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
