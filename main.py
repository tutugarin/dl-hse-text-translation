import torch

import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

from config import DEVICE, CONFIG
from dataset import TextDataset, collate_fn
from model import Seq2SeqTransformer, create_mask, generate_square_subsequent_mask


def train_epoch(model: Seq2SeqTransformer, optimizer, dataset: TextDataset):
    model.train()
    dataset.train()

    losses = 0
    train_dataloader = DataLoader(dataset, batch_size=CONFIG.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    for src, tgt in tqdm(train_dataloader, desc='training'):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        loss = torch.nn.CrossEntropyLoss(ignore_index=CONFIG.pad_id)(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model: Seq2SeqTransformer, dataset: TextDataset):
    model.eval()
    dataset.eval()

    losses = 0
    val_dataloader = DataLoader(dataset, batch_size=CONFIG.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        loss = torch.nn.CrossEntropyLoss(ignore_index=CONFIG.pad_id)(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == CONFIG.eos_id:
            break
    return ys


def translate(model: torch.nn.Module, src_sentence: str, text2ids, ids2text):
    model.eval()
    src = text2ids(src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=CONFIG.bos_id).flatten()
    return " ".join(ids2text(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


def main():
    dataset = TextDataset(CONFIG)

    transformer = Seq2SeqTransformer(
        num_encoder_layers=CONFIG.num_encoder_layers,
        num_decoder_layers=CONFIG.num_decoder_layers,
        emb_size=CONFIG.emb_size,
        nhead=CONFIG.nhead,
        src_vocab_size=len(dataset.src_tokenizer.vocab_transform),
        tgt_vocab_size=len(dataset.tgt_tokenizer.vocab_transform),
        dim_feedforward=CONFIG.dim_feedforward,
        dropout=CONFIG.dropout
    ).to(DEVICE)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=CONFIG.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = None

    for epoch in range(1, CONFIG.num_epochs + 1):
        train_loss = train_epoch(transformer, optimizer, dataset)
        val_loss = evaluate(transformer, dataset)

        if scheduler is not None:
            scheduler.step()

        print(f"""
            ======================================
            Epoch {epoch}: \n
            \t Train loss: {train_loss:.3f}, \n
            \t Val loss: {val_loss:.3f}, \n
        """)

    torch.save(transformer.state_dict(), 'transformer.model')
    torch.save(optimizer.state_dict(), 'transformer.opt')

    text2ids = dataset.text2ids
    ids2text = dataset.ids2text

    with open(CONFIG.test_file, 'r', encoding="utf-8") as f_input:
        with open('solution.txt', 'w', encoding="utf-8") as f_output:
            for sentence in f_input.readlines():
                print(translate(transformer, sentence, text2ids, ids2text).strip(" "), file=f_output)


if __name__ == '__main__':
    main()
