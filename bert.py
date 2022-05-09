import datasets
import datetime
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from transformers import BertForMaskedLM, BertTokenizerFast, AutoConfig
import tqdm


def collate(config):
    def collate_fn(batch):
        seq_len = max(item["input_ids"].shape[-1] for item in batch)
        batch_size = len(batch)

        input_ids = (
            torch.zeros((batch_size, seq_len)).long().cuda() + config.pad_token_id
        )
        attention_mask = torch.zeros((batch_size, seq_len)).long().cuda()
        for i, item in enumerate(batch):
            n = item["input_ids"].shape[-1]
            input_ids[i, :n] = item["input_ids"]
            attention_mask[i, :n] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return collate_fn


class MyDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        text = item["text"]

        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        input_ids = input_ids[:, :512]
        return {"input_ids": input_ids}


def main():
    batch_size = 2
    mask_p = 0.15
    lr = 1.0e-6
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    # model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    config = AutoConfig.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM(config)

    dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    dataset = dataset.filter(lambda example: bool(example["text"]))
    dataset = MyDataset(dataset, tokenizer)
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate(model.config), shuffle=True
    )

    model = model.cuda()

    optimizer = Adam(model.parameters(), lr=lr)

    delay = datetime.timedelta(seconds=1)
    last = datetime.datetime.now()

    with tqdm.tqdm(loader) as pbar:
        for batch in pbar:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            mask = torch.rand(input_ids.shape, device=torch.device("cuda")) < (
                1 - mask_p
            )
            masked_ids = input_ids.clone() * mask + 255 * ~mask
            masked_ids = attention_mask * masked_ids
            output = model(masked_ids)

            logits = F.log_softmax(output.logits, dim=-1)
            loss = F.nll_loss(
                logits.view(-1, model.config.vocab_size),
                input_ids.view(-1),
                ignore_index=model.config.pad_token_id,
            )
            loss.backward()
            optimizer.step()
            if datetime.datetime.now() - last > delay:
                pbar.set_description(f"Loss {loss.item():.2f}")
                last = datetime.datetime.now()


if __name__ == "__main__":
    main()
