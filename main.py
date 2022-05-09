from typing import Optional, Tuple
import datasets
import datetime
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
import tqdm


def collate(config):
    def collate_fn(batch):
        seq_len = max(
            max(item["input_ids"].shape[-1] for item in batch), config.kernel_size[0]
        )
        batch_size = len(batch)

        input_ids = torch.zeros((batch_size, seq_len)).long().cuda()
        attention_mask = torch.zeros((batch_size, seq_len)).long().cuda()
        for i, item in enumerate(batch):
            n = item["input_ids"].shape[-1]
            input_ids[i, :n] = item["input_ids"]
            attention_mask[i, :n] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return collate_fn


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(256, config.char_dim)
        self.projection = nn.Conv2d(
            1,
            config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            bias=False,
        )

    def forward(self, input_ids):
        out = self.embedding(input_ids)
        # Create fake color channel
        out = out.unsqueeze(1)
        out = self.projection(out)
        out = out.permute((0, 2, 1, 3)).squeeze(-1)
        return out


class Decoder(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.embedding = encoder.embedding
        self.projection = nn.ConvTranspose2d(
            config.out_channels,
            1,
            kernel_size=config.kernel_size,
            stride=config.stride,
            bias=False,
        )

    def forward(self, x):
        x = x.unsqueeze(-1).permute((0, 2, 1, 3))
        x = self.projection(x)
        x = x.squeeze(1)

        x = torch.matmul(x, self.embedding.weight.T)
        return x


class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(
            self.final_layer_norm(hidden_states)
        )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Wav2Vec2FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        self.intermediate_act_fn = F.gelu

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Wav2Vec2
class Wav2Vec2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [
                Wav2Vec2EncoderLayerStableLayerNorm(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.decoder = Decoder(self.encoder, config)

    def forward(self, input_ids):
        encoded = self.encoder(input_ids)
        hidden_states = self.layer_norm(encoded)
        hidden_states = self.dropout(hidden_states)
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                # attention_mask=attention_mask,
                # output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
        decoded = self.decoder(encoded)
        return decoded


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        text = item["text"]
        byte = text.encode("utf-8")
        tensor = torch.LongTensor(list(byte)).unsqueeze(0)
        return {"input_ids": tensor}


class Config:
    char_dim = 32
    hidden_size = 768
    num_attention_heads = 12

    num_hidden_layers = 12
    attention_dropout = 0.1
    activation_dropout = 0.1
    intermediate_size = 3072
    hidden_dropout = 0.1
    layer_norm_eps = 1e-5

    @property
    def out_channels(self):
        return self.hidden_size

    @property
    def kernel_size(self):
        return (16, self.char_dim)

    @property
    def stride(self):
        return (self.kernel_size[0] // 2, 1)


def main():
    batch_size = 2
    effective_batch_size = 16
    mask_end = False
    mask_p = 0.1
    lr = 1.0e-6
    n_epochs = 5

    config = Config()

    dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    dataset = dataset.filter(lambda example: bool(example["text"]))
    dataset = MyDataset(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate(config), shuffle=True
    )

    model = Model(config)
    model = model.cuda()

    optimizer = Adam(model.parameters(), lr=lr)

    delay = datetime.timedelta(seconds=10)
    last = datetime.datetime.now()

    batch_n = 0
    for i in range(n_epochs):
        with tqdm.tqdm(loader) as pbar:
            for batch in pbar:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                if mask_end:
                    masked_ids = input_ids.clone()
                    ends = attention_mask.cumsum(dim=1)[:, -1]
                    starts = ends - (ends * mask_p).int()
                    for i, (start, end) in enumerate(zip(starts, ends)):
                        masked_ids[i, start:end] = 255
                else:
                    mask = torch.rand(input_ids.shape, device=torch.device("cuda")) < (
                        1 - mask_p
                    )
                    masked_ids = input_ids.clone() * mask + 255 * ~mask
                    masked_ids = attention_mask * masked_ids
                output = model(masked_ids)

                logits = F.log_softmax(output, dim=-1)
                S = logits.shape[1]
                loss = F.nll_loss(
                    logits.view(-1, 256),
                    input_ids[:, :S].contiguous().view(-1),
                    ignore_index=0,
                )
                loss.backward()
                batch_n += batch_size
                if batch_n > effective_batch_size:
                    optimizer.step()
                    batch_n = 0

                if datetime.datetime.now() - last > delay:
                    pbar.set_description(f"Loss {loss.item():.2f}")
                    last = datetime.datetime.now()

                    masked_str = bytes(masked_ids[0].tolist())
                    input_str = bytes(input_ids[0].tolist())
                    predicted_str = bytes(logits[0].argmax(dim=-1).tolist())

                    if mask_end:
                        start, stop = max(ends[0].item() - 50, 0), ends[0].item()
                    else:
                        start, stop = 0, 50
                    print(
                        masked_str[start:stop]
                        .replace(b"\xff", b"X")
                        .decode("utf-8", errors="replace")
                        .strip()
                    )
                    print(
                        input_str[start:stop].decode("utf-8", errors="replace").strip()
                    )
                    print(
                        predicted_str[start:stop]
                        .decode("utf-8", errors="replace")
                        .strip()
                    )


if __name__ == "__main__":
    main()
