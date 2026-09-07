import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from bayan.attention import attention, MultiHeadAttention, causal_mask


def main():
    torch.manual_seed(42)

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    actual = attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)
    matches = torch.allclose(actual, expected, atol=1e-6)
    print(f"Matches PyTorch reference (atol=1e-6): {matches}")

    output, weights = attention(q, k, v, return_weights=True)
    print("\nAttention weight matrix (batch 0, head 0):")
    print(weights[0, 0].round(decimals=3))
    print(f"Row sums (should all be ~1.0): {weights[0, 0].sum(dim=-1)}")

    d_model = 16
    num_heads = 4
    seq_len = 5
    batch_size = 1

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    out = mha(x)
    print(f"\nMultiHeadAttention output shape: {out.shape} (expected {(batch_size, seq_len, d_model)})")

    mask = torch.ones(seq_len, seq_len)
    mask[0, 3] = 0
    q2 = x.unsqueeze(1)
    k2 = x.unsqueeze(1)
    v2 = x.unsqueeze(1)
    _, masked_weights = attention(q2, k2, v2, mask=mask, return_weights=True)
    print("\nMasked attention weights (row 0, should be ~0 at column 3):")
    print(masked_weights[0, 0, 0].round(decimals=4))

    causal_seq_len = 5
    x_causal = torch.randn(1, 1, causal_seq_len, 8)
    causal = causal_mask(causal_seq_len)
    _, causal_weights = attention(x_causal, x_causal, x_causal, mask=causal, return_weights=True)

    print("\nCausal attention weights (should be lower triangular):")
    print(causal_weights[0, 0].round(decimals=3))

    upper_triangle = torch.triu(causal_weights[0, 0], diagonal=1)
    is_lower_triangular = torch.allclose(upper_triangle, torch.zeros_like(upper_triangle))
    print(f"Is lower triangular (upper part ~0): {is_lower_triangular}")
    print("Model family: Decoder-style causal attention")


def inspect_attention_maps():
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    texts = [
        "الخدمة سيئة جداً وين المسؤول",
        "There is a pothole near Tahlia Street",
    ]

    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    tokens_per_seq = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
    print("\nTokens:")
    for i, toks in enumerate(tokens_per_seq):
        print(f"  seq {i}: {toks}")

    with torch.no_grad():
        out_masked = model(input_ids=input_ids, attention_mask=attention_mask)
        out_unmasked = model(input_ids=input_ids, attention_mask=torch.ones_like(attention_mask))

    attn_masked = out_masked.attentions
    attn_unmasked = out_unmasked.attentions

    num_layers = len(attn_masked)
    num_heads = attn_masked[0].shape[1]
    print(f"\nnum_layers={num_layers}, num_heads={num_heads}")

    pad_positions = (attention_mask == 0)

    def pad_mass(attn_tuple):
        total = 0.0
        total_weight = 0.0
        for layer_attn in attn_tuple:
            b = layer_attn.shape[0]
            for bi in range(b):
                pad_idx = pad_positions[bi].nonzero(as_tuple=True)[0]
                if len(pad_idx) == 0:
                    continue
                mass = layer_attn[bi, :, :, pad_idx].sum().item()
                total += mass
                total_weight += layer_attn[bi].sum().item()
        return total / total_weight if total_weight > 0 else 0.0

    masked_pad_mass = pad_mass(attn_masked)
    unmasked_pad_mass = pad_mass(attn_unmasked)
    print(f"\nPad attention mass WITH correct mask: {masked_pad_mass:.4f}")
    print(f"Pad attention mass WITHOUT mask (all-ones): {unmasked_pad_mass:.4f}")

    last_layer = attn_masked[-1][0]
    seq_len_0 = int(attention_mask[0].sum().item())
    diag_scores = []
    for h in range(num_heads):
        head_attn = last_layer[h][:seq_len_0, :seq_len_0]
        adjacency_mass = 0.0
        for i in range(head_attn.shape[0]):
            lo, hi = max(0, i - 1), min(head_attn.shape[0], i + 2)
            adjacency_mass += head_attn[i, lo:hi].sum().item()
        diag_scores.append(adjacency_mass / head_attn.shape[0])
    best_head = max(range(num_heads), key=lambda h: diag_scores[h])
    print(f"\nMost adjacency-looking head in last layer: head {best_head} (avg local mass={diag_scores[best_head]:.3f})")

    sep_positions = [toks.index("[SEP]") if "[SEP]" in toks else None for toks in tokens_per_seq]
    sep_mass_total = 0.0
    sep_mass_count = 0
    for layer_attn in attn_masked:
        for bi, sep_pos in enumerate(sep_positions):
            if sep_pos is None:
                continue
            mass = layer_attn[bi, :, :, sep_pos].mean().item()
            sep_mass_total += mass
            sep_mass_count += 1
    avg_sep_mass = sep_mass_total / sep_mass_count if sep_mass_count else 0.0
    print(f"\nAverage attention mass directed at [SEP] token (all layers/heads): {avg_sep_mass:.4f}")


if __name__ == "__main__":
    main()
    inspect_attention_maps()
