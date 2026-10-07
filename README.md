# SeeLM

This is my small language model.

## Data

The training corpora are not in the repository (they are gigabytes). Fetch them with:

```
./download_data.sh               # TinyStories and OpenWebText
./download_data.sh tinystories   # TinyStories only
```

Then train a tokenizer:

```
uv run python tokenization/train_bpe.py
```

The trained vocabulary and merges live in `tokenization/outputs/`.
