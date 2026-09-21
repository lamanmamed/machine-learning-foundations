# Data

The repository does not include `PB_data.npy`.

The vowel experiment expects the Peterson and Barney data in NumPy dictionary format with at least these arrays:

```text
f1
f2
phoneme_id
```

Run the experiment by passing the local file path:

```bash
python experiments/vowel_gmm.py path/to/PB_data.npy
```

The diabetes regression script uses the same public raw diabetes table referenced by the original experiment. The Iris experiments use scikit-learn's built-in Iris dataset.
