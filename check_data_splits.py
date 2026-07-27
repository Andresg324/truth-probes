import numpy as np

# The 8 statement pairs that are identical once punctuation and case are normalised.
# Each inner list holds the two positions in mixed.json.
DUP_PAIRS = [
    [9, 224],    # greenland is the worlds largest island
    [50, 258],   # water boils at 100 degrees celsius at sea level
    [51, 242],   # the chemical symbol for gold is au
    [103, 302],  # the berlin wall fell in 1989
    [105, 294],  # the united states declared independence in 1776
    [159, 330],  # two plus two equals four
    [160, 354],  # a pentagon has five sides
    [164, 360],  # one kilometer equals one thousand meters
]

for tag in ["0.5B", "1.5B", "3B", "gemma-2b"]:
    z = np.load(f"results/{tag}/split.npz")
    # tr/va/te index the 800-row example array; each statement makes 2 rows,
    # so statement number = row // 2. This is the same grouping the splitter used.
    train = set(np.concatenate([z["tr_idx"], z["va_idx"]]) // 2)
    test  = set(z["te_idx"] // 2)

    straddle = []
    for a, b in DUP_PAIRS:
        # leakage means one copy is in training and its twin is in the test fold
        if (a in train and b in test) or (b in train and a in test):
            straddle.append((a, b))

    print(f"{tag:9s} duplicate pairs split across train/test: {len(straddle)} of 8  {straddle}")