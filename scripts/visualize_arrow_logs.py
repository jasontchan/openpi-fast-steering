import re
import sys
import numpy as np
import matplotlib.pyplot as plt


def parse_vectors(text: str, label: str) -> np.ndarray:
    pattern = rf"{re.escape(label)}:\s*\[([^\]]+)\]"
    matches = re.findall(pattern, text, flags=re.MULTILINE | re.DOTALL)

    if not matches:
        raise ValueError(f"No matches found for label: {label}")

    return np.array([
        np.fromstring(m.replace("\n", " "), sep=" ")
        for m in matches
    ])


def main(log_path: str):
    with open(log_path, "r") as f:
        text = f.read()

    initial = parse_vectors(text, "initial joint velocity arrows")
    decoded_first = parse_vectors(text, "decoded first timestep")
    decoded_mean = parse_vectors(text, "decoded mean first 8 dims")

    print("Parsed arrays:")
    print("initial:", initial.shape)
    print("decoded_first:", decoded_first.shape)
    print("decoded_mean:", decoded_mean.shape)

    joint_labels = [f"J{i+1}" for i in range(7)] + ["gripper"]

    # Nonzero samples
    nonzero_idx = np.where(np.linalg.norm(initial[:, :7], axis=1) > 1e-6)[0]
    Xnz = initial[nonzero_idx, :7]

    print("\nNonzero command indices:", nonzero_idx.tolist())

    # 1. Heatmap of initial commands
    plt.figure(figsize=(10, 5))
    plt.imshow(initial, aspect="auto")
    plt.colorbar(label="joint velocity")
    plt.xticks(np.arange(8), joint_labels)
    plt.yticks(np.arange(len(initial)))
    plt.xlabel("Action dimension")
    plt.ylabel("Sample index")
    plt.title("Initial joint velocity arrows over time")
    plt.tight_layout()
    plt.show()

    # 2. Heatmap of decoded mean
    plt.figure(figsize=(10, 5))
    plt.imshow(decoded_mean, aspect="auto")
    plt.colorbar(label="decoded joint velocity")
    plt.xticks(np.arange(8), joint_labels)
    plt.yticks(np.arange(len(decoded_mean)))
    plt.xlabel("Action dimension")
    plt.ylabel("Sample index")
    plt.title("Decoded FAST action mean over time")
    plt.tight_layout()
    plt.show()

    # 3. Reconstruction error heatmap
    errors = decoded_mean - initial

    plt.figure(figsize=(10, 5))
    plt.imshow(errors, aspect="auto")
    plt.colorbar(label="decoded - initial")
    plt.xticks(np.arange(8), joint_labels)
    plt.yticks(np.arange(len(errors)))
    plt.xlabel("Action dimension")
    plt.ylabel("Sample index")
    plt.title("FAST reconstruction error")
    plt.tight_layout()
    plt.show()

    # 4. Initial vs decoded scatter
    plt.figure(figsize=(5, 5))
    plt.scatter(initial[:, :8].flatten(), decoded_mean[:, :8].flatten())

    lim = max(
        abs(initial[:, :8]).max(),
        abs(decoded_mean[:, :8]).max(),
        1e-6,
    )

    plt.plot([-lim, lim], [-lim, lim])
    plt.xlabel("Initial command")
    plt.ylabel("Decoded command")
    plt.title("Initial vs decoded FAST action values")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # 5. Direction similarity between nonzero joint velocity commands
    if len(Xnz) > 1:
        Xnorm = Xnz / (np.linalg.norm(Xnz, axis=1, keepdims=True) + 1e-8)
        cos_sim = Xnorm @ Xnorm.T

        plt.figure(figsize=(6, 5))
        plt.imshow(cos_sim, vmin=-1, vmax=1)
        plt.colorbar(label="cosine similarity")
        plt.title("Direction similarity between nonzero commands")
        plt.xlabel("Nonzero sample index")
        plt.ylabel("Nonzero sample index")
        plt.tight_layout()
        plt.show()

    # 6. Mean absolute joint contribution
    if len(Xnz) > 0:
        mean_abs = np.mean(np.abs(Xnz), axis=0)

        plt.figure(figsize=(7, 3))
        plt.bar(joint_labels[:7], mean_abs)
        plt.ylabel("mean |joint velocity|")
        plt.title("Average joint contribution magnitude")
        plt.tight_layout()
        plt.show()

    # 7. Norm over time
    norms = np.linalg.norm(initial[:, :7], axis=1)

    plt.figure(figsize=(8, 3))
    plt.plot(norms, marker="o")
    plt.xlabel("Sample index")
    plt.ylabel("||joint velocity||")
    plt.title("Joint velocity command magnitude over time")
    plt.tight_layout()
    plt.show()

    # 8. Individual nonzero command bars
    for idx in nonzero_idx:
        plt.figure(figsize=(7, 3))
        plt.bar(joint_labels, initial[idx])
        plt.axhline(0, linewidth=1)
        plt.ylim(-0.8, 0.8)
        plt.title(f"Initial joint velocity command, sample {idx}")
        plt.ylabel("velocity")
        plt.tight_layout()
        plt.show()

    # 9. Print useful numerical summary
    print("\nMean abs joint contribution over nonzero samples:")
    if len(Xnz) > 0:
        for label, val in zip(joint_labels[:7], np.mean(np.abs(Xnz), axis=0)):
            print(f"{label}: {val:.4f}")

    print("\nFAST reconstruction error summary:")
    print("mean abs error:", np.mean(np.abs(errors)))
    print("max abs error:", np.max(np.abs(errors)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_arrow_logs.py arrow_log.txt")
        sys.exit(1)

    main(sys.argv[1])
