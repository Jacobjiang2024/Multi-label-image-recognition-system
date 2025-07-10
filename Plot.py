import matplotlib.pyplot as plt
from collections import Counter

def plot_label_distribution(data, label_column='Labels', figsize=(10, 5)):
    """
    Reads a CSV file, parses multi-label annotations, counts occurrences,
    and plots label distribution.

    Parameters:
    - csv_path (str): Path to the CSV file.
    - label_column (str): Name of the column containing label strings.
    - figsize (tuple): Size of the plot (width, height).
    """
    # Load and process labels
    label_strs = data[label_column].astype(str).tolist()

    label_counts = Counter()
    for s in label_strs:
        for label in s.split():
            if label.isdigit():
                label_counts[int(label)] += 1

    # Prepare for plotting
    labels = sorted(label_counts.keys())
    counts = [label_counts[l] for l in labels]

    # Plot
    plt.figure(figsize=figsize)
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Label Distribution in Dataset')
    plt.xticks(labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
