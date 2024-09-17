import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import glob

# Model configurations
model_configs = [
    {"folder": "./results/llama-2-7b-hf-lminfinite/", "name": "llama-2-7b-hf-lminfinite", "title": "LM-Infinite", "pretrained_len": 4000},
    {"folder": "./results/llama-2-7b-hf-slimpajama-landmark/", "name": "llama-2-7b-hf-slimpajama-landmark", "title": "Landmark", "pretrained_len": 32000},
    {"folder": "./results/llama-2-7b-hf-slimpajama-longlora-32k/", "name": "llama-2-7b-hf-slimpajama-longlora-32k", "title": "LongLora", "pretrained_len": 32000},
    {"folder": "./results/llama-2-7b-hf-slimpajama-pi-32k/", "name": "llama-2-7b-hf-slimpajama-pi-32k", "title": "PI", "pretrained_len": 32000},
    {"folder": "./results/llama-2-7b-hf-slimpajama-yarn-32k/", "name": "llama-2-7b-hf-slimpajama-yarn-32k", "title": "YaRN", "pretrained_len": 32000},
    {"folder": "./results/llama-2-7b-hf-ntk-frozen/", "name": "llama-2-7b-hf-ntk-frozen", "title": "NTK-F", "pretrained_len": 4000},
    {"folder": "./results/llama-2-7b-hf-slimpajama-ntk-32k/", "name": "llama-2-7b-hf-slimpajama-ntk-32k", "title": "NTK-32k", "pretrained_len": 32000},
    {"folder": "./results/llama-2-7b-hf-slimpajama-ntk-64k/", "name": "llama-2-7b-hf-slimpajama-ntk-64k", "title": "NTK-64k", "pretrained_len": 62000},
    {"folder": "./results/llama-2-7b-hf-slimpajama-ntk-64k-2B/", "name": "llama-2-7b-hf-slimpajama-ntk-64k-2B", "title": "NTK-64k-2B", "pretrained_len": 62000}
]

def load_data(config):
    json_files = glob.glob(f"{config['folder']}*.json")
    data = []
    plot_max_length = 64 * 1000
    expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
    
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            context_length = json_data.get("context_length", None)
            if context_length and context_length < plot_max_length:
                model_response = json_data.get("model_response", "").lower().split()
                score = len(set(model_response).intersection(expected_answer)) / len(expected_answer)
                data.append({
                    "Document Depth": json_data.get("depth_percent", None),
                    "Context Length": context_length,
                    "Score": score
                })
    return pd.DataFrame(data)

def create_heatmap(df, ax, title, pretrain_length):
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    pivot_table = pd.pivot_table(df, values='Score', index='Document Depth', columns='Context Length', aggfunc='mean')
    sns.heatmap(pivot_table, vmin=0, vmax=1, cmap=cmap, ax=ax, cbar=False, linewidths=0.5, linecolor='grey', linestyle='--')
    ax.set_title(title, fontsize=12)
    ax.axvline(x=pretrain_length + 0.5, color='white', linestyle='--', linewidth=3)

def main():
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 9))
    y_labels = ['0%', '', '', '33%', '', '', '67%', '', '', '100%']
    x_labels = ['1K', '', '8k', '', '', '16k', '', '', '', '', '32k', '', '', '', '', '', '', '', '', '64K']
    
    for ax, config in zip(axes.flatten(), model_configs):
        df = load_data(config)
        create_heatmap(df, ax, config['title'], config['pretrained_len'])
        if ax.is_first_col():
            ax.set_yticklabels(y_labels, rotation=0, color="black")
        else:
            ax.set_yticklabels([])
        ax.set_xticklabels(x_labels, rotation=0, color="black")

    fig.text(0, 0.5, 'Depth of Needle', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.savefig("./img/needle.png", dpi=1000)

if __name__ == "__main__":
    main()
