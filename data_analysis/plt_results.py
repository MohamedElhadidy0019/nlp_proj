import matplotlib.pyplot as plt

# Data
models = ['LLaMA 1B', 'LLaMA Instruct 3B', 'XLM RoBERTa 279M']
exact_match_ratios = [0.38, 0.27, 0.15]
colors = ['skyblue', 'orange', 'lightgreen']

model_sizes = [1, 3, 0.279]  # Sizes in billions

symbols = ['*', 'o', 's']  # Star, circle, and square
colors = ['skyblue', 'orange', 'lightgreen']


# Create the plot with larger symbols
plt.figure(figsize=(8, 5))
for size, ratio, color, symbol, model in zip(model_sizes, exact_match_ratios, colors, symbols, models):
    plt.scatter(size, ratio, color=color, marker=symbol, s=200, label=f'{model} ({ratio:.2f})')  # Increased size to 200

# Title and labels
plt.title('Exact Match Ratios vs Model Size', fontsize=14)
plt.xlabel('Model Size (Billion Parameters)', fontsize=12)
plt.ylabel('Exact Match Ratio', fontsize=12)
plt.ylim(0, 0.5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(model_sizes, ['1B', '3B', '279M'])

# Add a legend
plt.legend(title="Models", fontsize=10, title_fontsize=12, loc='upper right')

# Show the diagram
plt.tight_layout()
# plt.show()
# save with thigh rousltion 
plt.savefig('exact_match_ratios_vs_model_size.png', dpi=300)
