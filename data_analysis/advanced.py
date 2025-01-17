import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Input data
data = {
    'Instigator': {'occurrences': 27, 'true_positives': 5, 'false_positives': 2, 'false_negatives': 22},
    'Conspirator': {'occurrences': 5, 'true_positives': 1, 'false_positives': 4, 'false_negatives': 4},
    'Tyrant': {'occurrences': 20, 'true_positives': 4, 'false_positives': 2, 'false_negatives': 16},
    'Foreign Adversary': {'occurrences': 35, 'true_positives': 24, 'false_positives': 17, 'false_negatives': 11},
    'Traitor': {'occurrences': 7, 'true_positives': 0, 'false_positives': 1, 'false_negatives': 7},
    'Spy': {'occurrences': 1, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 1},
    'Saboteur': {'occurrences': 6, 'true_positives': 2, 'false_positives': 1, 'false_negatives': 4},
    'Corrupt': {'occurrences': 10, 'true_positives': 3, 'false_positives': 2, 'false_negatives': 7},
    'Incompetent': {'occurrences': 15, 'true_positives': 5, 'false_positives': 5, 'false_negatives': 10},
    'Terrorist': {'occurrences': 12, 'true_positives': 6, 'false_positives': 2, 'false_negatives': 6},
    'Deceiver': {'occurrences': 8, 'true_positives': 1, 'false_positives': 2, 'false_negatives': 7},
    'Bigot': {'occurrences': 7, 'true_positives': 0, 'false_positives': 1, 'false_negatives': 7},
    'Guardian': {'occurrences': 30, 'true_positives': 27, 'false_positives': 8, 'false_negatives': 3},
    'Martyr': {'occurrences': 0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 0},
    'Peacemaker': {'occurrences': 12, 'true_positives': 8, 'false_positives': 3, 'false_negatives': 4},
    'Rebel': {'occurrences': 3, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 3},
    'Underdog': {'occurrences': 5, 'true_positives': 1, 'false_positives': 0, 'false_negatives': 4},
    'Virtuous': {'occurrences': 3, 'true_positives': 1, 'false_positives': 2, 'false_negatives': 2},
    'Forgotten': {'occurrences': 0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 0},
    'Exploited': {'occurrences': 3, 'true_positives': 0, 'false_positives': 1, 'false_negatives': 3},
    'Victim': {'occurrences': 48, 'true_positives': 47, 'false_positives': 2, 'false_negatives': 1},
    'Scapegoat': {'occurrences': 0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
}

# Prepare data
metrics = []
for subclass, values in data.items():
    occurrences = values['occurrences']
    tp = values['true_positives']
    fp = values['false_positives']
    fn = values['false_negatives']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics.append({
        'Subclass': subclass,
        'Occurrences': occurrences,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'True Positives': tp,
        'False Positives': fp,
        'False Negatives': fn
    })

# Convert to DataFrame
df = pd.DataFrame(metrics)


df.fillna(0, inplace=True)

# Plot relationship between occurrences and F1 Score
plt.figure(figsize=(10, 6))
plt.scatter(df['Occurrences'], df['F1 Score'], color='skyblue', edgecolor='black')
plt.title('Subclass Occurrences vs. F1 Score', fontsize=14)
plt.xlabel('Occurrences', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Annotate points with subclass names
for i, row in df.iterrows():
    plt.text(row['Occurrences'] + 0.5, row['F1 Score'], row['Subclass'], fontsize=9)

plt.tight_layout()
# plt.savefig('/home/mohamed/repos/nlp_proj/data_analysis/occurrences_vs_f1_score.png')
plt.show()
plt.close()


# Visualization 1: Bar Chart for Occurrences
plt.figure(figsize=(12, 6))
plt.bar(df['Subclass'], df['Occurrences'], color='skyblue')
plt.title('Occurrences per Subclass')
plt.xlabel('Subclass')
plt.ylabel('Occurrences')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# plt.savefig('occurrences_bar_chart.png')
plt.show()
plt.close()

# Visualization 2: Stacked Bar Chart for TP, FP, FN
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(df['Subclass'], df['True Positives'], label='True Positives', color='green')
ax.bar(df['Subclass'], df['False Positives'], label='False Positives', color='orange', bottom=df['True Positives'])
ax.bar(df['Subclass'], df['False Negatives'], label='False Negatives', color='red', bottom=df['True Positives'] + df['False Positives'])
ax.set_title('True Positives, False Positives, and False Negatives per Subclass')
ax.set_xlabel('Subclass')
ax.set_ylabel('Count')
plt.xticks(rotation=45, ha='right')
ax.legend()
plt.tight_layout()
# plt.savefig('stacked_bar_chart.png')
plt.show()
plt.close()

# Visualization 3: Box Plot for Precision, Recall, and F1
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Precision', 'Recall', 'F1 Score']])
plt.title('Distribution of Precision, Recall, and F1 Scores')
plt.ylabel('Score')
plt.tight_layout()
# plt.savefig('metrics_box_plot.png')
plt.show()
plt.close()

# Visualization 4: Line Plot for Occurrences vs F1 Score
sorted_df = df.sort_values(by='Occurrences')
plt.figure(figsize=(12, 6))
plt.plot(sorted_df['Occurrences'], sorted_df['F1 Score'], marker='o', label='F1 Score', color='purple')
plt.title('Occurrences vs F1 Score')
plt.xlabel('Occurrences')
plt.ylabel('F1 Score')
plt.grid(True)
plt.tight_layout()
# plt.savefig('occurrences_vs_f1_line_plot.png')
plt.show()
plt.close()

# Visualization 5: Correlation Heatmap
correlation_matrix = df[['Occurrences', 'Precision', 'Recall', 'F1 Score']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
# plt.savefig('correlation_heatmap.png')
plt.show()
plt.close()

# Save Correlation Matrix as CSV
correlation_matrix.to_csv('correlation_matrix.csv')
