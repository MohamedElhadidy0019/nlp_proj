# Classification report data
classification_report = {
    'Terrorist': {'precision': 0.0650887573964497, 'recall': 0.9166666666666666, 'f1-score': 0.12154696132596685, 'support': 12.0},
    'Corrupt': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 10.0},
    'Foreign Adversary': {'precision': 0.03389830508474576, 'recall': 0.05555555555555555, 'f1-score': 0.042105263157894736, 'support': 36.0},
    'Spy': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 1.0},
    'Saboteur': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 6.0},
    'Victim': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 64.0},
    'Conspirator': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 5.0},
    'Traitor': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 7.0},
    'Deceiver': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 8.0},
    'Rebel': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 4.0},
    'Instigator': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 29.0},
    'Tyrant': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 21.0},
    'Guardian': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 38.0},
    'Incompetent': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 19.0},
    'Underdog': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 6.0},
    'Virtuous': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 7.0},
    'Bigot': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 8.0},
    'Exploited': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 3.0},
    'Peacemaker': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 14.0},
    'Scapegoat': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 1.0},
    'accuracy': 0.043478260869565216,
    'macro avg': {'precision': 0.004949353124059773, 'recall': 0.04861111111111111, 'f1-score': 0.008182611224193078, 'support': 299.0},
    'weighted avg': {'precision': 0.006693659103037605, 'recall': 0.043478260869565216, 'f1-score': 0.00994766892841409, 'support': 299.0}
}

# Extract subclass metrics and calculate accuracy
for subclass, metrics in classification_report.items():
    if subclass not in ['accuracy', 'macro avg', 'weighted avg']:
        recall = metrics['recall']
        support = metrics['support']
        accuracy = recall  # Since accuracy = TP / support, and recall = TP / (TP + FN), and support = TP + FN
        print(f"Subclass: {subclass}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1-score']:.4f}")
        print(f"  Support: {metrics['support']}")
        print(f"  Accuracy: {accuracy:.4f}")
        print()