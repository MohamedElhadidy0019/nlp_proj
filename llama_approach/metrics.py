from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict
import csv


def evaluate_predictionss(pred_file, ground_truth_file):
    """
    Evaluate predictions against ground truth.

    Args:
        pred_file (str): Path to the predictions file.
        ground_truth_file (str): Path to the ground truth file.

    Returns:
        dict: Metrics for each main class, each subclass, and overall EMR.
    """
    # Load the ground truth
    ground_truth = {}
    with open(ground_truth_file, 'r') as gt_file:
        reader = csv.reader(gt_file)
        counter = 0
        for row in reader:
            _, _, _, _, main_role, subclasses_str = row
            subclasses_list = eval(subclasses_str)  # Convert string to list
            ground_truth[counter] = {'main_class': main_role, 'subclasses': subclasses_list}
            counter += 1

    # Load the predictions
    predictions = {}
    with open(pred_file, 'r') as pred_file:
        reader = csv.reader(pred_file)
        counter = 0
        for row in reader:
            main_role, subclasses_str = row
            subclasses_list = eval(subclasses_str.strip('"'))  # Convert string to list
            predictions[counter] = {'main_class': main_role, 'subclasses': subclasses_list}
            counter += 1

    # Initialize metrics
    all_main_classes = ['Antagonist', 'Protagonist', 'Innocent']  # Add all main classes here
    main_classes = ['Antagonist', 'Protagonist', 'Innocent']
    main_class_to_subclasses = {
        'Antagonist': ['Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary',
                       'Traitor', 'Spy', 'Saboteur', 'Corrupt', 'Incompetent',
                       'Terrorist', 'Deceiver', 'Bigot'],
        'Protagonist': ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous'],
        'Innocent': ['Forgotten', 'Exploited', 'Victim', 'Scapegoat']
    }
    all_subclasses = []
    # loop over main_class_to_subclass
    for key in main_class_to_subclasses:
        for v in main_class_to_subclasses[key]:
            all_subclasses.append(v)

    metrics = {
        'main_class': {cls: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for cls in all_main_classes},
        'subclasses': {subcls: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for subcls in all_subclasses},
        'EMR': 0,
        'total_samples': len(ground_truth),
    }

    # Iterate over ground truth and predictions
    for idx, gt_data in ground_truth.items():
        gt_main_class = gt_data['main_class']
        gt_subclasses = set(gt_data['subclasses'])

        pred_data = predictions.get(idx, {'main_class': None, 'subclasses': []})
        pred_main_class = pred_data['main_class']
        pred_subclasses = set(pred_data['subclasses'])

        # Main class metrics
        for main_class in all_main_classes:
            if gt_main_class == main_class and pred_main_class == main_class:
                metrics['main_class'][main_class]['TP'] += 1
            elif gt_main_class == main_class and pred_main_class != main_class:
                metrics['main_class'][main_class]['FN'] += 1
            elif gt_main_class != main_class and pred_main_class == main_class:
                metrics['main_class'][main_class]['FP'] += 1
            else:
                metrics['main_class'][main_class]['TN'] += 1

        # Subclass metrics
        for subclass in all_subclasses:
            if subclass in gt_subclasses and subclass in pred_subclasses:
                metrics['subclasses'][subclass]['TP'] += 1
            elif subclass in gt_subclasses and subclass not in pred_subclasses:
                metrics['subclasses'][subclass]['FN'] += 1
            elif subclass not in gt_subclasses and subclass in pred_subclasses:
                metrics['subclasses'][subclass]['FP'] += 1
            else:
                metrics['subclasses'][subclass]['TN'] += 1

        # Exact Match Ratio
        if gt_main_class == pred_main_class and gt_subclasses == pred_subclasses:
            metrics['EMR'] += 1

    # Calculate final metrics
    results = {'main_class': {}, 'subclasses': {}, 'EMR': metrics['EMR'] / metrics['total_samples']}
    for main_class, stats in metrics['main_class'].items():
        results['main_class'][main_class] = {
            'precision': stats['TP'] / (stats['TP'] + stats['FP']) if stats['TP'] + stats['FP'] > 0 else 0,
            'recall': stats['TP'] / (stats['TP'] + stats['FN']) if stats['TP'] + stats['FN'] > 0 else 0,
            'f1_score': 2 * stats['TP'] / (2 * stats['TP'] + stats['FP'] + stats['FN']) if 2 * stats['TP'] + stats['FP'] + stats['FN'] > 0 else 0,
        }
    for subclass, stats in metrics['subclasses'].items():
        results['subclasses'][subclass] = {
            'precision': stats['TP'] / (stats['TP'] + stats['FP']) if stats['TP'] + stats['FP'] > 0 else 0,
            'recall': stats['TP'] / (stats['TP'] + stats['FN']) if stats['TP'] + stats['FN'] > 0 else 0,
            'f1_score': 2 * stats['TP'] / (2 * stats['TP'] + stats['FP'] + stats['FN']) if 2 * stats['TP'] + stats['FP'] + stats['FN'] > 0 else 0,
        }

    return results



def main():
    
    # pred_file = '/home/mohamed/repos/nlp_proj/llama_save/og_model.txt'
    pred_file = '/home/mohamed/repos/nlp_proj/llama_save/subclass_model.txt'
    ground_truth_file = '/home/mohamed/repos/nlp_proj/llama_save/test.csv'


    results = evaluate_predictionss(pred_file, ground_truth_file)

    # Print metrics
    print("Main Class Metrics:")
    for main_class, stats in results['main_class'].items():
        print(f"{main_class}: Precision={stats['precision']:.2f}, Recall={stats['recall']:.2f}, F1 Score={stats['f1_score']:.2f}")

    print("\nSubclass Metrics:")
    for subclass, stats in results['subclasses'].items():
        print(f"{subclass}: Precision={stats['precision']:.2f}, Recall={stats['recall']:.2f}, F1 Score={stats['f1_score']:.2f}")

        print(f"\nExact Match Ratio (EMR): {results['EMR']:.2f}")


if __name__ == '__main__':
    main()