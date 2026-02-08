import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import re
from collections import defaultdict

def parse_evaluation_file(filename):
    """Parse the evaluation file to extract confusion matrix and metrics."""
    print(f"Parsing file: {filename}")  # Debug print
    
    metrics = {}
    confusion_matrix = np.zeros((2, 2))
    
    if not os.path.exists(filename):
        print(f"Error: File does not exist: {filename}")
        return confusion_matrix, metrics
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
        # Print first 200 characters for debugging
        print(f"File content preview: {content[:200]}...")
        
        # Use regular expressions to extract the confusion matrix values
        
        # Look for the true positive value (between True Dynamic and Pred Dynamic)
        tp_match = re.search(r'│\s*True Dynamic\s*│\s*(\d+)\s*│', content)
        if tp_match:
            confusion_matrix[0, 0] = int(tp_match.group(1))
        
        # Look for the false negative value (between True Dynamic and Pred Static)
        fn_match = re.search(r'│\s*True Dynamic\s*│\s*\d+\s*│\s*(\d+)\s*│', content)
        if fn_match:
            confusion_matrix[0, 1] = int(fn_match.group(1))
        
        # Look for the false positive value (between True Static and Pred Dynamic)
        fp_match = re.search(r'│\s*True Static\s*│\s*(\d+)\s*│', content)
        if fp_match:
            confusion_matrix[1, 0] = int(fp_match.group(1))
        
        # Look for the true negative value (between True Static and Pred Static)
        tn_match = re.search(r'│\s*True Static\s*│\s*\d+\s*│\s*(\d+)\s*│', content)
        if tn_match:
            confusion_matrix[1, 1] = int(tn_match.group(1))
        
        # Parse metrics using regex
        accuracy_match = re.search(r'Accuracy:\s*([\d.]+)%', content)
        if accuracy_match:
            metrics['accuracy'] = float(accuracy_match.group(1)) / 100
        
        precision_match = re.search(r'Precision:\s*([\d.]+)%', content)
        if precision_match:
            metrics['precision'] = float(precision_match.group(1)) / 100
        
        recall_match = re.search(r'Recall:\s*([\d.]+)%', content)
        if recall_match:
            metrics['recall'] = float(recall_match.group(1)) / 100
        
        f1_match = re.search(r'F1 Score:\s*([\d.]+)%', content)
        if f1_match:
            metrics['f1'] = float(f1_match.group(1)) / 100
        
        # Extract filter statistics
        dynamic_match = re.search(r'Dynamic Objects Filtered:\s*(\d+)', content)
        if dynamic_match:
            metrics['dynamic_filtered'] = int(dynamic_match.group(1))
            
        ground_match = re.search(r'Ground Points Filtered:\s*(\d+)', content)
        if ground_match:
            metrics['ground_filtered'] = int(ground_match.group(1))
            
        sky_match = re.search(r'Sky Points Filtered:\s*(\d+)', content)
        if sky_match:
            metrics['sky_filtered'] = int(sky_match.group(1))
            
        edge_match = re.search(r'Edge Points Filtered:\s*(\d+)', content)
        if edge_match:
            metrics['edge_filtered'] = int(edge_match.group(1))
        
        # Try to extract temporal filtered if available (for improved filter)
        temporal_match = re.search(r'Temporal Points Filtered:\s*(\d+)', content)
        if temporal_match:
            metrics['temporal_filtered'] = int(temporal_match.group(1))
        
        # Also try to extract the cluster filtered points (for improved filter)
        cluster_match = re.search(r'Cluster points filtered:\s*(\d+)', content)
        if cluster_match:
            metrics['cluster_filtered'] = int(cluster_match.group(1))
        
        print(f"Extracted confusion matrix:\n{confusion_matrix}")
        print(f"Extracted metrics: {metrics}")
        
    except Exception as e:
        print(f"Error parsing file: {e}")
    
    return confusion_matrix, metrics

def plot_confusion_matrix(matrix, output_file, title="Point Cloud Filter Confusion Matrix"):
    """Plot a confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    
    # Calculate derived metrics to include in the plot
    tp, fn = matrix[0, 0], matrix[0, 1]
    fp, tn = matrix[1, 0], matrix[1, 1]
    
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Ensure the matrix is converted to float for plotting
    matrix_float = matrix.astype(float)
    
    df_cm = pd.DataFrame(matrix_float, 
                     index=['True Dynamic', 'True Static'], 
                     columns=['Pred Dynamic', 'Pred Static'])
    
    # Calculate percentages for annotations
    row_sums = df_cm.sum(axis=1)
    percentage_matrix = df_cm.divide(row_sums, axis=0).round(2) * 100
    
    # Create annotations with both count and percentage
    annot_matrix = df_cm.astype(int).astype(str) + '\n(' + percentage_matrix.astype(str) + '%)'
    
    # Plot heatmap
    ax = sns.heatmap(df_cm, annot=annot_matrix, fmt='', cmap='Blues', cbar=True, 
                 annot_kws={"size": 12})
    
    # Add metrics as text
    plt.figtext(0.02, 0.02, f"Accuracy: {accuracy:.2%}\nPrecision: {precision:.2%}\nRecall: {recall:.2%}\nF1 Score: {f1:.2%}", 
              fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(title, fontsize=16)
    plt.ylabel('True Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return accuracy, precision, recall, f1

def plot_latest_and_first_matrices(all_matrices, frames, output_dir):
    """Plot the first and latest confusion matrices side by side for comparison."""
    if len(all_matrices) < 2:
        print("Not enough matrices for comparison")
        return
    
    # Sort frames and corresponding matrices
    sorted_indices = np.argsort(frames)
    sorted_frames = [frames[i] for i in sorted_indices]
    sorted_matrices = [all_matrices[i] for i in sorted_indices]
    
    # Get first and latest matrices
    first_matrix = sorted_matrices[0]
    latest_matrix = sorted_matrices[-1]
    first_frame = sorted_frames[0]
    latest_frame = sorted_frames[-1]
    
    # Set up the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # First matrix (subplot 1)
    df_first = pd.DataFrame(first_matrix.astype(float), 
                          index=['True Dynamic', 'True Static'],
                          columns=['Pred Dynamic', 'Pred Static'])
    
    # Calculate percentages for first matrix
    row_sums_first = df_first.sum(axis=1)
    percentage_first = df_first.divide(row_sums_first, axis=0).round(2) * 100
    annot_first = df_first.astype(int).astype(str) + '\n(' + percentage_first.astype(str) + '%)'
    
    # Plot first matrix
    sns.heatmap(df_first, annot=annot_first, fmt='', cmap='Blues', cbar=True, 
              annot_kws={"size": 12}, ax=axes[0])
    axes[0].set_title(f"Initial Filter (Frame {first_frame})", fontsize=16)
    axes[0].set_ylabel('True Class', fontsize=14)
    axes[0].set_xlabel('Predicted Class', fontsize=14)
    
    # Calculate metrics for first matrix
    tp, fn = first_matrix[0, 0], first_matrix[0, 1]
    fp, tn = first_matrix[1, 0], first_matrix[1, 1]
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Add metrics text for first matrix
    axes[0].text(0.05, -0.15, f"Accuracy: {accuracy:.2%}\nPrecision: {precision:.2%}\nRecall: {recall:.2%}\nF1 Score: {f1:.2%}", 
               transform=axes[0].transAxes, fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8))
    
    # Latest matrix (subplot 2)
    df_latest = pd.DataFrame(latest_matrix.astype(float), 
                           index=['True Dynamic', 'True Static'],
                           columns=['Pred Dynamic', 'Pred Static'])
    
    # Calculate percentages for latest matrix
    row_sums_latest = df_latest.sum(axis=1)
    percentage_latest = df_latest.divide(row_sums_latest, axis=0).round(2) * 100
    annot_latest = df_latest.astype(int).astype(str) + '\n(' + percentage_latest.astype(str) + '%)'
    
    # Plot latest matrix
    sns.heatmap(df_latest, annot=annot_latest, fmt='', cmap='Blues', cbar=True, 
              annot_kws={"size": 12}, ax=axes[1])
    axes[1].set_title(f"Latest Filter (Frame {latest_frame})", fontsize=16)
    axes[1].set_ylabel('True Class', fontsize=14)
    axes[1].set_xlabel('Predicted Class', fontsize=14)
    
    # Calculate metrics for latest matrix
    tp, fn = latest_matrix[0, 0], latest_matrix[0, 1]
    fp, tn = latest_matrix[1, 0], latest_matrix[1, 1]
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Add metrics text for latest matrix
    axes[1].text(0.05, -0.15, f"Accuracy: {accuracy:.2%}\nPrecision: {precision:.2%}\nRecall: {recall:.2%}\nF1 Score: {f1:.2%}", 
               transform=axes[1].transAxes, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

def aggregate_metrics_by_frame_range(frames, metrics_dict, bin_size=100):
    """Aggregate metrics by frame ranges to reduce noise."""
    # Create bins for frame ranges
    max_frame = max(frames)
    bins = list(range(0, max_frame + bin_size, bin_size))
    binned_metrics = defaultdict(lambda: defaultdict(list))
    
    # Assign metrics to bins
    for i, frame in enumerate(frames):
        bin_idx = min(len(bins) - 2, frame // bin_size)
        bin_start = bins[bin_idx]
        bin_end = bins[bin_idx + 1] - 1
        bin_label = f"{bin_start}-{bin_end}"
        
        for metric_name, values in metrics_dict.items():
            if i < len(values):  # Ensure index is valid
                binned_metrics[bin_label][metric_name].append(values[i])
    
    # Calculate average for each bin
    bin_labels = []
    aggregated_metrics = defaultdict(list)
    
    for bin_label in sorted(binned_metrics.keys(), key=lambda x: int(x.split('-')[0])):
        bin_labels.append(bin_label)
        
        for metric_name in metrics_dict.keys():
            if binned_metrics[bin_label][metric_name]:  # Check if there are values
                avg_value = sum(binned_metrics[bin_label][metric_name]) / len(binned_metrics[bin_label][metric_name])
                aggregated_metrics[metric_name].append(avg_value)
            else:
                aggregated_metrics[metric_name].append(0)  # Default if no values
    
    return bin_labels, aggregated_metrics

def plot_metrics_over_time(frames, metrics_dict, output_file, title="Filter Performance Metrics Over Time"):
    """Plot metrics over time with cleaner visualization."""
    # Aggregate metrics to reduce noise
    bin_labels, aggregated_metrics = aggregate_metrics_by_frame_range(frames, metrics_dict)
    
    plt.figure(figsize=(12, 8))
    
    line_styles = {'accuracy': '-', 'precision': '--', 'recall': '-.', 'f1': ':'}
    colors = {'accuracy': 'blue', 'precision': 'orange', 'recall': 'green', 'f1': 'red'}
    markers = {'accuracy': 'o', 'precision': 's', 'recall': '^', 'f1': 'D'}
    
    for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
        if metric_name in aggregated_metrics and len(aggregated_metrics[metric_name]) > 0:
            plt.plot(bin_labels, aggregated_metrics[metric_name], 
                   label=metric_name.capitalize(),
                   color=colors[metric_name],
                   linestyle=line_styles[metric_name],
                   marker=markers[metric_name],
                   linewidth=2,
                   markersize=8)
    
    # Add value annotations for first and last points
    for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
        if metric_name in aggregated_metrics and len(aggregated_metrics[metric_name]) > 1:
            values = aggregated_metrics[metric_name]
            # Annotate first point
            plt.annotate(f"{values[0]:.3f}", 
                       (bin_labels[0], values[0]),
                       textcoords="offset points",
                       xytext=(0,10), 
                       ha='center')
            # Annotate last point
            plt.annotate(f"{values[-1]:.3f}", 
                       (bin_labels[-1], values[-1]),
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center')
    
    plt.xlabel('Frame Range', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.7, 1.0)  # Set y-axis to focus on the relevant range
    
    # Add text summary of trends
    trends = {}
    for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
        if metric_name in aggregated_metrics and len(aggregated_metrics[metric_name]) > 1:
            values = aggregated_metrics[metric_name]
            start, end = values[0], values[-1]
            change = ((end - start) / start) * 100 if start != 0 else 0
            trends[metric_name] = change
    
    trend_text = "Performance Trends:\n"
    for metric_name, change in trends.items():
        trend_text += f"{metric_name.capitalize()}: {change:.1f}% change\n"
    
    plt.figtext(0.02, 0.02, trend_text, fontsize=10, 
              bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return aggregated_metrics

def plot_filter_categories(frames, metrics_dict, output_file, title="Points Filtered by Category"):
    """Plot filter categories over time with improved visualization."""
    # Aggregate metrics to reduce noise
    bin_labels, aggregated_metrics = aggregate_metrics_by_frame_range(frames, metrics_dict)
    
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers for each category
    categories = ['dynamic_filtered', 'ground_filtered', 'sky_filtered', 'edge_filtered', 
                'temporal_filtered', 'cluster_filtered']
    colors = ['red', 'brown', 'skyblue', 'purple', 'orange', 'green']
    markers = ['o', 's', '^', 'D', 'x', '+']
    
    # Plot each category that has data
    for i, category in enumerate(categories):
        if category in aggregated_metrics and any(aggregated_metrics[category]):
            label = category.replace('_', ' ').title()
            plt.plot(bin_labels, aggregated_metrics[category], 
                   label=label,
                   color=colors[i % len(colors)],
                   marker=markers[i % len(markers)],
                   linewidth=2,
                   markersize=8)
    
    # Calculate total filtered points for each frame range
    if any(category in aggregated_metrics for category in categories):
        total_filtered = []
        for i in range(len(bin_labels)):
            total = 0
            for category in categories:
                if category in aggregated_metrics and i < len(aggregated_metrics[category]):
                    total += aggregated_metrics[category][i]
            total_filtered.append(total)
        
        # Plot total as a thick black line
        plt.plot(bin_labels, total_filtered, 
               label='Total Filtered',
               color='black', 
               linewidth=3,
               linestyle='-',
               marker='*',
               markersize=10)
    
    plt.xlabel('Frame Range', fontsize=14)
    plt.ylabel('Number of Points', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add annotations for first and last points of total filtered
    if 'total_filtered' in locals() and len(total_filtered) > 1:
        # Annotate first point
        plt.annotate(f"{total_filtered[0]}", 
                   (bin_labels[0], total_filtered[0]),
                   textcoords="offset points",
                   xytext=(0,10), 
                   ha='center')
        # Annotate last point
        plt.annotate(f"{total_filtered[-1]}", 
                   (bin_labels[-1], total_filtered[-1]),
                   textcoords="offset points", 
                   xytext=(0,10), 
                   ha='center')
    
    # Add summary statistics
    if any(category in aggregated_metrics for category in categories):
        # Calculate average values for each category
        summary_text = "Average Points Filtered:\n"
        for category in categories:
            if category in aggregated_metrics and any(aggregated_metrics[category]):
                avg_value = sum(aggregated_metrics[category]) / len(aggregated_metrics[category])
                summary_text += f"{category.replace('_', ' ').title()}: {avg_value:.1f}\n"
        
        if 'total_filtered' in locals():
            avg_total = sum(total_filtered) / len(total_filtered)
            summary_text += f"Total: {avg_total:.1f}\n"
        
        plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                  bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_precision_recall_curve(all_frames, all_metrics, output_file):
    """Plot precision-recall curve to show the tradeoff."""
    # Extract precision and recall values
    precision_values = []
    recall_values = []
    
    for i, frame in enumerate(all_frames):
        if 'precision' in all_metrics and 'recall' in all_metrics:
            if i < len(all_metrics['precision']) and i < len(all_metrics['recall']):
                precision_values.append(all_metrics['precision'][i])
                recall_values.append(all_metrics['recall'][i])
    
    if not precision_values or not recall_values:
        print("No precision/recall data available for curve")
        return
    
    # Sort by recall (x-axis)
    sorted_indices = np.argsort(recall_values)
    sorted_recall = [recall_values[i] for i in sorted_indices]
    sorted_precision = [precision_values[i] for i in sorted_indices]
    sorted_frames = [all_frames[i] for i in sorted_indices]
    
    plt.figure(figsize=(10, 8))
    
    # Plot the curve
    plt.plot(sorted_recall, sorted_precision, 'b-', linewidth=2)
    
    # Add points with frame numbers
    num_points = len(sorted_frames)
    max_annotations = min(10, num_points)  # Limit number of annotations to avoid clutter
    annotation_indices = np.linspace(0, num_points-1, max_annotations, dtype=int)
    
    for i in annotation_indices:
        plt.plot(sorted_recall[i], sorted_precision[i], 'ro', markersize=8)
        plt.annotate(f"Frame {sorted_frames[i]}", 
                   (sorted_recall[i], sorted_precision[i]),
                   textcoords="offset points",
                   xytext=(0,10), 
                   ha='center',
                   fontsize=9)
    
    # Add F1 score isocurves
    f1_values = [0.7, 0.8, 0.9, 0.95]
    x = np.linspace(0.5, 1.0, 100)
    
    for f1 in f1_values:
        # Formula: precision = (f1 * recall) / (2 * recall - f1)
        y = [(f1 * r) / (2 * r - f1) if (2 * r - f1) != 0 else None for r in x]
        valid_indices = [i for i, val in enumerate(y) if val is not None and val >= 0 and val <= 1]
        if valid_indices:
            valid_x = [x[i] for i in valid_indices]
            valid_y = [y[i] for i in valid_indices]
            plt.plot(valid_x, valid_y, 'g--', alpha=0.5)
            
            # Add F1 label at the middle of the curve
            middle_idx = len(valid_x) // 2
            if middle_idx < len(valid_x):
                plt.annotate(f"F1={f1}", 
                           (valid_x[middle_idx], valid_y[middle_idx]),
                           textcoords="offset points",
                           xytext=(5,0), 
                           fontsize=9)
    
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve for Point Cloud Filtering', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.axis([0.7, 1.0, 0.7, 1.0])  # Focus on the relevant part of the curve
    
    plt.savefig(output_file, dpi=300)
    plt.close()

def generate_metrics_table(all_frames, all_metrics, output_file):
    """Generate a summary table of metrics."""
    if not all_frames or not all_metrics:
        print("No data available for metrics table")
        return
    
    # Prepare data for table
    data = {
        'Frame': all_frames,
        'Accuracy': all_metrics.get('accuracy', [0] * len(all_frames)),
        'Precision': all_metrics.get('precision', [0] * len(all_frames)),
        'Recall': all_metrics.get('recall', [0] * len(all_frames)),
        'F1 Score': all_metrics.get('f1', [0] * len(all_frames)),
        'Dynamic': all_metrics.get('dynamic_filtered', [0] * len(all_frames)),
        'Ground': all_metrics.get('ground_filtered', [0] * len(all_frames)),
        'Sky': all_metrics.get('sky_filtered', [0] * len(all_frames)),
        'Edge': all_metrics.get('edge_filtered', [0] * len(all_frames))
    }
    
    # Add temporal and cluster filtered if available
    if 'temporal_filtered' in all_metrics:
        data['Temporal'] = all_metrics['temporal_filtered']
    if 'cluster_filtered' in all_metrics:
        data['Cluster'] = all_metrics['cluster_filtered']
    
    # Create pandas DataFrame
    df = pd.DataFrame(data)
    
    # Calculate summary statistics
    summary = df.describe()
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    summary.to_csv(output_file.replace('.csv', '_summary.csv'))
    
    print(f"Metrics table saved to {output_file}")
    return df

def main():
    eval_dir = './filter_evaluation/'
    output_dir = './filter_visualization/'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all evaluation files
    eval_files = sorted(glob.glob(os.path.join(eval_dir, 'filter_eval_*.txt')))
    print(f"Found {len(eval_files)} evaluation files: {eval_files}")
    
    all_frames = []
    all_matrices = []
    all_metrics = defaultdict(list)
    
    # Parse all evaluation files
    for file_path in eval_files:
        frame_num = int(os.path.basename(file_path).split('_')[2].split('.')[0])
        all_frames.append(frame_num)
        
        confusion_matrix, metrics = parse_evaluation_file(file_path)
        all_matrices.append(confusion_matrix)
        
        # Store metrics
        for key, value in metrics.items():
            all_metrics[key].append(value)
        
        # Generate individual confusion matrix visualization
        output_file = os.path.join(output_dir, f'confusion_matrix_{frame_num}.png')
        plot_confusion_matrix(confusion_matrix, output_file, 
                             title=f"Point Cloud Filter Confusion Matrix (Frame {frame_num})")
    
    # Plot comparison between first and latest matrices
    if len(all_matrices) >= 2:
        plot_latest_and_first_matrices(all_matrices, all_frames, output_dir)
    
    # Plot metrics over time with improvements
    if all_frames:
        # Plot performance metrics
        metrics_file = os.path.join(output_dir, 'metrics_over_time.png')
        aggregated_metrics = plot_metrics_over_time(all_frames, all_metrics, metrics_file)
        
        # Plot filter categories
        categories_file = os.path.join(output_dir, 'filter_categories.png')
        plot_filter_categories(all_frames, all_metrics, categories_file)
        
        # Plot precision-recall curve
        pr_curve_file = os.path.join(output_dir, 'precision_recall_curve.png')
        plot_precision_recall_curve(all_frames, all_metrics, pr_curve_file)
        
        # Generate metrics table
        metrics_table_file = os.path.join(output_dir, 'metrics_summary.csv')
        generate_metrics_table(all_frames, all_metrics, metrics_table_file)
    else:
        print("No evaluation files found or processed.")
        
        # Fallback: Create manual visualization based on confusion matrix
        print("Creating manual visualization based on console output...")
        
        # Use the confusion matrix from the point cloud filter
        manual_matrix = np.array([
            [251621, 1894390],  # True Dynamic (TP, FN)
            [2370, 41892]       # True Static (FP, TN)
        ])
        
        # Plot manual confusion matrix
        output_file = os.path.join(output_dir, 'confusion_matrix_manual.png')
        accuracy, precision, recall, f1 = plot_confusion_matrix(manual_matrix, output_file, 
                                                             title="Current Point Cloud Filter Confusion Matrix")
        
        print(f"Manual metrics: Accuracy={accuracy*100:.2f}%, Precision={precision*100:.2f}%, Recall={recall*100:.2f}%, F1={f1*100:.2f}%")
        
        # Create expected improved matrix (prediction of how the improved filter should perform)
        improved_matrix = np.array([
            [1500000, 645000],  # More true positives, fewer false negatives
            [10000, 34000]      # Slightly more false positives, similar true negatives
        ])
        
        # Plot improved matrix prediction
        improved_file = os.path.join(output_dir, 'confusion_matrix_improved_prediction.png')
        imp_accuracy, imp_precision, imp_recall, imp_f1 = plot_confusion_matrix(
            improved_matrix, improved_file, 
            title="Predicted Performance with Improved Filter"
        )
        
        # Plot side-by-side comparison of current and improved matrices
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Current matrix
        df_current = pd.DataFrame(manual_matrix.astype(float), 
                               index=['True Dynamic', 'True Static'],
                               columns=['Pred Dynamic', 'Pred Static'])
        
        # Calculate percentages
        row_sums_current = df_current.sum(axis=1)
        percentage_current = df_current.divide(row_sums_current, axis=0).round(2) * 100
        annot_current = df_current.astype(int).astype(str) + '\n(' + percentage_current.astype(str) + '%)'
        
        sns.heatmap(df_current, annot=annot_current, fmt='', cmap='Blues', cbar=True, 
                  annot_kws={"size": 12}, ax=axes[0])
        axes[0].set_title("Current Filter Performance", fontsize=16)
        axes[0].set_ylabel('True Class', fontsize=14)
        axes[0].set_xlabel('Predicted Class', fontsize=14)
        
        # Add metrics for current matrix
        axes[0].text(0.05, -0.15, 
                   f"Accuracy: {accuracy:.2%}\nPrecision: {precision:.2%}\nRecall: {recall:.2%}\nF1 Score: {f1:.2%}", 
                   transform=axes[0].transAxes, fontsize=12,
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Improved matrix prediction
        df_improved = pd.DataFrame(improved_matrix.astype(float), 
                                index=['True Dynamic', 'True Static'],
                                columns=['Pred Dynamic', 'Pred Static'])
        
        # Calculate percentages
        row_sums_improved = df_improved.sum(axis=1)
        percentage_improved = df_improved.divide(row_sums_improved, axis=0).round(2) * 100
        annot_improved = df_improved.astype(int).astype(str) + '\n(' + percentage_improved.astype(str) + '%)'
        
        sns.heatmap(df_improved, annot=annot_improved, fmt='', cmap='Greens', cbar=True, 
                  annot_kws={"size": 12}, ax=axes[1])
        axes[1].set_title("Projected Improved Filter Performance", fontsize=16)
        axes[1].set_ylabel('True Class', fontsize=14)
        axes[1].set_xlabel('Predicted Class', fontsize=14)
        
        # Add metrics for improved matrix
        axes[1].text(0.05, -0.15, 
                   f"Accuracy: {imp_accuracy:.2%}\nPrecision: {imp_precision:.2%}\nRecall: {imp_recall:.2%}\nF1 Score: {imp_f1:.2%}", 
                   transform=axes[1].transAxes, fontsize=12,
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Add improvement statistics between the matrices
        improvement_text = f"""Performance Improvement:
Accuracy: {(imp_accuracy-accuracy)*100:.1f}% increase
Precision: {(imp_precision-precision)*100:.1f}% change
Recall: {(imp_recall-recall)*100:.1f}% increase
F1 Score: {(imp_f1-f1)*100:.1f}% increase

False Negative Reduction: {((manual_matrix[0,1]-improved_matrix[0,1])/manual_matrix[0,1]*100):.1f}%
"""
        
        fig.text(0.5, 0.01, improvement_text, ha='center', fontsize=14,
               bbox=dict(facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_current_vs_improved.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create example filter categories data for visualization
        manual_categories = {
            'dynamic_filtered': [25000, 50000, 75000, 100000],  # Increasing trend
            'ground_filtered': [15000, 18000, 17000, 16000],
            'sky_filtered': [10000, 12000, 11000, 10500],
            'edge_filtered': [8000, 8500, 8200, 8300],
            'temporal_filtered': [0, 15000, 25000, 30000],      # New in improved filter
            'cluster_filtered': [0, 10000, 15000, 20000]        # New in improved filter
        }
        
        example_frames = [100, 500, 900, 1300]  # Example frame numbers
        
        # Plot example filter categories
        categories_file = os.path.join(output_dir, 'filter_categories_example.png')
        plot_filter_categories(example_frames, manual_categories, categories_file,
                            title="Example Points Filtered by Category with Improved Filter")

if __name__ == '__main__':
    main()