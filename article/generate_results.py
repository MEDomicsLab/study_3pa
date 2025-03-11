"""

"""
from utils import extract_results_data
from plot_metrics import generate_mdr_curves, generate_combined_curves
from tree_visualizer import Visualizer


# Extract results files
folder_path = "results"  # Change this to your folder path
results_list = extract_results_data(folder_path)

# Generate MDR curves for each result
for results in results_list:
    tv = Visualizer(results)
    tv.visualize(samp_ratio=0, data_set='test')
    generate_mdr_curves(results)

# # Generate Accuracies vs Declaration Rate of all results combined
generate_combined_curves(results_list, metric='Accuracy')
