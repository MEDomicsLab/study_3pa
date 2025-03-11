"""

"""
import matplotlib.pyplot as plt

from result_parameters import metrics_parameters, file_parameters, combined_results_parameters


def generate_mdr_curves(results):

    filename = results['filename']
    for key in file_parameters.keys():
        if filename.startswith(key):
            plot_parameters = file_parameters[key]
            break

    mdr_values = results['loadedFiles']['test']['metrics_dr']

    declaration_rates = sorted(map(int, mdr_values.keys()))
    fig = plt.figure(figsize=(10, 6))

    plt.rcParams.update(**plot_parameters['rcParams'])
    for axis in ['top', 'bottom', 'left', 'right']:
        fig.gca().spines[axis].set_linewidth(0.5)

    for metric, color in metrics_parameters.items():
        values = []
        for dr in declaration_rates:
            dr_str = str(dr)
            if metric in mdr_values[dr_str]['metrics']:
                values.append(mdr_values[dr_str]['metrics'][metric])
            elif metric in ['Positive%', 'population_percentage', 'min_confidence_level', 'mean_confidence_level']:
                values.append(mdr_values[dr_str][metric])
            elif metric == 'NB' and dr != 0 and not plot_parameters['threshold'] is None:
                w = plot_parameters['threshold'] / (1 - plot_parameters['threshold'])
                prevalence = mdr_values[dr_str]['Positive%'] / 100
                sensitivity = mdr_values[dr_str]['metrics']['Sensitivity']
                specificity = mdr_values[dr_str]['metrics']['Specificity']
                if not any(value is None for value in [w, prevalence, sensitivity, specificity]):
                    nb = sensitivity * prevalence - (1 - specificity) * (1 - prevalence) * w
                    values.append(nb)
                else:
                    values.append(None)
            else:
                values.append(None)  # Handle missing values
        metric_label = metric if metric != 'Auc' else 'AUC'
        plt.plot(declaration_rates, values, label=metric_label, color=color,  # , marker='o'
                 linewidth=plot_parameters['linewidth'])

    # If the dr parameter is different than 100, add the vertical line
    if plot_parameters["dr"] != 100:
        plt.axvline(x=plot_parameters["dr"], color='k', linestyle='--', linewidth=plot_parameters['linewidth'])  # Adjust color and line style as needed

    plt.xlabel("Declaration Rate", fontsize=plot_parameters['rcParams']['font.size'] * 1.5)
    plt.ylabel("Metric Value", fontsize=plot_parameters['rcParams']['font.size'] * 1.5)
    if plot_parameters['show_legend']:
        plt.legend(fontsize=plot_parameters['rcParams']['font.size'] * 1.25)
    plt.title(plot_parameters.get("title", "Metrics vs Declaration Rate"),
              fontsize=plot_parameters['rcParams']['font.size'] * 1.75)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=2)
    plt.savefig('figures/' + plot_parameters.get("title", "MDR") + '/mdr.svg', format="svg")
    # plt.show()
    if not plot_parameters["show_legend"]:
        # Create a separate figure for the legend
        fig_legend, ax_legend = plt.subplots(figsize=(2, 2))  # Adjust size as needed
        ax_legend.axis("off")  # Remove axes

        legend = fig_legend.legend(*fig.axes[0].get_legend_handles_labels(), loc="center")

        # Save the legend as an image
        fig_legend.savefig('figures/' + plot_parameters.get("title", "MDR") + '/legend.svg',
                           format="svg", bbox_inches="tight", dpi=300)


def generate_combined_curves(results_list, metric='Accuracy'):
    plt.figure(figsize=(10, 6))

    for results in results_list:
        filename = results['filename']
        for key in file_parameters.keys():
            if filename.startswith(key):
                plot_parameters = file_parameters[key]
                break

        mdr_values = results['loadedFiles']['test']['metrics_dr']

        declaration_rates = sorted(map(int, mdr_values.keys()))
        values = []
        for rate in declaration_rates:
            rate_str = str(rate)
            if metric in mdr_values[rate_str]['metrics']:
                values.append(mdr_values[rate_str]['metrics'][metric])
            elif metric in ['Positive%', 'population_percentage', 'min_confidence_level', 'mean_confidence_level']:
                values.append(mdr_values[rate_str][metric])
            else:
                values.append(None)  # Handle missing values<

        plt.plot(declaration_rates, values, marker=combined_results_parameters['marker'],
                 label=plot_parameters['title'], linewidth=combined_results_parameters['linewidth'])

    plt.xlabel("Declaration Rate")
    plt.ylabel(metric)
    if combined_results_parameters['show_legend']:
        plt.legend()
    plt.title(f"{metric} vs Declaration Rate of all results combined")
    plt.grid()
    plt.rcParams.update(**plot_parameters['rcParams'])
    plt.savefig('figures/mdr_combined.svg', format="svg")
    # plt.show()
