import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skewnorm
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

# Set random seed for reproducibility
np.random.seed(15)

n_points = 75

# Define Seaborn color palette (to match colors between density and scatter plots)
palette = sns.color_palette("deep")
hospital_a_color = palette[0]  # Blue
hospital_b_color = palette[2]  # Green

# Class 0, Hospital A (circles) - Sinusoidal pattern with some noise
class0_groupA_x = np.clip(skewnorm.rvs(-3, loc=6, scale=3, size=n_points), 0, 12.5)
class0_groupA_y = (3 * np.sin(0.5 * class0_groupA_x - 1.4) +
                   np.random.normal(scale=1.0, size=n_points) *
                   (np.abs(6.5 - class0_groupA_x) + 2) / 6 - 0.3 * class0_groupA_x + 0.5)

# Class 0, Hospital B (triangles)
class0_groupB_x = np.clip(skewnorm.rvs(3, loc=6, scale=3, size=n_points), 0, 12.5)
class0_groupB_y = (3 * np.sin(0.5 * class0_groupB_x - 1.4) +
                   np.random.normal(scale=1.0, size=n_points) *
                   (np.abs(6.5 - class0_groupB_x) + 1) / 4 - 0.15 * (12 - class0_groupB_x))

# Class 1, Hospital A (circles)
class1_groupA_x = np.clip(skewnorm.rvs(-3, loc=6, scale=3, size=n_points), 0, 12.5)
class1_groupA_y = (3 * np.sin(0.5 * class1_groupA_x - 1.4) +
                   1.5 + np.random.normal(scale=1.0, size=n_points) *
                   (np.abs(6.5 - class1_groupA_x) + 1) / 4 +
                   0.15 * class1_groupA_x)

# Class 1, Hospital B (triangles)
class1_groupB_x = np.clip(skewnorm.rvs(3, loc=6, scale=3, size=n_points), 0, 12.5)
class1_groupB_y = 3 * np.sin(0.5 * class1_groupB_x - 1.4) + 1.5 + np.random.normal(scale=1.0, size=n_points) - 0.1 * class1_groupB_x + 1.2

# Combine X values for Hospital A and B (for density plot)
hospitalA_x = np.concatenate([class0_groupA_x, class1_groupA_x])
hospitalA_y = np.concatenate([class0_groupA_y, class1_groupA_y])
hospitalB_x = np.concatenate([class0_groupB_x, class1_groupB_x])

# Define the sinusoidal function to fit
def sinusoidal_model(x, a, b, c):
    return a * np.sin(b * x + c)

# Fit the sinusoidal model to the data from Set A (Hospital A)
params, _ = curve_fit(sinusoidal_model, hospitalA_x, hospitalA_y, p0=[3, 0.5, -1.4])

# Generate fitted values for the model
x_fitted = np.linspace(0, 12.65, 100)
y_fitted = sinusoidal_model(x_fitted, *params)

# Define the decision boundary (for comparison purposes)
def sinusoidal_boundary(x):
    return 3 * np.sin(0.5 * x - 1.4) + 0.75

x_boundary = np.linspace(0, 12.65, 100)
y_boundary = sinusoidal_boundary(x_boundary)

def plot_figure(class0_x, class0_y, class1_x, class1_y, hospitalA_x, hospitalB_x, figure_name):
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(10, 4, hspace=0.3, wspace=0.3)

    # Subplot for scatter plot
    ax_main = fig.add_subplot(gs[0:9, 0:4])

    # Subplot for KDE of Hospital A and B
    ax_bottom = fig.add_subplot(gs[9, 0:4], sharex=ax_main)

    # Scatter plot for Class 0
    ax_main.scatter(class0_x[0], class0_y[0], marker='o', color=hospital_a_color, s=60, alpha=0.5)
    if class0_x[1] is not None:
        ax_main.scatter(class0_x[1], class0_y[1], marker='o', color=hospital_b_color, s=60, alpha=0.5)

    # Scatter plot for Class 1
    ax_main.scatter(class1_x[0], class1_y[0], marker='^', color=hospital_a_color, s=60, alpha=0.75, facecolors='none')
    if class1_x[1] is not None:
        ax_main.scatter(class1_x[1], class1_y[1], marker='^', color=hospital_b_color, s=60, alpha=0.75, facecolors='none')

    # # Plot the decision boundary
    # ax_main.plot(x_boundary, y_boundary, color='black', linestyle='--', linewidth=3)

    # Plot the fitted sinusoidal model from Set A
    ax_main.plot(x_fitted, y_fitted, color='black', linestyle='--', linewidth=3, label='Fitted Model (Set A)')

    # Set axis limits
    ax_main.set_ylim([-4, 6.5])
    ax_main.set_xlim([-1, 13])

    # KDE plots
    sns.kdeplot(hospitalA_x, color=hospital_a_color, fill=True, ax=ax_bottom, label='Set A')
    if hospitalB_x is not None:
        sns.kdeplot(hospitalB_x, color=hospital_b_color, fill=True, ax=ax_bottom, label='Set B')

    # Hide tick labels
    ax_main.set_xticklabels([])
    ax_main.set_yticklabels([])
    ax_bottom.set_yticklabels([])
    ax_main.xaxis.set_ticks_position('none')
    ax_main.yaxis.set_ticks_position('none')
    ax_bottom.xaxis.set_ticks_position('none')
    ax_bottom.yaxis.set_ticks_position('none')

    ax_main.set_ylabel('Y', size=20, weight='bold')
    ax_bottom.set_xlabel('X', size=20, weight='bold')

    ax_main.spines[['right', 'top']].set_visible(False)
    for axis in ['bottom', 'left']:
        ax_main.spines[axis].set_linewidth(4)
    ax_bottom.spines[['right', 'top', 'left']].set_visible(False)
    ax_bottom.set_ylabel('')

    # Custom legend handles
    class0_handle = Line2D([0], [0], marker='o', color='k', label='Class 0', markersize=10, linestyle='None')
    class1_handle = Line2D([0], [0], marker='^', color='k', label='Class 1', markersize=10, linestyle='None', markerfacecolor='none')

    # Combine legends
    handles_bottom, labels_bottom = ax_bottom.get_legend_handles_labels()
    combined_handles = [class0_handle, class1_handle] + handles_bottom
    combined_labels = ['Class 0', 'Class 1'] + labels_bottom
    fig.legend(combined_handles, combined_labels, loc='upper right', bbox_to_anchor=(0.9, 0.9), frameon=False, prop={'size': 15})

    plt.savefig(f"figures/{figure_name}.svg")
    # plt.show()

# Plot both figures
plot_figure(
    [class0_groupA_x, class0_groupB_x], [class0_groupA_y, class0_groupB_y],
    [class1_groupA_x, class1_groupB_x], [class1_groupA_y, class1_groupB_y],
    hospitalA_x, hospitalB_x, "shift_concept"
)

# Second figure with only Set A
plot_figure(
    [class0_groupA_x, None], [class0_groupA_y, None],
    [class1_groupA_x, None], [class1_groupA_y, None],
    hospitalA_x, None, "shift_concept_setA_only"
)
