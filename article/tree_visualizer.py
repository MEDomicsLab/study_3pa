import os
import re
from jinja2 import Environment, FileSystemLoader

from result_parameters import file_parameters, text_colors


class Visualizer:
    def __init__(self, results):
        self.save_folder = 'figures'
        self.config = results['loadedFiles']['infoConfig']
        if 'reference' in results['loadedFiles']:
            self.ref_profile = results['loadedFiles']['reference']['profiles']
        else:
            self.ref_profile = None
        self.test_profile = results['loadedFiles']['test']['profiles']
        self.template_folder = "tree_template"
        for key in file_parameters.keys():
            if results['filename'].startswith(key):
                self.experiment_name = 'figures/' + file_parameters[key]['title']
                self.profile_metrics = file_parameters[key]['profile_metrics']
                self.profile_depth = file_parameters[key]['profile_depth']
                self.dr = file_parameters[key]['dr']
                self.threshold = file_parameters[key]['threshold']
                self.profiles_highlights = file_parameters[key]['profiles_highlights']
                break

    def read_tree_section(self, samp_ratio, dr, data_set):
        """Retrieve nodes based on user-defined sample ratio and data ratio."""
        if data_set == "reference":
            tree_profiles = self.ref_profile
        elif data_set == "test":
            tree_profiles = self.test_profile
        else:
            raise ValueError("The 'data_set' parameter must be either 'reference' or 'test'.")

        try:
            profiles_to_visualize = tree_profiles[str(samp_ratio)][str(dr)]
        except KeyError:
            raise ValueError(f"No nodes found for samp_ratio={samp_ratio} and dr={dr}.")

        # Round condition values
        for profile in profiles_to_visualize:
            profile['path'] = [re.sub(r'(?<!\w)(\d+\.\d+|\d+)(?!\w)',
                                      lambda m: str(round(float(m.group()), 1)), s) for s in profile['path']]
            # Rename Auc to AUC
            if 'Auc' in profile['metrics']:
                profile['metrics']['AUC'] = profile['metrics']['Auc']

        # Measure NB
        if self.threshold is not None:
            for profile in profiles_to_visualize:
                w = self.threshold / (1 - self.threshold)
                prevalence = profile['node information']['Positive%'] / 100
                sensitivity = profile['metrics']['Sensitivity']
                specificity = profile['metrics']['Specificity']
                if not any(value is None for value in [w, prevalence, sensitivity, specificity]):
                    nb = sensitivity * prevalence - (1 - specificity) * (1 - prevalence) * w
                    profile['metrics']['NB'] = nb
                else:
                    profile['metrics']['NB'] = None

        if dr != 100:  # Get results of DR=100% to compare each metrics
            original_profiles = self.read_tree_section(samp_ratio, 100, data_set)
            for profile in profiles_to_visualize:
                original_profile_metrics = original_profiles[profile['id'] - 1]['metrics']
                for original_metric_name, original_metric_value in original_profile_metrics.items():
                    if profile['metrics'][original_metric_name] is None:
                        profile['metrics'][f'diff_{original_metric_name}'] = None
                    else:
                        profile['metrics'][f'diff_{original_metric_name}'] = (profile['metrics'][original_metric_name] -
                                                                              original_metric_value)

        print(f"Nodes successfully loaded for samp_ratio={samp_ratio}, dr={dr}")
        return profiles_to_visualize

    def add_color_highlights(self, profiles):
        if len(self.profiles_highlights) == 0:
            return profiles
        first_profile = profiles[0]
        for profile in profiles:
            if profile['id'] in self.profiles_highlights:
                profile['highlight'] = 1
                profile['text_color'] = {}
                for metric_name, metric_value in profile['metrics'].items():
                    delta_metric_value = metric_value - first_profile['metrics'][metric_name]
                    if delta_metric_value < 0:
                        profile['text_color'][metric_name] = text_colors['less']
                    elif delta_metric_value > 0:
                        profile['text_color'][metric_name] = text_colors['greater']
                    else:
                        profile['text_color'][metric_name] = text_colors['equal']
        return profiles

    def generate_tree_html(self, samp_ratio, dr, data_set, metrics_list=None, max_depth=None):
        """Generate the tree visualization HTML."""
        env = Environment(loader=FileSystemLoader(self.template_folder))
        template = env.get_template('tree.html')

        # Read the profiles for the specified data_set
        profiles_to_visualize = self.read_tree_section(samp_ratio=samp_ratio, dr=dr, data_set=data_set)

        # Add colors to text for profiles to highlight
        profiles_to_visualize = self.add_color_highlights(profiles_to_visualize)

        if metrics_list is not None:
            for profile in profiles_to_visualize:
                all_metrics = profile['metrics']
                profile['metrics'] = {key: all_metrics.get(key) for key in metrics_list if key in all_metrics.keys()}

        if max_depth is not None:
            profiles_to_visualize = [item for item in profiles_to_visualize if len(item['path']) <= max_depth]

        # Get the absolute path to the template folder
        base_path = self.template_folder

        # Render the HTML with the list of nodes and base path
        rendered_html = template.render(
            nodes=profiles_to_visualize,
            base_path=base_path
        )

        # Determine output path
        if data_set == "reference":
            output_path = os.path.join(self.experiment_name, f"tree_visualization_reference_{samp_ratio}_{dr}.html")
        elif data_set == "test":
            output_path = os.path.join(self.experiment_name, f"tree_visualization_test_{samp_ratio}_{dr}.html")
        else:
            raise ValueError("The 'data_set' parameter must be either 'reference' or 'test'.")

        # Save the HTML
        # Ensure the directory exists
        os.makedirs(self.experiment_name, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(rendered_html)

        print(f"Tree visualization generated: {output_path}")

    def visualize(self, samp_ratio, data_set, dr=None, metrics_list=None, max_depth=None):
        """Main method to run the visualization pipeline."""
        if metrics_list is None:
            metrics_list = self.profile_metrics
        if max_depth is None:
            max_depth = self.profile_depth
        if dr is None:
            if self.dr != 100:
                self.generate_tree_html(samp_ratio, dr=self.dr, data_set=data_set, metrics_list=metrics_list,
                                        max_depth=max_depth)
                dr = 100
        self.generate_tree_html(samp_ratio, dr, data_set=data_set, metrics_list=metrics_list, max_depth=max_depth)
