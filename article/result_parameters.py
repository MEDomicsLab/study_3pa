"""

"""

# Parameters
file_parameters = {'poym':
                       {'title': 'POYM',
                        'show_legend': True,
                        'linewidth': 4,
                        'rcParams': {'font.family': ['serif'],
                                     'font.size': 15,
                                     'mathtext.fontset': 'dejavuserif'},
                        'profile_metrics': ['Specificity', 'Sensitivity', 'NPV', 'PPV', 'AUC'],  # , 'diff_AUC'
                        'profile_depth': 3,
                        'profiles_highlights': [91, 34],
                        'dr': 95,
                        'threshold': 0.1235550345060529
                        },
                   'iv': {'title': 'Internal Validation',
                          'show_legend': False,
                          'linewidth': 3,
                          'rcParams': {'font.family': ['serif'],
                                       'font.size': 15},
                          'profile_metrics': ['Specificity', 'Sensitivity', 'NPV', 'PPV', 'AUC', 'diff_NB'],
                          'profile_depth': 2,
                          'profiles_highlights': [2, 59],
                          'dr': 93,
                          'threshold': 0.19050934303395273
                          },
                   'tv': {'title': 'Temporal Validation',
                          'show_legend': False,
                          'linewidth': 3,
                          'rcParams': {'font.family': ['serif'],
                                       'font.size': 15},
                          'profile_metrics': ['Specificity', 'Sensitivity', 'NPV', 'PPV', 'AUC'],
                          'profile_depth': 3,
                          'profiles_highlights': [2, 64],
                          'dr': 91,
                          'threshold': 0.19050934303395273
                          },
                   'ev199': {'title': 'External Validation 199',
                             'show_legend': False,
                             'linewidth': 4,
                             'rcParams': {'font.family': ['serif'],
                                          'font.size': 15},
                             'profile_metrics': ['Specificity', 'Sensitivity', 'NPV', 'PPV', 'AUC'],
                             'profile_depth': 4,
                             'profiles_highlights': [],
                             'dr': 94,
                             'threshold': 0.19050934303395273
                             },
                   'ev188': {'title': 'External Validation 188',
                             'show_legend': False,
                             'linewidth': 4,
                             'rcParams': {'font.family': ['serif'],
                                          'font.size': 15},
                             'profile_metrics': ['Specificity', 'Sensitivity', 'NPV', 'PPV', 'AUC'],
                             'profile_depth': 3,
                             'profiles_highlights': [],
                             'dr': 89,
                             'threshold': 0.19050934303395273
                             },
                   'ev122': {'title': 'External Validation 122',
                             'show_legend': False,
                             'linewidth': 4,
                             'rcParams': {'font.family': ['serif'],
                                          'font.size': 15},
                             'profile_metrics': ['Specificity', 'Sensitivity', 'NPV', 'PPV', 'AUC'],
                             'profile_depth': 2,
                             'profiles_highlights': [],
                             'dr': 95,
                             'threshold': 0.19050934303395273
                             },
                   'ev167': {'title': 'External Validation 167',
                             'show_legend': False,
                             'linewidth': 4,
                             'rcParams': {'font.family': ['serif'],
                                          'font.size': 15},
                             'profile_metrics': ['Specificity', 'Sensitivity', 'NPV', 'PPV', 'AUC'],
                             'profile_depth': 2,
                             'profiles_highlights': [],
                             'dr': 83,
                             'threshold': 0.19050934303395273
                             }}

text_colors = {'less': 'red',
               'equal': 'black',
               'greater': "rgb(85, 107, 47)"}

metrics_parameters = {'Specificity': (0, 0, 1),  # (0.35, 0.70, 0.90)
                      'Sensitivity': (1, 0, 0),
                      'NPV': (1, 0.647, 0),
                      'PPV': (0, 0.60, 0.50),
                      'Auc': (0.5, 0.5, 0.5),  # (0.95, 0.90, 0.25)
                      # 'NB': 'green'
                      }

combined_results_parameters = {'show_legend': True,
                               'linewidth': 3,
                               'marker': None,
                               'rcParams': {'font.family': ['serif'],
                                            'font.size': 15,
                                            'mathtext.fontset': 'dejavuserif'}
                               }
