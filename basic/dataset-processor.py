# Rewrite the plotting script to use sklearn datasets
#
# Ref. http://www.neural.cz/dataset-exploration-boston-house-pricing.html

import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_boston

plots_dir  = 'plots'
plots_format = 'png'

dataset = load_boston()
housing_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

os.makedirs(plots_dir, exist_ok=True)


for feature1_idx, feature1_name in enumerate(housing_df.columns):
   for feature2_idx, feature2_name in enumerate(housing_df.columns):
      for feature3_idx, feature3_name in enumerate(housing_df.columns):
         if (feature1_idx < feature2_idx) and (feature2_idx < feature3_idx):
            print ('Generating 3d scatter plot with ' + feature1_name + ' + ' + feature2_name + ' + ' + feature3_name)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(housing_df[feature1_name], housing_df[feature2_name], housing_df[feature3_name])
            ax.set_title('3D scatter plot')
            ax.set_xlabel(feature1_name)
            ax.set_ylabel(feature2_name)
            ax.set_zlabel(feature3_name)
#            ax.legend()
            plots_file = plots_dir + '/3d_scatter_' + feature1_name + '_' + feature2_name + '_' +  feature3_name + '.' + plots_format
            plt.savefig(plots_file, format=plots_format)
            plt.clf()
            plt.close()

