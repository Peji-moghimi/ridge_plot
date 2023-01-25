# RidgePlot.py
from typing import Union, Optional
import pandas as pd

def ridge_plot(df: Union[str, pd.DataFrame], 
               cat_col: str, 
               num_col: str, 
               out: Optional[str]=None, 
               xlabel: Optional[str]=None, 
               option_col: Optional[str]=None, 
               palette: Optional[str]="coolwarm", 
               title: Optional[str]=None, 
               sort_by: Optional[dict]=None, 
               show: Optional[bool]=True):
    """
    Create a ridge plot of a dataset, and returns the plot object.
    
    Parameters:
    - df (pandas.DataFrame or str): pandas.DataFrame or path to the .csv file for the dataframe (when running as a script).
    - cat_col (str): Name of the categorical column (plot rows).
    - num_col (str): Name of the numerical column (for infering KDE distributions).
    - out (str, optional): Output file path. Default is None.
    - xlabel (str, optional): Label for the x-axis. Defaults to num_col.
    - option_col (str, optional): Name of an optional column to use for colouring the KDE distributions. 
      Default behaviour is colouring distributions by the relative means. Default is None.
    - palette (str, optional): seaborn/matplotlib colour palette for option_col. Default is coolwarm.
    - title (str, optional): Title for the plot. Default is None.
    - sort_by (str, optional): Dict object mapping featues of cat_col to ascending integers. 
      Default behaviour is ordering cat_col features by their corresponding num_col(mean - variance) values.
      Default is None.
    - show (bool, optional): Display the plot or not. 
      Recommended to set to False if using seaborn/matplotlib functions/methods on the returned object. Default is True.
      
    Returns:
    - matplotlib.pyplot : The plot object
    """
    
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    if isinstance(df,str):
        if not df.endswith((".csv",".tsv")):
            raise ValueError(f"The input argument df should be a path to a csv or tsv file, but got {df}")
        elif df.endswith(".tsv"):
            df = pd.read_csv(df, sep="\t")
        else:
            df = pd.read_csv(df)
    elif not isinstance(df, pd.DataFrame):
        raise ValueError(f"The input argument df should be either a path to a csv or tsv file or a pandas Dataframe, but got {df}")
    
    if not sort_by:
        # defining a dictionnary of the categrical features in cat_col, ordered by their mean - variance
        df['mean_cat_col'] = df.groupby(cat_col)[num_col].transform('mean')
        df['var_cat_col'] = df.groupby(cat_col)[num_col].transform('var')
        df['sort_by'] = df['mean_cat_col']/df['var_cat_col']
        sort_by = df.groupby(cat_col, as_index=False)['sort_by'].mean().reset_index().sort_values('sort_by')[cat_col].to_dict()

    if not option_col:
        # df column for mean of each feature in `cat_col` (to be used later in the FacetGrid plot).
        option_col = "mean"
        option_col_series = df.groupby(cat_col)[num_col].mean()
        df[option_col] = df[cat_col].map(option_col_series)
    # generate a color palette with Seaborn.color_palette()
    pal = sns.color_palette(palette=palette, n_colors=len(sort_by))

    # option_col, if provided, is passed to the 'hue' argument, which will be represented by colors with `palette`
    # NOTE: all option_col values, corresponding to one feature of cat_col, muct be identical.  
    g = sns.FacetGrid(df, row=cat_col, hue=option_col, aspect=15, height=0.75, palette=pal)
    # then we add the densities kdeplots for each cat_col feature.
    g.map(sns.kdeplot, num_col,
          bw_adjust=1, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)

    # here we add a white line that represents the contour of each kdeplot
    g.map(sns.kdeplot, num_col, 
          bw_adjust=1, clip_on=False, 
          color="w", lw=2)

    # here we add a horizontal line for each plot
    g.map(plt.axhline, y=0,
          lw=2, clip_on=False)

    # we loop over the FacetGrid figure axes (g.axes.flat) and add the cat_col feature as text with the right color
    # notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
    for i, ax in enumerate(g.axes.flat):
        ax.text(-15, 0.02, sort_by[i],
                fontweight='bold', fontsize=15,
                color=ax.lines[-1].get_color())

    # we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
    g.fig.subplots_adjust(hspace=-0.3)

    # eventually we remove axes titles, yticks and spines
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
    if not xlabel:
        xlabel = num_col
    plt.xlabel(xlabel, fontweight='bold', fontsize=15)
    if not title:
        title = f"Ridge plot of `{num_col}` distributions for features in `{cat_col}`."
    g.fig.suptitle(title,
                   ha='center',
                   fontsize=20,
                   fontweight=20)

    if out:
        plt.savefig(out, dpi=400)
    if show:
        plt.show()

    return g

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Create a ridge plot of a dataset.")

    parser.add_argument("df", type=str, help="input dataframe file path (currently only accepts .csv/.tsv formats).")
    parser.add_argument("cat_col", type=str, help="Name of the categorical column (rows of the plot).")
    parser.add_argument("num_col", type=str, help="Name of the numerical column (used for KDE distributions).")
    parser.add_argument("out", type=str, help="Output file path.")
    parser.add_argument("--xlabel", metavar="-x", default=None, type=str, help="Label for the x-axis.")
    parser.add_argument("--option_col", metavar="-cc", type=str, default=None, help="Name of an optional column to use for colouring the distributions.")
    parser.add_argument("--palette", metavar="-p", type=str, default='coolwarm', 
                        help="matplotlib/seaborn colour palette.")
    parser.add_argument("--title", metavar="-t", type=str, default=None, help="Title for the plot.")
    parser.add_argument("--sort_by", metavar="-sb", type=dict, default=None, help="Sort the rows by this dictionary.")

    args = parser.parse_args()
    
    ridge_plot(args.df, args.cat_col, args.num_col, args.out, args.xlabel, args.option_col, args.palette, args.title, args.sort_by)