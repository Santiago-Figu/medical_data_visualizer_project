import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data from medical_examination.csv and assign it to the df variable.
path_file = os.path.join(os.getcwd(),'data','medical_examination.csv')
n = 6

df = pd.read_csv(filepath_or_buffer=path_file, delimiter=',')

print(f"df{df.head(n)}")

# 2. Add an overweight column to the data. To determine if a person is overweight, 
# first calculate their BMI by dividing their weight in kilograms by the square of their height in meters.
#  If that value is > 25 then the person is overweight. 
# Use the value 0 for NOT overweight and the value 1 for overweight.

df['bmi'] = (round(df['weight'] / ((df['height']/100)**2),2))

df['overweight'] = (df['bmi'] > 25).astype(int)

df.drop('bmi', axis=1, inplace=True)

# print(f"df{df.head(n)}")
# 3. Normalize data by making 0 always good and 1 always bad. 
# If the value of cholesterol or gluc is 1, set the value to 0. 
# If the value is more than 1, set the value to 1.

df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)

df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)





# Draw the Categorical Plot in the draw_cat_plot function.

def draw_cat_plot():
    print(f"df{df.head(n)}")
    # 5. Create a DataFrame for the cat plot using pd.melt with values from cholesterol, 
    # gluc, smoke, alco, active, and overweight in the df_cat variable.

    id_vars = ['cardio']  # (optional, for analysis by value, e.g. disease: cardio)
    value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']

    # Crear df_cat con pd.melt
    df_cat = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='variable',      # Name of the column that will contain the variables.
        value_name='value'        # Name of the column that will contain the values.
    )
    
    print(df_cat)

    # 6. Group and reformat the data in df_cat to split it by cardio. 
    # Show the counts of each feature.
    # You will have to rename one of the columns for the catplot to work correctly.

    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # df_cat = df_cat.rename(columns={'value': 'category'})

    # Resultado
    print(df_cat.head())
    # 7. Convert the data into long format and create a chart that shows the value counts of the categorical
    # features using the following method provided by the seaborn library import: sns.catplot().

    # Note: Part of this step is no longer necessary
    # because pd.melt already takes care of converting the data into long format.

    sns.set_theme(style="whitegrid")

    graph = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',  # "col" creates two separate graphs for cardio=0 and cardio=1
        kind='bar',
        height=4,
        aspect=1.5,
        palette={0: 'blue', 1: 'orange'}  # Colors for categories 0 and 1
    )

    
    graph.set_axis_labels("variable", "total")
    graph.set_titles("Cardio = {col_name}")
    # g.set_xticklabels(rotation=45)

    # 8. Get the figure for the output and store it in the fig variable.

    fig = graph.figure


    # 9. Do not modify the next two lines.
    fig.savefig('catplot.png')
    plt.close(fig)
    return fig


# 10 Draw the Heat Map in the draw_heat_map function.
def draw_heat_map():
    # 11. Clean the data in the df_heat variable by filtering out the following patient segments that
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Presión diastólica <= sistólica
        (df['height'] >= df['height'].quantile(0.025)) &  # Altura >= percentil 2.5
        (df['height'] <= df['height'].quantile(0.975)) &  # Altura <= percentil 97.5
        (df['weight'] >= df['weight'].quantile(0.025)) &  # Peso >= percentil 2.5
        (df['weight'] <= df['weight'].quantile(0.975))   # Peso <= percentil 97.5
    ].copy()

    # 12. Calculate the correlation matrix and store it in the corr variable.
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle and store it in the mask variable.
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure.
  
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15. Plot the correlation matrix using the method provided by the seaborn library import: sns.heatmap().

    ax = sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={
            "shrink": .5,
            "ticks": np.linspace(-0.2, 0.8, 6)
        },
        ax=ax
    )

    plt.title('Matriz de Correlación de Variables', pad=20)
    # plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 16. Do not modify the next two lines.
    # print("Generando imagen...")
    fig.savefig('heatmap.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    return fig
