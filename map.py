import pandas as pd
import plotly.figure_factory as ff


df = pd.read_csv(
    'https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')

selected_variable = ['Age_Group', 'Customer_Gender', 'Year', 'Profit']

data = df.loc[(df["Country"] == "France") & (df["State"] == "Loiret")]

data.drop(data.columns.difference(['Age_Group', 'Customer_Gender', 'Year', 'Profit', 'Revenue']), 1, inplace=True)

data = data.groupby(['Year', 'Age_Group', 'Customer_Gender']).agg('sum')

data.reset_index(inplace=True)

data = data.apply(lambda x: pd.factorize(x)[0] if x.name in ['Age_Group', 'Customer_Gender', 'Year'] else x).corr(method='pearson', min_periods=1)

figure = ff.create_annotated_heatmap(
    z=data.values,
    x=list(data.columns),
    y=list(data.index),
    annotation_text=data.round(2).values,
    showscale=True
)

figure['layout']['xaxis'].update(side='bottom')

figure.show()