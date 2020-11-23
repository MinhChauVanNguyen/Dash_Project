import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import plotly.express as px
import plotly.io as pio

df = pd.read_csv(
    'https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')

selected_variable = ['Age_Group', 'Customer_Gender', 'Year', 'Profit']

data = df.loc[(df["Country"] == "France") & (df["State"] == "Loiret")]

data.drop(data.columns.difference(['State', 'Age_Group', 'Customer_Gender', 'Year', 'Profit', 'Revenue']), 1,
          inplace=True)

data = data.groupby(['Year', 'Age_Group', 'Customer_Gender']).agg('sum')
data.reset_index(inplace=True)

for c in data.columns[:-2]:
    data[c] = LabelEncoder().fit_transform(data[c].values)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

poly_reg = PolynomialFeatures(degree=2)
X_train_poly = poly_reg.fit_transform(X_train)

regressor = LinearRegression()

regressor.fit(X_train_poly, y_train)

y_pred = regressor.predict(poly_reg.fit_transform(X_test))

y_pred_df = pd.DataFrame(y_pred, columns=['Y predict'])
y_test_df = pd.DataFrame(y_test, columns=['Y test'])
frames = [y_pred_df, y_test_df]
results = pd.concat(frames, axis=1)

results['Y_pred'] = results['Y predict'].map('{:,.0f}'.format)
results['Y_test'] = results['Y test'].map('{:,}'.format)

results['text'] = 'Y test : ' + results['Y_test'].astype(str) + '<br>' + \
                  'Y predict : ' + results['Y_pred'].astype(str)

print(results)

annotations = []
for i, row in results.iterrows():
    annotations.append(
        dict(x=row["Y test"],
             y=row["Y predict"],
             text=row["text"],
             xref="x",
             yref="y",
             showarrow=True,
             bordercolor='pink',
             borderpad=4,
             ax=20,
             ay=-30,
             align="right",
             bgcolor="#abd7eb",
             opacity=0.8
             )
    )

fig = px.scatter(results, x="Y test", y="Y predict")

fig.update_traces(
    textposition='top right',
)

fig.update_layout(
    annotations=annotations
)
fig.show()

# labels = {}
# labels["Variable"] = []
#
# for v in range(len(selected_variable)):
#     labels["Variable"].append("<b>" + selected_variable[v] + "</b>")
#     if "_" in selected_variable[v]:
#         labels["Variable"][v] = labels["Variable"][v].replace("_", " ")
#
# print(labels["Variable"])
