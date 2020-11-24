import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from Data import data_processing
from sklearn.metrics import r2_score, mean_squared_error


def update_data(country, state, variable, model, table=False):
    df = data_processing.df

    data = df.loc[(df["Country"] == country) & (df["State"] == state)]

    # add Revenue
    variable.append('Revenue')

    data = data[variable]

    # remove Revenue
    variable.pop()

    ind_variable = variable.copy()

    try:
        ind_variable = variable.copy()
        ind_variable.remove("Profit")
    except ValueError:
        pass
    finally:
        data = data.groupby(ind_variable).agg('sum')

    data.reset_index(inplace=True)

    heatmap_data = data.copy()

    # Encode categorical data
    for i in range(len(variable)):
        if variable[i] != "Profit":
            data[variable[i]] = LabelEncoder().fit_transform(data[variable[i]].values)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Fit the model on the train set
    poly_reg = PolynomialFeatures(degree=2)
    X_train_poly = poly_reg.fit_transform(X_train)

    if model == "SVM":
        regressor = SVR(kernel='linear', gamma=1e-8)
    elif model == "Decision":
        regressor = DecisionTreeRegressor(random_state=0)
    elif model == "Random":
        regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    else:
        regressor = LinearRegression()

    if model == "Poly":
        regressor.fit(X_train_poly, y_train)
        y_pred = regressor.predict(poly_reg.fit_transform(X_test))
    else:
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred, columns=['Y_pred'])
    y_test_df = pd.DataFrame(y_test, columns=['Y_test'])
    frames = [y_pred_df, y_test_df]
    results = pd.concat(frames, axis=1)

    if not table:
        results['Y predict'] = results['Y_pred'].map('{:,.0f}'.format)
        results['Y test'] = results['Y_test'].map('{:,}'.format)
    else:
        results['Y_pred'] = results['Y_pred'].map('{:,.0f}'.format)
        results['Y_test'] = results['Y_test'].map('{:,}'.format)

    if model == "Linear":
        importance = [round(num, 3) for num in regressor.coef_]
    elif model == "Poly":
        importance = np.delete(regressor.coef_, 0)
        importance = importance[0:len(variable)]
    elif model == "Decision" or model == "Random":
        importance = [round(num, 3) for num in regressor.feature_importances_]
    else:
        importance = list(regressor.coef_.flatten())

    return heatmap_data, results, importance

# @app.callback(
#     [Output(component_id='scatter_id', component_property='style'),
#      Output(component_id='table_id', component_property='style'),
#      Output(component_id='heatmap', component_property='figure'),
#      Output(component_id='scatter', component_property='figure'),
#      Output(component_id='feature_imp', component_property='figure'),
#      Output(component_id='table_two', component_property='data'),
#      Output(component_id='table_two', component_property='columns')
#      ],
#      #Output(component_id='r2', component_property='children')
#     [Input(component_id='slct_country', component_property='value'),
#      Input(component_id='slct_state', component_property='value'),
#      Input(component_id='slct_variable', component_property='value'),
#      Input(component_id='slct_model', component_property='value'),
#      Input(component_id='slct_output', component_property='value')
#      ]
# )
# def output_predict(selected_country, selected_state, selected_variable, selected_model, selected_radio):
#
#     regression_output = update_data(
#         country=selected_country,
#         state=selected_state,
#         variable=selected_variable,
#         model=selected_model,
#         table=False)
#
#     # HEATMAP
#     heatmap_data = regression_output[0]
#
#     heatmap_data = heatmap_data.apply(
#         lambda x: pd.factorize(x)[0] if x.name in ['Age_Group', 'Customer_Gender', 'Year'] else x).corr(
#         method='pearson', min_periods=1)
#
#     heatmap = ff.create_annotated_heatmap(
#         z=heatmap_data.values,
#         x=list(heatmap_data.columns),
#         y=list(heatmap_data.index),
#         annotation_text=heatmap_data.round(2).values,
#         showscale=True
#     )
#
#     heatmap['layout']['xaxis'].update(side='bottom')
#
#     heatmap.update_traces(dict(showscale=False))
#
#     # SCATTER PLOT
#     results = regression_output[1]
#
#     results['text'] = 'Y test : ' + results['Y test'].astype(str) + '<br>' + \
#                       'Y predict : ' + results['Y predict'].astype(str)
#
#     annotation = []
#
#     for i, row in results.iterrows():
#         annotation.append(
#             dict(x=row["Y test"],
#                  y=row["Y predict"],
#                  text=row["text"],
#                  xref="x",
#                  yref="y",
#                  showarrow=True,
#                  bordercolor='pink',
#                  borderpad=4,
#                  ax=20,
#                  ay=-30,
#                  align="right",
#                  bgcolor="#abd7eb",
#                  opacity=0.8
#                  )
#         )
#
#     scatter_plt = px.scatter(results, x="Y test", y="Y predict")
#
#     scatter_plt.update_traces(textposition='top right')
#
#     ## BREAKS THE WHOLE DASH
#     # scatter_plt.update_layout(
#     #     annotations=annotation
#     # )
#
#     # FEATURE IMPORTANCE
#     importance = regression_output[2]
#
#     feat_imp = pd.DataFrame({
#         'Variable': [x for x in selected_variable],
#         'Importance': importance
#     })
#
#     feat_imp["Indicator"] = np.where(feat_imp["Importance"] < 0, 'Negative', 'Positive')
#
#     plot_imp = px.bar(feat_imp,
#                       x='Variable',
#                       y='Importance',
#                       color='Indicator',
#                       title='Coefficients as Feature Importance'
#                       if selected_model == 'Linear' or selected_model == 'Poly' or selected_model == 'SVM'
#                       else 'Decision Trees Feature Importance'
#                       )
#
#     plot_imp.update_layout(title_x=0.5)
#
#     plot_imp.update_xaxes(tickfont=dict(size=11),
#                           ticktext=['<b>Year</b>', '<b>Age Group</b>', '<b>Customer Gender</b>', '<b>Profit</b>'],
#                           tickvals=selected_variable
#                           )
#
#     # r2 = r2_score(y_test.tolist(), y_pred.tolist())
#     # adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) /
#     #                           (X_train.shape[0] - X_train.shape[1] - 1)))
#     # metrics = u'R\u00b2' + ': {}'.format(r2) + ', adjusted ' + u'R\u00b2' + ': {}'.format(adj_r2)
#
#     # mse = mean_squared_error(y_test.tolist(), y_pred.tolist(), squared=False)
#     # rmse = mean_squared_error(y_test.tolist(), y_pred.tolist(), squared=False)
#
#     # Table
#     result_table = update_data(
#         country=selected_country,
#         state=selected_state,
#         variable=selected_variable,
#         model=selected_model,
#         table=True)[1]
#
#     columns = [{'name': col, 'id': col} for col in result_table.columns]
#     table_data = result_table.to_dict(orient='records')
#
#     if selected_radio == 'Table':
#         return {'display': 'none'}, {'display': 'block'}, heatmap, scatter_plt, plot_imp, table_data, columns
#     if selected_radio == 'Graph':
#         return {'display': 'block'}, {'display': 'none'}, heatmap, scatter_plt, plot_imp, table_data, columns



# labels = {}
# labels["Variable"] = []
#
# for v in range(len(selected_variable)):
#     labels["Variable"].append("<b>" + selected_variable[v] + "</b>")
#     if "_" in selected_variable[v]:
#         labels["Variable"][v] = labels["Variable"][v].replace("_", " ")
#
# print(labels["Variable"])
