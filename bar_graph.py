import plotly.express as px
import pandas as pd

#import psutil
import plotly.io as pio

#pio.renderers.default = 'png'

# df = pd.read_csv('https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')
#
# df.drop(df.columns.difference(['Age_Group','Product_Category','Order_Quantity', 'State']), 1, inplace=True)
# grouped_df = df.groupby(["State", "Age_Group", "Product_Category"])
# grouped_df = pd.DataFrame(grouped_df.sum().reset_index())
#
# df = df[df["State"] == "New South Wales"]
#
# # Plotly Express
# fig = px.bar(
#         data_frame=df,
#         x='Age_Group',
#         y='Order_Quantity',
#         color='Product_Category',
#         opacity=0.6,
#         category_orders={"Age_Group": ["Youth (<25)", "Young Adults (25-34)", "Adults (35-64)", "Seniors (64+)"],
#                                 "Product_Category": ["Clothing", "Bikes", "Accessories"]},
#         color_discrete_map={
#                  'Accessories':'#0059b2',
#                  'Bikes': '#4ca6ff',
#                  'Clothing': '#99ccff'},
#         hover_data={'Order_Quantity':':,.0f'},
#         labels={'Age_Group':'<b>Age group</b>',
#                      'Order_Quantity':'<b>Order quantity</b>',
#                      'Product_Category':'<b>Product category</b>'})
#
# fig.update_layout(
#     width=700,
#     height=500,
#     plot_bgcolor='rgba(0,0,0,0)',
#     legend_traceorder="reversed",
#     legend=dict(yanchor="top",
#                 orientation="h",
#                 bordercolor="Black",
#                 borderwidth=1.5),
#     xaxis=dict(mirror=True, ticks='outside', showline=True, linewidth=1.5, linecolor='black'),
#     yaxis=dict(mirror=True, ticks='outside', showline=True),
#     margin=dict(l=20, r=20, t=30, b=20),
#     title_x=0.53,
#     font=dict(family="Helvetica Neue,  sans-serif"),
#     hoverlabel=dict(
#         bgcolor="white",
#         font_family="Helvetica Neue,  sans-serif"
#         )
# )
#
# fig.show()

df = pd.read_csv('medical supplies.csv')
print(df.dtypes)
