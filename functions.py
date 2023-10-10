from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import openai

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

def readcsv(path):
    df = pd.read_csv(path)
    df['Semana do ano'] = round(df['Semana do ano'].div(100), 2)
    df['Taxa de conversão'] = round(df['Taxa de conversão'] * 100, 2)
    df['Quantidade'] = df['Quantidade'].astype(int)

    return df

def simples_aggrid(df):

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(enabled=False)
    gb.configure_default_column(editable=True)
    gridoptions = gb.build()
    df_grid = AgGrid(df, gridOptions=gridoptions, enable_enterprise_modules=True,
                     height=300, width='100%', key='grid', update_mode=GridUpdateMode.SELECTION_CHANGED)

    selected_rows = df_grid["data"]
    selected_rows = pd.DataFrame(selected_rows)

    return selected_rows


def varia_boxplot(df, agrupo, boxpoints, groupby):

    if groupby == 'Soma':
        df1_data1 = (df.groupby(['Data', agrupo]).sum()).reset_index()
    elif groupby == 'Média':
        df1_data1 = (df.groupby(['Data', agrupo]).mean()).reset_index()

    metrics1 = ['Receita', 'Transações', 'Taxa de conversão',
                'Sessões', 'Usuários','Novos usuários',
                'Páginas / sessão', 'Visualizações de página', 'Duração média da sessão',
                'Custo', 'Impressões', 'Cliques']

    fig0 = make_subplots(rows=4, cols=3, subplot_titles=metrics1, vertical_spacing=0.1, horizontal_spacing=0.08)

    for i, metric in enumerate(metrics1, 1):
        row = (i - 1) // 3 + 1
        col = (i - 1) % 3 + 1
        fig0.add_trace(
            go.Box(x=df1_data1[agrupo], y=df1_data1[metric],
                   name=metric, boxpoints=boxpoints),
            row=row,
            col=col
        )


    # Atualizar layout e exibir o gráfico
    fig0.update_layout(
        height=900, width=1000,
        font={'color': "#000000", 'family': "sans-serif"}, showlegend=False,
        margin=dict(l=50, r=50, b=50, t=90), autosize=False, hovermode="x unified")
    fig0.update_yaxes(
        title_font=dict(family='Sans-serif', size=18),
        tickfont=dict(family='Sans-serif', size=12), nticks=8, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3')

    fig0.update_xaxes(
        title_font=dict(family='Sans-serif', size=18),
        tickfont=dict(family='Sans-serif', size=12), nticks=10, showgrid=False)

    return fig0

def varia_boxplot_unique(df, agrupo, metrica, boxpoints, color, vermedia):

    df1_data1 = (df.groupby(['Data', agrupo])[metrica].sum().astype(int)).reset_index()

    fig0 = go.Figure()

    fig0.add_trace(
        go.Box(x=df1_data1[agrupo], y=df1_data1[metrica],
               name=metrica,
               boxpoints=boxpoints,
               marker_color=color)
    )
    if vermedia == 'Sim':
        media_metrica = df1_data1[metrica].median()
        # Adicionar linha com a média da métrica
        fig0.add_trace(
            go.Scatter(
                x=[df1_data1[agrupo].min(), df1_data1[agrupo].max()],
                y=[media_metrica, media_metrica],
                mode='lines',
                name='Mediana Geral', showlegend=False,
                line=dict(color='black', dash='dash')
            )
        )

    # Atualizar layout e exibir o gráfico
    fig0.update_layout(
        height=400,
        font={'color': "#000000", 'family': "sans-serif"}, showlegend=False,
        margin=dict(l=10, r=10, b=10, t=10), autosize=False, hovermode="x unified")
    fig0.update_yaxes(
        title_font=dict(family='Sans-serif', size=18),
        tickfont=dict(family='Sans-serif', size=12), nticks=8, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3')

    fig0.update_xaxes(
        title_font=dict(family='Sans-serif', size=18),
        tickfont=dict(family='Sans-serif', size=12), nticks=10, showgrid=False)

    return fig0






def area_norm(df, var0, metrica, var1, cor1, var2, cor2, var3, cor3, var4, cor4, var5, cor5):
    fig = go.Figure()
    df['Semana do ano'] = (df['Semana do ano'] * 100)

    df_grouped = df.groupby([var0, 'Mídia'])[metrica].sum().reset_index()


    df1 = df_grouped[(df_grouped['Mídia'] == var1)]
    values1 = df1[var0]
    y1 = df1[metrica]

    df2 = df_grouped[(df_grouped['Mídia'] == var2)]
    values2 = df2[var0]
    y2 = df2[metrica]

    df3 = df_grouped[(df_grouped['Mídia'] == var3)]
    values3 = df3[var0]
    y3 = df3[metrica]

    df4 = df_grouped[(df_grouped['Mídia'] == var4)]
    values4 = df4[var0]
    y4 = df4[metrica]

    df5 = df_grouped[(df_grouped['Mídia'] == var5)]
    values5 = df5[var0]
    y5 = df5[metrica]


    fig.add_trace(go.Scatter(
        x=values1, y=y1, name=var1,
        mode='lines',
        line=dict(width=3, color=cor1),
        stackgroup='one',
        groupnorm='percent'  # sets the normalization for the sum of the stackgroup
    ))

    fig.add_trace(go.Scatter(
        x=values2, y=y2, name=var2,
        mode='lines',
        line=dict(width=3, color=cor2),
        stackgroup='one'
    ))
    fig.add_trace(go.Scatter(
        x=values3, y=y3, name=var3,
        mode='lines',
        line=dict(width=3, color=cor3),
        stackgroup='one'
    ))
    fig.add_trace(go.Scatter(
        x=values4, y=y4, name=var4,
        mode='lines',
        line=dict(width=3, color=cor4),
        stackgroup='one'
    ))
    fig.add_trace(go.Scatter(
        x=values5, y=y5, name=var5,
        mode='lines',
        line=dict(width=3, color=cor5),
        stackgroup='one'
    ))


    fig.update_layout(
        showlegend=True,
        xaxis_type='category',
        yaxis=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'))

    fig.update_layout(
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        legend=dict(font_size=10, orientation="h", yanchor="top", y=1.10, xanchor="center", x=0.50),
        height=400, hovermode="x unified", autosize=False, dragmode=False, margin=dict(l=80, r=20, b=10, t=50)
    )
    fig.update_yaxes(
        title_text="Proporção da "+metrica+"por Mídia de Trafego", title_font=dict(family='Sans-serif', size=12),
        tickfont=dict(family='Sans-serif', size=12), nticks=5, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )
    fig.update_xaxes(
        dtick=5, tickfont=dict(family='Sans-serif', size=12), nticks=10, showgrid=False
    )

    return fig



def linha_(df, var0, metrica, var1, cor1, var2, cor2, var3, cor3, var4, cor4, var5, cor5):
    fig = go.Figure()
    df['Semana do ano'] = (df['Semana do ano'] * 100)

    df_grouped = df.groupby([var0, 'Mídia'])[metrica].sum().reset_index()


    df1 = df_grouped[(df_grouped['Mídia'] == var1)]
    values1 = df1[var0]
    y1 = df1[metrica]

    df2 = df_grouped[(df_grouped['Mídia'] == var2)]
    values2 = df2[var0]
    y2 = df2[metrica]

    df3 = df_grouped[(df_grouped['Mídia'] == var3)]
    values3 = df3[var0]
    y3 = df3[metrica]

    df4 = df_grouped[(df_grouped['Mídia'] == var4)]
    values4 = df4[var0]
    y4 = df4[metrica]

    df5 = df_grouped[(df_grouped['Mídia'] == var5)]
    values5 = df5[var0]
    y5 = df5[metrica]


    fig.add_trace(go.Scatter(
        x=values1, y=y1, name=var1,
        mode='lines',
        line=dict(width=3, color=cor1),
        stackgroup='one',
    ))

    fig.add_trace(go.Scatter(
        x=values2, y=y2, name=var2,
        mode='lines',
        line=dict(width=3, color=cor2),
        stackgroup='two'
    ))
    fig.add_trace(go.Scatter(
        x=values3, y=y3, name=var3,
        mode='lines',
        line=dict(width=3, color=cor3),
        stackgroup='three'
    ))
    fig.add_trace(go.Scatter(
        x=values4, y=y4, name=var4,
        mode='lines',
        line=dict(width=3, color=cor4),
        stackgroup='four'
    ))
    fig.add_trace(go.Scatter(
        x=values5, y=y5, name=var5,
        mode='lines',
        line=dict(width=3, color=cor5),
        stackgroup='five'
    ))

    fig.update_layout(
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        legend=dict(font_size=10, orientation="h", yanchor="top", y=1.10, xanchor="center", x=0.50),
        height=400, hovermode="x unified", autosize=False, dragmode=False, margin=dict(l=80, r=20, b=10, t=50)
    )
    fig.update_yaxes(
        title_text="Proporção da "+metrica+"por Mídia de Trafego", title_font=dict(family='Sans-serif', size=12),
        tickfont=dict(family='Sans-serif', size=12), nticks=5, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )
    fig.update_xaxes(
        dtick=5, tickfont=dict(family='Sans-serif', size=12), nticks=10, showgrid=False
    )

    return fig

def varia_point(df, agrupo, groupby, colorscales):

    if groupby == 'Soma':
        df1_data1 = (df.groupby(['Data', agrupo]).sum()).reset_index()
    elif groupby == 'Média':
        df1_data1 = (df.groupby(['Data', agrupo]).mean()).reset_index()

    metrics1 = ['Receita', 'Transações', 'Taxa de conversão',
                'Sessões', 'Usuários','Novos usuários',
                'Páginas / sessão', 'Visualizações de página', 'Duração média da sessão',
                'Custo', 'Impressões', 'Cliques']

    fig0 = make_subplots(rows=4, cols=3, subplot_titles=metrics1, vertical_spacing=0.1, horizontal_spacing=0.08)

    for i, metric in enumerate(metrics1, 1):
        row = (i - 1) // 3 + 1
        col = (i - 1) % 3 + 1
        fig0.add_trace(
            go.Scatter(x=df1_data1[agrupo], y=df1_data1[metric],
                       mode='markers', name=metric,
                       hovertemplate="</br><b>" + agrupo + ":</b> %{x:,.0f}" +
                                     "</br><b>" + metric + ":</b> %{y:,.0f}",
                       showlegend=False,
                       marker=dict(
                           size=8,
                           color=(df1_data1[metric]),
                           colorscale=colorscales)),
            row=row,
            col=col
        )


    # Atualizar layout e exibir o gráfico
    fig0.update_layout(
        height=900, width=1000,
        font={'color': "#000000", 'family': "sans-serif"}, showlegend=False,
        margin=dict(l=50, r=50, b=50, t=90), autosize=False, hovermode="x unified")
    fig0.update_yaxes(
        title_font=dict(family='Sans-serif', size=18),
        tickfont=dict(family='Sans-serif', size=12), nticks=8, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3')

    fig0.update_xaxes(
        title_font=dict(family='Sans-serif', size=18),
        tickfont=dict(family='Sans-serif', size=12), nticks=10, showgrid=False)

    return fig0

def plot_point(df, varx, vary, colorscales, vermedia):

    df = df.groupby(['Data', varx])[vary].sum().astype(int).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df[varx], y=df[vary],
                             mode='markers', name='',
                             hovertemplate="</br><b>"+varx+":</b> %{x:,.0f}" +
                                           "</br><b>"+vary+":</b> %{y:,.0f}",
                             showlegend=False,
                             marker=dict(
                                 size=10,
                                 color=((df[vary])),
                                 colorscale=colorscales)
                             ))
    if vermedia == 'Sim':
        media_metrica = df[vary].median()
        # Adicionar linha com a média da métrica
        fig.add_trace(
            go.Scatter(
                x=[df[varx].min(), df[varx].max()],
                y=[media_metrica, media_metrica],
                mode='lines',
                name='Mediana Geral', showlegend=False,
                line=dict(color='black', dash='dash')
            )
        )

    fig.update_layout(
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        height=400, hovermode="closest", autosize=False, margin=dict(l=80, r=20, b=10, t=30))
    fig.update_xaxes(
        title_text=varx, title_font=dict(family='Sans-serif', size=12), zeroline=False,
        tickfont=dict(family='Sans-serif', size=12), nticks=7, showgrid=True, gridwidth=0.8, gridcolor='#D3D3D3')
    fig.update_yaxes(
        title_text=vary, title_font=dict(family='Sans-serif', size=12), zeroline=False,
        tickfont=dict(family='Sans-serif', size=14), nticks=7, showgrid=True, gridwidth=0.8, gridcolor='#D3D3D3')

    return fig


def normalizado_usuario(df, metrica):
    fig = go.Figure()

    df_grouped = df.groupby(['Semana do ano', 'Tipo de usuário'])[metrica].sum().reset_index()


    df1 = df_grouped[(df_grouped['Tipo de usuário'] == 'New Visitor')]
    values1 = df1['Semana do ano']
    y1 = df1[metrica]

    df2 = df_grouped[(df_grouped['Tipo de usuário'] == 'Returning Visitor')]
    values2 = df2['Semana do ano']
    y2 = df2[metrica]


    fig.add_trace(go.Scatter(
        x=values1, y=y1, name='New Visitor',
        mode='lines',
        line=dict(width=3, color='#325771'),
        stackgroup='one',
        groupnorm='percent'  # sets the normalization for the sum of the stackgroup
    ))

    fig.add_trace(go.Scatter(
        x=values2, y=y2, name='Returning Visitor',
        mode='lines',
        line=dict(width=3, color='#fd6735'),
        stackgroup='one'
    ))


    fig.update_layout(
        showlegend=True,
        xaxis_type='category',
        yaxis=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'))

    fig.update_layout(
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        legend=dict(font_size=10, orientation="h", yanchor="top", y=1.3, xanchor="center", x=0.50),
        height=300, hovermode="x unified", autosize=False, dragmode=False, margin=dict(l=20, r=20, b=20, t=30)
    )
    fig.update_yaxes(
        title_text="Proporção da "+metrica, title_font=dict(family='Sans-serif', size=12),
        tickfont=dict(family='Sans-serif', size=12), nticks=5, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )
    fig.update_xaxes(
        dtick=5, tickfont=dict(family='Sans-serif', size=12),
        nticks=20, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )

    return fig


def linha_usuarios(df, metrica):

    df_grouped = df.groupby(['Semana do ano', 'Tipo de usuário'])[metrica].sum().reset_index()

    df1 = df_grouped[(df_grouped['Tipo de usuário'] == 'New Visitor')]
    values1 = df1['Semana do ano']
    y1 = df1[metrica]

    df2 = df_grouped[(df_grouped['Tipo de usuário'] == 'Returning Visitor')]
    values2 = df2['Semana do ano']
    y2 = df2[metrica]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=values1, y=y1, name='New Visitor',
        mode='lines',
        line=dict(width=3, color='#325771'),
        stackgroup='one',
    ))

    fig.add_trace(go.Scatter(
        x=values2, y=y2, name='Returning Visitor',
        mode='lines',
        line=dict(width=3, color='#fd6735'),
        stackgroup='two'
    ))

    fig.update_layout(
        showlegend=True, xaxis_type='category',
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        legend=dict(font_size=10, orientation="h", yanchor="top", y=1.10, xanchor="center", x=0.50),
        height=300, hovermode="x unified", autosize=False, dragmode=False, margin=dict(l=20, r=20, b=20, t=30)
    )
    fig.update_yaxes(
        title_text="Total de "+metrica, title_font=dict(family='Sans-serif', size=12),
        tickfont=dict(family='Sans-serif', size=12), nticks=5, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )
    fig.update_xaxes(
        dtick=5, tickfont=dict(family='Sans-serif', size=12),
        nticks=20, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )

    return fig


def normalizado_regiao(df, metrica):

    top_regiao = df.groupby(['Região'])[metrica].sum().reset_index().sort_values(metrica, ascending=False)
    top1 = top_regiao.iloc[0,0]
    top2 = top_regiao.iloc[1,0]
    top3 = top_regiao.iloc[2,0]


    df1 = df[(df['Região'] == top1)]
    df1 = df1.groupby(['Semana do ano'])[metrica].sum().reset_index()
    values1 = df1['Semana do ano']
    y1 = df1[metrica]

    df2 = df[(df['Região'] == top2)]
    df2 = df2.groupby(['Semana do ano'])[metrica].sum().reset_index()
    values2 = df2['Semana do ano']
    y2 = df2[metrica]

    df3 = df[(df['Região'] == top3)]
    df3 = df3.groupby(['Semana do ano'])[metrica].sum().reset_index()
    values3 = df3['Semana do ano']
    y3 = df3[metrica]

    df4 = df[ (df['Região'] != top1) & (df['Região'] != top2) & (df['Região'] != top3)]
    df4 = df4.groupby(['Semana do ano'])[metrica].sum().reset_index()
    values4 = df4['Semana do ano']
    y4 = df4[metrica]


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=values1, y=y1, name=top1,
        mode='lines',
        line=dict(width=3, color='#28e10a'),
        stackgroup='one',
        groupnorm='percent'  # sets the normalization for the sum of the stackgroup
    ))

    fig.add_trace(go.Scatter(
        x=values2, y=y2, name=top2,
        mode='lines',
        line=dict(width=3, color='#0500c7'),
        stackgroup='one'
    ))

    fig.add_trace(go.Scatter(
        x=values3, y=y3, name=top3,
        mode='lines',
        line=dict(width=3, color='#5c03fa'),
        stackgroup='one'
    ))

    fig.add_trace(go.Scatter(
        x=values4, y=y4, name='Outros',
        mode='lines',
        line=dict(width=3, color='#de00ed'),
        stackgroup='one'
    ))


    fig.update_layout(
        showlegend=True,
        xaxis_type='category',
        yaxis=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'))

    fig.update_layout(
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        legend=dict(font_size=10, orientation="h", yanchor="top", y=1.3, xanchor="center", x=0.50),
        height=300, hovermode="x unified", autosize=False, dragmode=False, margin=dict(l=20, r=20, b=20, t=50)
    )
    fig.update_yaxes(
        title_text="Proporção da "+metrica, title_font=dict(family='Sans-serif', size=12),
        tickfont=dict(family='Sans-serif', size=12), nticks=5, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )
    fig.update_xaxes(
        dtick=5, tickfont=dict(family='Sans-serif', size=12),
        nticks=20, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )

    return fig


def linha_regiao(df, metrica):

    top_regiao = df.groupby(['Região'])[metrica].sum().reset_index().sort_values(metrica, ascending=False)
    top1 = top_regiao.iloc[0,0]
    top2 = top_regiao.iloc[1,0]
    top3 = top_regiao.iloc[2,0]


    df1 = df[(df['Região'] == top1)]
    df1 = df1.groupby(['Semana do ano'])[metrica].sum().reset_index()
    values1 = df1['Semana do ano']
    y1 = df1[metrica]

    df2 = df[(df['Região'] == top2)]
    df2 = df2.groupby(['Semana do ano'])[metrica].sum().reset_index()
    values2 = df2['Semana do ano']
    y2 = df2[metrica]

    df3 = df[(df['Região'] == top3)]
    df3 = df3.groupby(['Semana do ano'])[metrica].sum().reset_index()
    values3 = df3['Semana do ano']
    y3 = df3[metrica]

    df4 = df[ (df['Região'] != top1) & (df['Região'] != top2) & (df['Região'] != top3)]
    df4 = df4.groupby(['Semana do ano'])[metrica].sum().reset_index()
    values4 = df4['Semana do ano']
    y4 = df4[metrica]


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=values1, y=y1, name=top1,
        mode='lines',
        line=dict(width=3, color='#28e10a'),
        stackgroup='one',
    ))

    fig.add_trace(go.Scatter(
        x=values4, y=y4, name='Outros',
        mode='lines',
        line=dict(width=3, color='#de00ed'),
        stackgroup='two'
    ))

    fig.add_trace(go.Scatter(
        x=values2, y=y2, name=top2,
        mode='lines',
        line=dict(width=3, color='#0500c7'),
        stackgroup='three'
    ))

    fig.add_trace(go.Scatter(
        x=values3, y=y3, name=top3,
        mode='lines',
        line=dict(width=3, color='#5c03fa'),
        stackgroup='four'
    ))


    fig.update_layout(
        showlegend=True, xaxis_type='category',
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        legend=dict(font_size=10, orientation="h", yanchor="top", y=1.3, xanchor="center", x=0.50),
        height=300, hovermode="x unified", autosize=False, dragmode=False, margin=dict(l=20, r=20, b=20, t=50)
    )
    fig.update_yaxes(
        title_text="Total de "+metrica, title_font=dict(family='Sans-serif', size=12),
        tickfont=dict(family='Sans-serif', size=12), nticks=5, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )
    fig.update_xaxes(
        dtick=5, tickfont=dict(family='Sans-serif', size=12),
        nticks=20, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )

    return fig

def normalizado_dispositivo(df, metrica):
    fig = go.Figure()

    df_grouped = df.groupby(['Semana do ano', 'Categoria do dispositivo'])[metrica].sum().reset_index()

    top_ = df.groupby(['Categoria do dispositivo'])[metrica].sum().reset_index().sort_values(metrica, ascending=False)

    top1 = top_.iloc[0,0]
    top2 = top_.iloc[1,0]
    top3 = top_.iloc[2,0]



    df1 = df_grouped[(df_grouped['Categoria do dispositivo'] == top1)]
    values1 = df1['Semana do ano']
    y1 = df1[metrica]

    df2 = df_grouped[(df_grouped['Categoria do dispositivo'] == top2)]
    values2 = df2['Semana do ano']
    y2 = df2[metrica]

    df3 = df_grouped[(df_grouped['Categoria do dispositivo'] == top3)]
    values3 = df3['Semana do ano']
    y3 = df3[metrica]


    fig.add_trace(go.Scatter(
        x=values1, y=y1, name=top1,
        mode='lines',
        line=dict(width=3, color='#225575'),
        stackgroup='one',
        groupnorm='percent'  # sets the normalization for the sum of the stackgroup
    ))

    fig.add_trace(go.Scatter(
        x=values2, y=y2, name=top2,
        mode='lines',
        line=dict(width=3, color='#60c04c'),
        stackgroup='one'
    ))

    fig.add_trace(go.Scatter(
        x=values3, y=y3, name=top3,
        mode='lines',
        line=dict(width=3, color='#e94a22'),
        stackgroup='one'
    ))


    fig.update_layout(
        showlegend=True,
        xaxis_type='category',
        yaxis=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'))

    fig.update_layout(
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        legend=dict(font_size=10, orientation="h", yanchor="top", y=1.20, xanchor="center", x=0.50),
        height=300, hovermode="x unified", autosize=False, dragmode=False, margin=dict(l=20, r=20, b=20, t=50)
    )
    fig.update_yaxes(
        title_text="Proporção da "+metrica, title_font=dict(family='Sans-serif', size=12),
        tickfont=dict(family='Sans-serif', size=12), nticks=5, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )
    fig.update_xaxes(
        dtick=5, tickfont=dict(family='Sans-serif', size=12),
        nticks=20, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )

    return fig


def linha_dispositivo(df, metrica):
    fig = go.Figure()

    df_grouped = df.groupby(['Semana do ano', 'Categoria do dispositivo'])[metrica].sum().reset_index()


    df1 = df_grouped[(df_grouped['Categoria do dispositivo'] == 'mobile')]
    values1 = df1['Semana do ano']
    y1 = df1[metrica]

    df2 = df_grouped[(df_grouped['Categoria do dispositivo'] == 'desktop')]
    values2 = df2['Semana do ano']
    y2 = df2[metrica]

    df3 = df_grouped[(df_grouped['Categoria do dispositivo'] == 'tablet')]
    values3 = df3['Semana do ano']
    y3 = df3[metrica]


    fig.add_trace(go.Scatter(
        x=values1, y=y1, name='mobile',
        mode='lines',
        line=dict(width=3, color='#225575'),
        stackgroup='one',
    ))

    fig.add_trace(go.Scatter(
        x=values2, y=y2, name='desktop',
        mode='lines',
        line=dict(width=3, color='#60c04c'),
        stackgroup='two'
    ))

    fig.add_trace(go.Scatter(
        x=values3, y=y3, name='tablet',
        mode='lines',
        line=dict(width=3, color='#e94a22'),
        stackgroup='three'
    ))


    fig.update_layout(
        showlegend=True,xaxis_type='category',
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        legend=dict(font_size=10, orientation="h", yanchor="top", y=1.10, xanchor="center", x=0.50),
        height=300, hovermode="x unified", autosize=False, dragmode=False, margin=dict(l=20, r=20, b=20, t=50)
    )
    fig.update_yaxes(
        title_text="Total de "+metrica, title_font=dict(family='Sans-serif', size=12),
        tickfont=dict(family='Sans-serif', size=12), nticks=5, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )
    fig.update_xaxes(
        dtick=5, tickfont=dict(family='Sans-serif', size=12),
        nticks=20, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )

    return fig


#############################################################################################

def bar_plot(df, var1, var2, cor1):

    df = df.sort_values(var2, ascending=True)

    values = df[var1]
    y = df[var2]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=y, text=y, textposition='inside', insidetextanchor='start', name='',
        textfont=dict(size=30, color='white', family='Arial'),
        hovertemplate="</br><b>"+var1+":</b> %{x}" +
                      "</br><b>Frequência:</b> %{y}",
        marker_color=cor1))




    fig.update_layout(
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        height=300, margin=dict(l=20, r=20, b=20, t=20), autosize=False,
        dragmode=False, hovermode="x", clickmode="event+select")
    fig.update_yaxes(
        tickfont=dict(family='Sans-serif', size=12), showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3')
    fig.update_xaxes(
        dtick=1, tickfont=dict(family='Sans-serif', size=10), showgrid=False, gridwidth=0.5, gridcolor='#D3D3D3'
    )

    for figure in fig.data:
        figure.update(
            selected=dict(marker=dict(color="#E30613")),
            unselected=dict(marker=dict(color="#05A854", opacity=1)),
        )

    return fig

def bar_plot_horiz(df, categico, numerico, cor1, y_name):

    df[numerico] = df[numerico].astype(int)
    df = df.sort_values(numerico, ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[numerico], y=df[categico], text=df[numerico], textposition='inside', insidetextanchor='end', name='',
        textfont=dict(color='white', family='Arial'),
        textangle=0,
        hovertemplate="</br><b>"+y_name+":</b> %{y}" +
                      "</br><b>Total:</b> %{x}",
        orientation='h',
        marker_color=cor1))

    fig.update_layout(
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        height=300, margin=dict(l=20, r=50, b=20, t=20), autosize=False,
        dragmode=False, hovermode="closest", clickmode="event+select")
    fig.update_yaxes(
        title_text=categico,
        title_font=dict(family='Sans-serif', size=10),
        tickfont=dict(family='Sans-serif', size=10),
        showgrid=False)
    fig.update_xaxes(
        tickfont=dict(family='Sans-serif', size=10), nticks=5,
        showgrid=True, gridwidth=0.8, gridcolor='#D3D3D3')

    for figure in fig.data:
        figure.update(
            selected=dict(marker=dict(color=cor1)),
            unselected=dict(marker=dict(color=cor1, opacity=0.3)),
        )


    return fig


def bar_plot_horiz_mes(df, categico, numerico, cor1, y_name):


    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[numerico], y=df[categico], text=df[numerico], textposition='inside', insidetextanchor='end', name='',
        textfont=dict(color='white', family='Arial'),
        textangle=0,
        hovertemplate="</br><b>"+y_name+":</b> %{y}" +
                      "</br><b>Total:</b> %{x}",
        orientation='h',
        marker_color=cor1))

    fig.update_layout(
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        height=300, margin=dict(l=20, r=20, b=20, t=20), autosize=False,
        dragmode=False, hovermode="closest", clickmode="event+select")
    fig.update_yaxes(
        tickfont=dict(family='Sans-serif', size=12),
        showgrid=False)
    fig.update_xaxes(
        tickfont=dict(family='Sans-serif', size=12),
        showgrid=True, gridwidth=0.8, gridcolor='#D3D3D3')

    for figure in fig.data:
        figure.update(
            selected=dict(marker=dict(color=cor1)),
            unselected=dict(marker=dict(color=cor1, opacity=0.3)),
        )


    return fig


def pizza_tipo_usuario(labels, values):

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4,
                             marker=dict(colors=['#000080', '#008080', '#006080']))])

    fig.update_layout(
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF",
        legend=dict(font_size=12, orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.50),
        height=300, margin=dict(l=20, r=20, b=80, t=30)
    )

    return fig




def linha_data(x, y, color):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y, name='',
        mode='lines+markers',
        line=dict(width=1, color=color),
        marker=dict(line=dict(width=0.5),size=3, symbol='diamond', color=color),
        stackgroup='one',
    ))


    fig.update_layout(
        showlegend=False, xaxis_type='category',
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        height=300, hovermode="x unified", autosize=False, dragmode=False, margin=dict(l=20, r=20, b=20, t=20)
    )
    fig.update_yaxes(
        tickfont=dict(family='Sans-serif', size=12), nticks=5, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )
    fig.update_xaxes(
        tickfont=dict(family='Sans-serif', size=12),
        showgrid=False, gridwidth=0.5, gridcolor='#D3D3D3'
    )

    return fig




def linha_situacao_data(df, tempo):

    df = df.groupby([tempo, 'Situação'])['Quantidade'].sum().reset_index()
    df_e = df[(df['Situação'] == 'Entregue')]
    df_c = df[(df['Situação'] == 'Cancelado')]
    df_a = df[(df['Situação'] == 'Em aberto')]
    df_o = df[(df['Situação'] != 'Entregue') & (df['Situação'] != 'Cancelado') & (df['Situação'] != 'Em aberto')]


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_e[tempo], y=df_e['Quantidade'], name='Entregue',
        mode='lines+markers',
        line=dict(width=1, color='#197119'),
        marker=dict(line=dict(width=0.5),size=3, symbol='diamond', color='#197119'),
        stackgroup='one',
    ))
    fig.add_trace(go.Scatter(
        x=df_c[tempo], y=df_c['Quantidade'], name='Cancelado',
        mode='lines+markers',
        line=dict(width=1, color='#711919'),
        marker=dict(line=dict(width=0.5), size=3, symbol='diamond', color='#711919'),
        stackgroup='two',
    ))
    fig.add_trace(go.Scatter(
        x=df_a[tempo], y=df_a['Quantidade'], name='Em aberto',
        mode='lines+markers',
        line=dict(width=1, color='#195b71'),
        marker=dict(line=dict(width=0.5), size=3, symbol='diamond', color='#195b71'),
        stackgroup='three',
    ))
    fig.add_trace(go.Scatter(
        x=df_o[tempo], y=df_o['Quantidade'], name='Outros',
        mode='lines+markers',
        line=dict(width=1, color='#711971'),
        marker=dict(line=dict(width=0.5), size=3, symbol='diamond', color='#711971'),
        stackgroup='three',
    ))

    fig.update_layout(
        showlegend=True, xaxis_type='category',
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        legend=dict(font_size=10, orientation="h", yanchor="top", y=1.10, xanchor="center", x=0.50),
        height=300, hovermode="x unified", autosize=False, dragmode=False, margin=dict(l=20, r=20, b=20, t=20)
    )
    fig.update_yaxes(
        tickfont=dict(family='Sans-serif', size=12), nticks=5, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )
    fig.update_xaxes(
        tickfont=dict(family='Sans-serif', size=12),
        showgrid=False, gridwidth=0.5, gridcolor='#D3D3D3'
    )

    return fig



def get_month_number(date):
    year = date.year
    month = date.month

    month = f'{year}.{month:02d}'

    return float(month)

def get_week_number(date):
    year = date.year
    week_number = date.isocalendar()[1]

    semana = f'{year}.{week_number:02d}'

    return float(semana)

def get_year_number(date):
    year = date.year

    return int(year)

def get_week_name(date):
    dia_semana = date.strftime('%A')

    if dia_semana == 'Monday':
        dia_semana = 'Segunda'
    elif dia_semana == 'Tuesday':
        dia_semana = 'Terça'
    elif dia_semana == 'Wednesday':
        dia_semana = 'Quarta'
    elif dia_semana == 'Thursday':
        dia_semana = 'Quinta'
    elif dia_semana == 'Friday':
        dia_semana = 'Sexta'
    elif dia_semana == 'Saturday':
        dia_semana = 'Sábado'
    elif dia_semana == 'Sunday':
        dia_semana = 'Domingo'

    dia_semana = f'{dia_semana}'

    return str(dia_semana)

def get_month_name(date):
    mes_nome = date.strftime('%B')

    if mes_nome == 'January':
        mes_nome = 'Janeiro'
    elif mes_nome == 'February':
        mes_nome = 'Fevereiro'
    elif mes_nome == 'March':
        mes_nome = 'Março'
    elif mes_nome == 'April':
        mes_nome = 'Abril'
    elif mes_nome == 'May':
        mes_nome = 'Maio'
    elif mes_nome == 'June':
        mes_nome = 'Junho'
    elif mes_nome == 'July':
        mes_nome = 'Julho'
    elif mes_nome == 'August':
        mes_nome = 'Agosto'
    elif mes_nome == 'September':
        mes_nome = 'Setembro'
    elif mes_nome == 'October':
        mes_nome = 'Outubro'
    elif mes_nome == 'November':
        mes_nome = 'Novembro'
    elif mes_nome == 'December':
        mes_nome = 'Dezembro'

    mes_nome = f'{mes_nome}'

    return str(mes_nome)




def merge_data(df, df2, tempo, status):

    if status is False:
        merge1 = df.groupby([tempo])['Quantidade'].count().reset_index()

        merge2 = df2.groupby([tempo])['Quantidade'].count().reset_index()
        merge2['Quantidade'] = 0

        merged_df = pd.merge(merge1, merge2, on=tempo, how='outer')
        merged_df = merged_df.fillna(0)
        merged_df = merged_df.rename(columns={"Quantidade_x": "Quantidade"})
        merged_df = merged_df[[tempo, 'Quantidade']].sort_values(tempo, ascending=True)

    elif status is True:
        merge1 = df.groupby([tempo, 'Situação']).count().reset_index()

        merge2 = df2.groupby([tempo, 'Situação']).count().reset_index()
        merge2['Quantidade'] = 0

        merged_df = pd.merge(merge1, merge2, on=[tempo, 'Situação'], how='outer')
        merged_df = merged_df[[tempo, 'Situação', 'Quantidade_x']]
        merged_df = merged_df.fillna(0)
        merged_df = merged_df.rename(columns={"Quantidade_x": "Quantidade"})
        merged_df = merged_df.sort_values(tempo, ascending=True)

    return merged_df

def merge_data_sum(df, df2, tempo,  status):

    if status is False:
        merge1 = df.groupby([tempo])['Quantidade'].sum().reset_index()

        merge2 = df2.groupby([tempo])['Quantidade'].sum().reset_index()
        merge2['Quantidade'] = 0

        merged_df = pd.merge(merge1, merge2, on=tempo, how='outer')
        merged_df = merged_df.fillna(0)
        merged_df = merged_df.rename(columns={"Quantidade_x": "Quantidade"})
        merged_df = merged_df[[tempo, 'Quantidade']].sort_values(tempo, ascending=True)

    elif status is True:
        merge1 = df.groupby([tempo, 'Situação']).sum().reset_index()

        merge2 = df2.groupby([tempo, 'Situação']).sum().reset_index()
        merge2['Quantidade'] = 0

        merged_df = pd.merge(merge1, merge2, on=[tempo, 'Situação'], how='outer')
        merged_df = merged_df[[tempo, 'Situação', 'Quantidade_x']]
        merged_df = merged_df.fillna(0)
        merged_df = merged_df.rename(columns={"Quantidade_x": "Quantidade"})
        merged_df = merged_df.sort_values(tempo, ascending=True)

    return merged_df


def ajuste_tempo(df):
    df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d')
    df['Data'] = df['Data'].dt.date


    df.fillna('None', inplace=True)
    df['Semana do ano'] = df['Data'].apply(get_week_number)
    df['Semana'] = df['Data'].apply(get_week_name)
    df['Mês do ano'] = df['Data'].apply(get_month_number)
    df['Mês'] = df['Data'].apply(get_month_name)
    df['Ano'] = df['Data'].apply(get_year_number)

    return df

def ajuste_tempo2(df):
    df['Data'] = pd.to_datetime(df['Data'], format='%Y/%m/%d')
    df['Data'] = df['Data'].dt.date


    df.fillna('None', inplace=True)
    df['Semana do ano'] = df['Data'].apply(get_week_number)
    df['Semana'] = df['Data'].apply(get_week_name)
    df['Mês do ano'] = df['Data'].apply(get_month_number)
    df['Mês'] = df['Data'].apply(get_month_name)
    df['Ano'] = df['Data'].apply(get_year_number)

    return df


def plot_hotmap(df):

    df_map = df.groupby(['Semana', 'Mês'])['ID'].agg('count').reset_index().sort_values('Semana', ascending=True)

    jan_seg = df_map[(df_map['Mês'] == 'Janeiro') & (df_map['Semana'] == 'Segunda')]['ID']
    jan_ter = df_map[(df_map['Mês'] == 'Janeiro') & (df_map['Semana'] == 'Terça')]['ID']
    jan_qua = df_map[(df_map['Mês'] == 'Janeiro') & (df_map['Semana'] == 'Quarta')]['ID']
    jan_qui = df_map[(df_map['Mês'] == 'Janeiro') & (df_map['Semana'] == 'Quinta')]['ID']
    jan_sex = df_map[(df_map['Mês'] == 'Janeiro') & (df_map['Semana'] == 'Sexta')]['ID']
    jan_sab = df_map[(df_map['Mês'] == 'Janeiro') & (df_map['Semana'] == 'Sábado')]['ID']
    jan_dom = df_map[(df_map['Mês'] == 'Janeiro') & (df_map['Semana'] == 'Domingo')]['ID']

    fev_seg = df_map[(df_map['Mês'] == 'Fevereiro') & (df_map['Semana'] == 'Segunda')]['ID']
    fev_ter = df_map[(df_map['Mês'] == 'Fevereiro') & (df_map['Semana'] == 'Terça')]['ID']
    fev_qua = df_map[(df_map['Mês'] == 'Fevereiro') & (df_map['Semana'] == 'Quarta')]['ID']
    fev_qui = df_map[(df_map['Mês'] == 'Fevereiro') & (df_map['Semana'] == 'Quinta')]['ID']
    fev_sex = df_map[(df_map['Mês'] == 'Fevereiro') & (df_map['Semana'] == 'Sexta')]['ID']
    fev_sab = df_map[(df_map['Mês'] == 'Fevereiro') & (df_map['Semana'] == 'Sábado')]['ID']
    fev_dom = df_map[(df_map['Mês'] == 'Fevereiro') & (df_map['Semana'] == 'Domingo')]['ID']

    mar_seg = df_map[(df_map['Mês'] == 'Março') & (df_map['Semana'] == 'Segunda')]['ID']
    mar_ter = df_map[(df_map['Mês'] == 'Março') & (df_map['Semana'] == 'Terça')]['ID']
    mar_qua = df_map[(df_map['Mês'] == 'Março') & (df_map['Semana'] == 'Quarta')]['ID']
    mar_qui = df_map[(df_map['Mês'] == 'Março') & (df_map['Semana'] == 'Quinta')]['ID']
    mar_sex = df_map[(df_map['Mês'] == 'Março') & (df_map['Semana'] == 'Sexta')]['ID']
    mar_sab = df_map[(df_map['Mês'] == 'Março') & (df_map['Semana'] == 'Sábado')]['ID']
    mar_dom = df_map[(df_map['Mês'] == 'Março') & (df_map['Semana'] == 'Domingo')]['ID']

    abr_seg = df_map[(df_map['Mês'] == 'Abril') & (df_map['Semana'] == 'Segunda')]['ID']
    abr_ter = df_map[(df_map['Mês'] == 'Abril') & (df_map['Semana'] == 'Terça')]['ID']
    abr_qua = df_map[(df_map['Mês'] == 'Abril') & (df_map['Semana'] == 'Quarta')]['ID']
    abr_qui = df_map[(df_map['Mês'] == 'Abril') & (df_map['Semana'] == 'Quinta')]['ID']
    abr_sex = df_map[(df_map['Mês'] == 'Abril') & (df_map['Semana'] == 'Sexta')]['ID']
    abr_sab = df_map[(df_map['Mês'] == 'Abril') & (df_map['Semana'] == 'Sábado')]['ID']
    abr_dom = df_map[(df_map['Mês'] == 'Abril') & (df_map['Semana'] == 'Domingo')]['ID']

    mai_seg = df_map[(df_map['Mês'] == 'Maio') & (df_map['Semana'] == 'Segunda')]['ID']
    mai_ter = df_map[(df_map['Mês'] == 'Maio') & (df_map['Semana'] == 'Terça')]['ID']
    mai_qua = df_map[(df_map['Mês'] == 'Maio') & (df_map['Semana'] == 'Quarta')]['ID']
    mai_qui = df_map[(df_map['Mês'] == 'Maio') & (df_map['Semana'] == 'Quinta')]['ID']
    mai_sex = df_map[(df_map['Mês'] == 'Maio') & (df_map['Semana'] == 'Sexta')]['ID']
    mai_sab = df_map[(df_map['Mês'] == 'Maio') & (df_map['Semana'] == 'Sábado')]['ID']
    mai_dom = df_map[(df_map['Mês'] == 'Maio') & (df_map['Semana'] == 'Domingo')]['ID']

    jun_seg = df_map[(df_map['Mês'] == 'Junho') & (df_map['Semana'] == 'Segunda')]['ID']
    jun_ter = df_map[(df_map['Mês'] == 'Junho') & (df_map['Semana'] == 'Terça')]['ID']
    jun_qua = df_map[(df_map['Mês'] == 'Junho') & (df_map['Semana'] == 'Quarta')]['ID']
    jun_qui = df_map[(df_map['Mês'] == 'Junho') & (df_map['Semana'] == 'Quinta')]['ID']
    jun_sex = df_map[(df_map['Mês'] == 'Junho') & (df_map['Semana'] == 'Sexta')]['ID']
    jun_sab = df_map[(df_map['Mês'] == 'Junho') & (df_map['Semana'] == 'Sábado')]['ID']
    jun_dom = df_map[(df_map['Mês'] == 'Junho') & (df_map['Semana'] == 'Domingo')]['ID']

    jul_seg = df_map[(df_map['Mês'] == 'Julho') & (df_map['Semana'] == 'Segunda')]['ID']
    jul_ter = df_map[(df_map['Mês'] == 'Julho') & (df_map['Semana'] == 'Terça')]['ID']
    jul_qua = df_map[(df_map['Mês'] == 'Julho') & (df_map['Semana'] == 'Quarta')]['ID']
    jul_qui = df_map[(df_map['Mês'] == 'Julho') & (df_map['Semana'] == 'Quinta')]['ID']
    jul_sex = df_map[(df_map['Mês'] == 'Julho') & (df_map['Semana'] == 'Sexta')]['ID']
    jul_sab = df_map[(df_map['Mês'] == 'Julho') & (df_map['Semana'] == 'Sábado')]['ID']
    jul_dom = df_map[(df_map['Mês'] == 'Julho') & (df_map['Semana'] == 'Domingo')]['ID']

    ago_seg = df_map[(df_map['Mês'] == 'Agosto') & (df_map['Semana'] == 'Segunda')]['ID']
    ago_ter = df_map[(df_map['Mês'] == 'Agosto') & (df_map['Semana'] == 'Terça')]['ID']
    ago_qua = df_map[(df_map['Mês'] == 'Agosto') & (df_map['Semana'] == 'Quarta')]['ID']
    ago_qui = df_map[(df_map['Mês'] == 'Agosto') & (df_map['Semana'] == 'Quinta')]['ID']
    ago_sex = df_map[(df_map['Mês'] == 'Agosto') & (df_map['Semana'] == 'Sexta')]['ID']
    ago_sab = df_map[(df_map['Mês'] == 'Agosto') & (df_map['Semana'] == 'Sábado')]['ID']
    ago_dom = df_map[(df_map['Mês'] == 'Agosto') & (df_map['Semana'] == 'Domingo')]['ID']

    set_seg = df_map[(df_map['Mês'] == 'Setembro') & (df_map['Semana'] == 'Segunda')]['ID']
    set_ter = df_map[(df_map['Mês'] == 'Setembro') & (df_map['Semana'] == 'Terça')]['ID']
    set_qua = df_map[(df_map['Mês'] == 'Setembro') & (df_map['Semana'] == 'Quarta')]['ID']
    set_qui = df_map[(df_map['Mês'] == 'Setembro') & (df_map['Semana'] == 'Quinta')]['ID']
    set_sex = df_map[(df_map['Mês'] == 'Setembro') & (df_map['Semana'] == 'Sexta')]['ID']
    set_sab = df_map[(df_map['Mês'] == 'Setembro') & (df_map['Semana'] == 'Sábado')]['ID']
    set_dom = df_map[(df_map['Mês'] == 'Setembro') & (df_map['Semana'] == 'Domingo')]['ID']

    out_seg = df_map[(df_map['Mês'] == 'Outubro') & (df_map['Semana'] == 'Segunda')]['ID']
    out_ter = df_map[(df_map['Mês'] == 'Outubro') & (df_map['Semana'] == 'Terça')]['ID']
    out_qua = df_map[(df_map['Mês'] == 'Outubro') & (df_map['Semana'] == 'Quarta')]['ID']
    out_qui = df_map[(df_map['Mês'] == 'Outubro') & (df_map['Semana'] == 'Quinta')]['ID']
    out_sex = df_map[(df_map['Mês'] == 'Outubro') & (df_map['Semana'] == 'Sexta')]['ID']
    out_sab = df_map[(df_map['Mês'] == 'Outubro') & (df_map['Semana'] == 'Sábado')]['ID']
    out_dom = df_map[(df_map['Mês'] == 'Outubro') & (df_map['Semana'] == 'Domingo')]['ID']

    nov_seg = df_map[(df_map['Mês'] == 'Novembro') & (df_map['Semana'] == 'Segunda')]['ID']
    nov_ter = df_map[(df_map['Mês'] == 'Novembro') & (df_map['Semana'] == 'Terça')]['ID']
    nov_qua = df_map[(df_map['Mês'] == 'Novembro') & (df_map['Semana'] == 'Quarta')]['ID']
    nov_qui = df_map[(df_map['Mês'] == 'Novembro') & (df_map['Semana'] == 'Quinta')]['ID']
    nov_sex = df_map[(df_map['Mês'] == 'Novembro') & (df_map['Semana'] == 'Sexta')]['ID']
    nov_sab = df_map[(df_map['Mês'] == 'Novembro') & (df_map['Semana'] == 'Sábado')]['ID']
    nov_dom = df_map[(df_map['Mês'] == 'Novembro') & (df_map['Semana'] == 'Domingo')]['ID']

    dez_seg = df_map[(df_map['Mês'] == 'Dezembro') & (df_map['Semana'] == 'Segunda')]['ID']
    dez_ter = df_map[(df_map['Mês'] == 'Dezembro') & (df_map['Semana'] == 'Terça')]['ID']
    dez_qua = df_map[(df_map['Mês'] == 'Dezembro') & (df_map['Semana'] == 'Quarta')]['ID']
    dez_qui = df_map[(df_map['Mês'] == 'Dezembro') & (df_map['Semana'] == 'Quinta')]['ID']
    dez_sex = df_map[(df_map['Mês'] == 'Dezembro') & (df_map['Semana'] == 'Sexta')]['ID']
    dez_sab = df_map[(df_map['Mês'] == 'Dezembro') & (df_map['Semana'] == 'Sábado')]['ID']
    dez_dom = df_map[(df_map['Mês'] == 'Dezembro') & (df_map['Semana'] == 'Domingo')]['ID']

    jan_seg = 0 if len(jan_seg) == 0 else jan_seg.values[0]
    jan_ter = 0 if len(jan_ter) == 0 else jan_ter.values[0]
    jan_qua = 0 if len(jan_qua) == 0 else jan_qua.values[0]
    jan_qui = 0 if len(jan_qui) == 0 else jan_qui.values[0]
    jan_sex = 0 if len(jan_sex) == 0 else jan_sex.values[0]
    jan_sab = 0 if len(jan_sab) == 0 else jan_sab.values[0]
    jan_dom = 0 if len(jan_dom) == 0 else jan_dom.values[0]

    fev_seg = 0 if len(fev_seg) == 0 else fev_seg.values[0]
    fev_ter = 0 if len(fev_ter) == 0 else fev_ter.values[0]
    fev_qua = 0 if len(fev_qua) == 0 else fev_qua.values[0]
    fev_qui = 0 if len(fev_qui) == 0 else fev_qui.values[0]
    fev_sex = 0 if len(fev_sex) == 0 else fev_sex.values[0]
    fev_sab = 0 if len(fev_sab) == 0 else fev_sab.values[0]
    fev_dom = 0 if len(fev_dom) == 0 else fev_dom.values[0]

    mar_seg = 0 if len(mar_seg) == 0 else mar_seg.values[0]
    mar_ter = 0 if len(mar_ter) == 0 else mar_ter.values[0]
    mar_qua = 0 if len(mar_qua) == 0 else mar_qua.values[0]
    mar_qui = 0 if len(mar_qui) == 0 else mar_qui.values[0]
    mar_sex = 0 if len(mar_sex) == 0 else mar_sex.values[0]
    mar_sab = 0 if len(mar_sab) == 0 else mar_sab.values[0]
    mar_dom = 0 if len(mar_dom) == 0 else mar_dom.values[0]

    abr_seg = 0 if len(abr_seg) == 0 else abr_seg.values[0]
    abr_ter = 0 if len(abr_ter) == 0 else abr_ter.values[0]
    abr_qua = 0 if len(abr_qua) == 0 else abr_qua.values[0]
    abr_qui = 0 if len(abr_qui) == 0 else abr_qui.values[0]
    abr_sex = 0 if len(abr_sex) == 0 else abr_sex.values[0]
    abr_sab = 0 if len(abr_sab) == 0 else abr_sab.values[0]
    abr_dom = 0 if len(abr_dom) == 0 else abr_dom.values[0]

    mai_seg = 0 if len(mai_seg) == 0 else mai_seg.values[0]
    mai_ter = 0 if len(mai_ter) == 0 else mai_ter.values[0]
    mai_qua = 0 if len(mai_qua) == 0 else mai_qua.values[0]
    mai_qui = 0 if len(mai_qui) == 0 else mai_qui.values[0]
    mai_sex = 0 if len(mai_sex) == 0 else mai_sex.values[0]
    mai_sab = 0 if len(mai_sab) == 0 else mai_sab.values[0]
    mai_dom = 0 if len(mai_dom) == 0 else mai_dom.values[0]

    jun_seg = 0 if len(jun_seg) == 0 else jun_seg.values[0]
    jun_ter = 0 if len(jun_ter) == 0 else jun_ter.values[0]
    jun_qua = 0 if len(jun_qua) == 0 else jun_qua.values[0]
    jun_qui = 0 if len(jun_qui) == 0 else jun_qui.values[0]
    jun_sex = 0 if len(jun_sex) == 0 else jun_sex.values[0]
    jun_sab = 0 if len(jun_sab) == 0 else jun_sab.values[0]
    jun_dom = 0 if len(jun_dom) == 0 else jun_dom.values[0]

    jul_seg = 0 if len(jul_seg) == 0 else jul_seg.values[0]
    jul_ter = 0 if len(jul_ter) == 0 else jul_ter.values[0]
    jul_qua = 0 if len(jul_qua) == 0 else jul_qua.values[0]
    jul_qui = 0 if len(jul_qui) == 0 else jul_qui.values[0]
    jul_sex = 0 if len(jul_sex) == 0 else jul_sex.values[0]
    jul_sab = 0 if len(jul_sab) == 0 else jul_sab.values[0]
    jul_dom = 0 if len(jul_dom) == 0 else jul_dom.values[0]

    ago_seg = 0 if len(ago_seg) == 0 else ago_seg.values[0]
    ago_ter = 0 if len(ago_ter) == 0 else ago_ter.values[0]
    ago_qua = 0 if len(ago_qua) == 0 else ago_qua.values[0]
    ago_qui = 0 if len(ago_qui) == 0 else ago_qui.values[0]
    ago_sex = 0 if len(ago_sex) == 0 else ago_sex.values[0]
    ago_sab = 0 if len(ago_sab) == 0 else ago_sab.values[0]
    ago_dom = 0 if len(ago_dom) == 0 else ago_dom.values[0]

    set_seg = 0 if len(set_seg) == 0 else set_seg.values[0]
    set_ter = 0 if len(set_ter) == 0 else set_ter.values[0]
    set_qua = 0 if len(set_qua) == 0 else set_qua.values[0]
    set_qui = 0 if len(set_qui) == 0 else set_qui.values[0]
    set_sex = 0 if len(set_sex) == 0 else set_sex.values[0]
    set_sab = 0 if len(set_sab) == 0 else set_sab.values[0]
    set_dom = 0 if len(set_dom) == 0 else set_dom.values[0]

    out_seg = 0 if len(out_seg) == 0 else out_seg.values[0]
    out_ter = 0 if len(out_ter) == 0 else out_ter.values[0]
    out_qua = 0 if len(out_qua) == 0 else out_qua.values[0]
    out_qui = 0 if len(out_qui) == 0 else out_qui.values[0]
    out_sex = 0 if len(out_sex) == 0 else out_sex.values[0]
    out_sab = 0 if len(out_sab) == 0 else out_sab.values[0]
    out_dom = 0 if len(out_dom) == 0 else out_dom.values[0]

    nov_seg = 0 if len(nov_seg) == 0 else nov_seg.values[0]
    nov_ter = 0 if len(nov_ter) == 0 else nov_ter.values[0]
    nov_qua = 0 if len(nov_qua) == 0 else nov_qua.values[0]
    nov_qui = 0 if len(nov_qui) == 0 else nov_qui.values[0]
    nov_sex = 0 if len(nov_sex) == 0 else nov_sex.values[0]
    nov_sab = 0 if len(nov_sab) == 0 else nov_sab.values[0]
    nov_dom = 0 if len(nov_dom) == 0 else nov_dom.values[0]

    dez_seg = 0 if len(dez_seg) == 0 else dez_seg.values[0]
    dez_ter = 0 if len(dez_ter) == 0 else dez_ter.values[0]
    dez_qua = 0 if len(dez_qua) == 0 else dez_qua.values[0]
    dez_qui = 0 if len(dez_qui) == 0 else dez_qui.values[0]
    dez_sex = 0 if len(dez_sex) == 0 else dez_sex.values[0]
    dez_sab = 0 if len(dez_sab) == 0 else dez_sab.values[0]
    dez_dom = 0 if len(dez_dom) == 0 else dez_dom.values[0]


    matriz = [
        [jan_sab, fev_sab, mar_sab, abr_sab, mai_sab, jun_sab, jul_sab, ago_sab, set_sab, out_sab, nov_sab, dez_sab],
        [jan_sex, fev_sex, mar_sex, abr_sex, mai_sex, jun_sex, jul_sex, ago_sex, set_sex, out_sex, nov_sex, dez_sex],
        [jan_qui, fev_qui, mar_qui, abr_qui, mai_qui, jun_qui, jul_qui, ago_qui, set_qui, out_qui, nov_qui, dez_qui],
        [jan_qua, fev_qua, mar_qua, abr_qua, mai_qua, jun_qua, jul_qua, ago_qua, set_qua, out_qua, nov_qua, dez_qua],
        [jan_ter, fev_ter, mar_ter, abr_ter, mai_ter, jun_ter, jul_ter, ago_ter, set_ter, out_ter, nov_ter, dez_ter],
        [jan_seg, fev_seg, mar_seg, abr_seg, mai_seg, jun_seg, jul_seg, ago_seg, set_seg, out_seg, nov_seg, dez_seg],
        [jan_dom, fev_dom, mar_dom, abr_dom, mai_dom, jun_dom, jul_dom, ago_dom, set_dom, out_dom, nov_dom, dez_dom],

    ]

    fig = go.Figure(data=go.Heatmap(
        z=matriz, name="", text=matriz,
        y=['Sábado', 'Sexta', 'Quinta', 'Quarta', 'Terça' , 'Segunda', 'Domingo'],
        x=['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'],
        texttemplate="%{text:,.0f}",
        hovertemplate="</br><b>Semana:</b> %{x}" +
                      "</br><b>Mês:</b> %{y}" +
                      "</br><b>Compras:</b> %{z:,.0f}",
        showscale=False,
        colorscale='Portland'))
    fig.update_layout(height=300, margin=dict(l=20, r=20, b=20, t=20),
                      paper_bgcolor="#F8F8FF", font={'size': 16})

    return fig




def plot_hotmap_semana(df):

    df_map = df.groupby(['Semana do ano', 'Semana'])['ID'].agg('count').reset_index().sort_values('Semana do ano',
                                                                                                  ascending=True)
    seg_31 = df_map[(df_map['Semana do ano'] == 2023.31) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_31 = df_map[(df_map['Semana do ano'] == 2023.31) & (df_map['Semana'] == 'Terça')]['ID']
    qua_31 = df_map[(df_map['Semana do ano'] == 2023.31) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_31 = df_map[(df_map['Semana do ano'] == 2023.31) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_31 = df_map[(df_map['Semana do ano'] == 2023.31) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_31 = df_map[(df_map['Semana do ano'] == 2023.31) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_31 = df_map[(df_map['Semana do ano'] == 2023.31) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_30 = df_map[(df_map['Semana do ano'] == 2023.30) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_30 = df_map[(df_map['Semana do ano'] == 2023.30) & (df_map['Semana'] == 'Terça')]['ID']
    qua_30 = df_map[(df_map['Semana do ano'] == 2023.30) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_30 = df_map[(df_map['Semana do ano'] == 2023.30) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_30 = df_map[(df_map['Semana do ano'] == 2023.30) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_30 = df_map[(df_map['Semana do ano'] == 2023.30) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_30 = df_map[(df_map['Semana do ano'] == 2023.30) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_29 = df_map[(df_map['Semana do ano'] == 2023.29) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_29 = df_map[(df_map['Semana do ano'] == 2023.29) & (df_map['Semana'] == 'Terça')]['ID']
    qua_29 = df_map[(df_map['Semana do ano'] == 2023.29) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_29 = df_map[(df_map['Semana do ano'] == 2023.29) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_29 = df_map[(df_map['Semana do ano'] == 2023.29) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_29 = df_map[(df_map['Semana do ano'] == 2023.29) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_29 = df_map[(df_map['Semana do ano'] == 2023.29) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_28 = df_map[(df_map['Semana do ano'] == 2023.28) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_28 = df_map[(df_map['Semana do ano'] == 2023.28) & (df_map['Semana'] == 'Terça')]['ID']
    qua_28 = df_map[(df_map['Semana do ano'] == 2023.28) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_28 = df_map[(df_map['Semana do ano'] == 2023.28) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_28 = df_map[(df_map['Semana do ano'] == 2023.28) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_28 = df_map[(df_map['Semana do ano'] == 2023.28) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_28 = df_map[(df_map['Semana do ano'] == 2023.28) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_27 = df_map[(df_map['Semana do ano'] == 2023.27) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_27 = df_map[(df_map['Semana do ano'] == 2023.27) & (df_map['Semana'] == 'Terça')]['ID']
    qua_27 = df_map[(df_map['Semana do ano'] == 2023.27) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_27 = df_map[(df_map['Semana do ano'] == 2023.27) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_27 = df_map[(df_map['Semana do ano'] == 2023.27) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_27 = df_map[(df_map['Semana do ano'] == 2023.27) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_27 = df_map[(df_map['Semana do ano'] == 2023.27) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_26 = df_map[(df_map['Semana do ano'] == 2023.26) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_26 = df_map[(df_map['Semana do ano'] == 2023.26) & (df_map['Semana'] == 'Terça')]['ID']
    qua_26 = df_map[(df_map['Semana do ano'] == 2023.26) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_26 = df_map[(df_map['Semana do ano'] == 2023.26) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_26 = df_map[(df_map['Semana do ano'] == 2023.26) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_26 = df_map[(df_map['Semana do ano'] == 2023.26) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_26 = df_map[(df_map['Semana do ano'] == 2023.26) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_25 = df_map[(df_map['Semana do ano'] == 2023.25) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_25 = df_map[(df_map['Semana do ano'] == 2023.25) & (df_map['Semana'] == 'Terça')]['ID']
    qua_25 = df_map[(df_map['Semana do ano'] == 2023.25) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_25 = df_map[(df_map['Semana do ano'] == 2023.25) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_25 = df_map[(df_map['Semana do ano'] == 2023.25) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_25 = df_map[(df_map['Semana do ano'] == 2023.25) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_25 = df_map[(df_map['Semana do ano'] == 2023.25) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_24 = df_map[(df_map['Semana do ano'] == 2023.24) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_24 = df_map[(df_map['Semana do ano'] == 2023.24) & (df_map['Semana'] == 'Terça')]['ID']
    qua_24 = df_map[(df_map['Semana do ano'] == 2023.24) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_24 = df_map[(df_map['Semana do ano'] == 2023.24) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_24 = df_map[(df_map['Semana do ano'] == 2023.24) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_24 = df_map[(df_map['Semana do ano'] == 2023.24) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_24 = df_map[(df_map['Semana do ano'] == 2023.24) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_23 = df_map[(df_map['Semana do ano'] == 2023.23) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_23 = df_map[(df_map['Semana do ano'] == 2023.23) & (df_map['Semana'] == 'Terça')]['ID']
    qua_23 = df_map[(df_map['Semana do ano'] == 2023.23) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_23 = df_map[(df_map['Semana do ano'] == 2023.23) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_23 = df_map[(df_map['Semana do ano'] == 2023.23) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_23 = df_map[(df_map['Semana do ano'] == 2023.23) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_23 = df_map[(df_map['Semana do ano'] == 2023.23) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_22 = df_map[(df_map['Semana do ano'] == 2023.22) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_22 = df_map[(df_map['Semana do ano'] == 2023.22) & (df_map['Semana'] == 'Terça')]['ID']
    qua_22 = df_map[(df_map['Semana do ano'] == 2023.22) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_22 = df_map[(df_map['Semana do ano'] == 2023.22) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_22 = df_map[(df_map['Semana do ano'] == 2023.22) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_22 = df_map[(df_map['Semana do ano'] == 2023.22) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_22 = df_map[(df_map['Semana do ano'] == 2023.22) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_21 = df_map[(df_map['Semana do ano'] == 2023.21) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_21 = df_map[(df_map['Semana do ano'] == 2023.21) & (df_map['Semana'] == 'Terça')]['ID']
    qua_21 = df_map[(df_map['Semana do ano'] == 2023.21) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_21 = df_map[(df_map['Semana do ano'] == 2023.21) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_21 = df_map[(df_map['Semana do ano'] == 2023.21) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_21 = df_map[(df_map['Semana do ano'] == 2023.21) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_21 = df_map[(df_map['Semana do ano'] == 2023.21) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_20 = df_map[(df_map['Semana do ano'] == 2023.20) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_20 = df_map[(df_map['Semana do ano'] == 2023.20) & (df_map['Semana'] == 'Terça')]['ID']
    qua_20 = df_map[(df_map['Semana do ano'] == 2023.20) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_20 = df_map[(df_map['Semana do ano'] == 2023.20) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_20 = df_map[(df_map['Semana do ano'] == 2023.20) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_20 = df_map[(df_map['Semana do ano'] == 2023.20) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_20 = df_map[(df_map['Semana do ano'] == 2023.20) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_19 = df_map[(df_map['Semana do ano'] == 2023.19) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_19 = df_map[(df_map['Semana do ano'] == 2023.19) & (df_map['Semana'] == 'Terça')]['ID']
    qua_19 = df_map[(df_map['Semana do ano'] == 2023.19) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_19 = df_map[(df_map['Semana do ano'] == 2023.19) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_19 = df_map[(df_map['Semana do ano'] == 2023.19) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_19 = df_map[(df_map['Semana do ano'] == 2023.19) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_19 = df_map[(df_map['Semana do ano'] == 2023.19) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_18 = df_map[(df_map['Semana do ano'] == 2023.18) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_18 = df_map[(df_map['Semana do ano'] == 2023.18) & (df_map['Semana'] == 'Terça')]['ID']
    qua_18 = df_map[(df_map['Semana do ano'] == 2023.18) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_18 = df_map[(df_map['Semana do ano'] == 2023.18) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_18 = df_map[(df_map['Semana do ano'] == 2023.18) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_18 = df_map[(df_map['Semana do ano'] == 2023.18) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_18 = df_map[(df_map['Semana do ano'] == 2023.18) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_17 = df_map[(df_map['Semana do ano'] == 2023.17) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_17 = df_map[(df_map['Semana do ano'] == 2023.17) & (df_map['Semana'] == 'Terça')]['ID']
    qua_17 = df_map[(df_map['Semana do ano'] == 2023.17) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_17 = df_map[(df_map['Semana do ano'] == 2023.17) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_17 = df_map[(df_map['Semana do ano'] == 2023.17) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_17 = df_map[(df_map['Semana do ano'] == 2023.17) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_17 = df_map[(df_map['Semana do ano'] == 2023.17) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_16 = df_map[(df_map['Semana do ano'] == 2023.16) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_16 = df_map[(df_map['Semana do ano'] == 2023.16) & (df_map['Semana'] == 'Terça')]['ID']
    qua_16 = df_map[(df_map['Semana do ano'] == 2023.16) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_16 = df_map[(df_map['Semana do ano'] == 2023.16) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_16 = df_map[(df_map['Semana do ano'] == 2023.16) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_16 = df_map[(df_map['Semana do ano'] == 2023.16) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_16 = df_map[(df_map['Semana do ano'] == 2023.16) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_15 = df_map[(df_map['Semana do ano'] == 2023.15) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_15 = df_map[(df_map['Semana do ano'] == 2023.15) & (df_map['Semana'] == 'Terça')]['ID']
    qua_15 = df_map[(df_map['Semana do ano'] == 2023.15) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_15 = df_map[(df_map['Semana do ano'] == 2023.15) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_15 = df_map[(df_map['Semana do ano'] == 2023.15) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_15 = df_map[(df_map['Semana do ano'] == 2023.15) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_15 = df_map[(df_map['Semana do ano'] == 2023.15) & (df_map['Semana'] == 'Sábado')]['ID']


    seg_14 = df_map[(df_map['Semana do ano'] == 2023.14) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_14 = df_map[(df_map['Semana do ano'] == 2023.14) & (df_map['Semana'] == 'Terça')]['ID']
    qua_14 = df_map[(df_map['Semana do ano'] == 2023.14) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_14 = df_map[(df_map['Semana do ano'] == 2023.14) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_14 = df_map[(df_map['Semana do ano'] == 2023.14) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_14 = df_map[(df_map['Semana do ano'] == 2023.14) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_14 = df_map[(df_map['Semana do ano'] == 2023.14) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_13 = df_map[(df_map['Semana do ano'] == 2023.13) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_13 = df_map[(df_map['Semana do ano'] == 2023.13) & (df_map['Semana'] == 'Terça')]['ID']
    qua_13 = df_map[(df_map['Semana do ano'] == 2023.13) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_13 = df_map[(df_map['Semana do ano'] == 2023.13) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_13 = df_map[(df_map['Semana do ano'] == 2023.13) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_13 = df_map[(df_map['Semana do ano'] == 2023.13) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_13 = df_map[(df_map['Semana do ano'] == 2023.13) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_12 = df_map[(df_map['Semana do ano'] == 2023.12) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_12 = df_map[(df_map['Semana do ano'] == 2023.12) & (df_map['Semana'] == 'Terça')]['ID']
    qua_12 = df_map[(df_map['Semana do ano'] == 2023.12) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_12 = df_map[(df_map['Semana do ano'] == 2023.12) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_12 = df_map[(df_map['Semana do ano'] == 2023.12) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_12 = df_map[(df_map['Semana do ano'] == 2023.12) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_12 = df_map[(df_map['Semana do ano'] == 2023.12) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_11 = df_map[(df_map['Semana do ano'] == 2023.11) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_11 = df_map[(df_map['Semana do ano'] == 2023.11) & (df_map['Semana'] == 'Terça')]['ID']
    qua_11 = df_map[(df_map['Semana do ano'] == 2023.11) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_11 = df_map[(df_map['Semana do ano'] == 2023.11) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_11 = df_map[(df_map['Semana do ano'] == 2023.11) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_11 = df_map[(df_map['Semana do ano'] == 2023.11) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_11 = df_map[(df_map['Semana do ano'] == 2023.11) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_10 = df_map[(df_map['Semana do ano'] == 2023.10) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_10 = df_map[(df_map['Semana do ano'] == 2023.10) & (df_map['Semana'] == 'Terça')]['ID']
    qua_10 = df_map[(df_map['Semana do ano'] == 2023.10) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_10 = df_map[(df_map['Semana do ano'] == 2023.10) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_10 = df_map[(df_map['Semana do ano'] == 2023.10) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_10 = df_map[(df_map['Semana do ano'] == 2023.10) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_10 = df_map[(df_map['Semana do ano'] == 2023.10) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_09 = df_map[(df_map['Semana do ano'] == 2023.09) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_09 = df_map[(df_map['Semana do ano'] == 2023.09) & (df_map['Semana'] == 'Terça')]['ID']
    qua_09 = df_map[(df_map['Semana do ano'] == 2023.09) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_09 = df_map[(df_map['Semana do ano'] == 2023.09) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_09 = df_map[(df_map['Semana do ano'] == 2023.09) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_09 = df_map[(df_map['Semana do ano'] == 2023.09) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_09 = df_map[(df_map['Semana do ano'] == 2023.09) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_08 = df_map[(df_map['Semana do ano'] == 2023.08) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_08 = df_map[(df_map['Semana do ano'] == 2023.08) & (df_map['Semana'] == 'Terça')]['ID']
    qua_08 = df_map[(df_map['Semana do ano'] == 2023.08) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_08 = df_map[(df_map['Semana do ano'] == 2023.08) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_08 = df_map[(df_map['Semana do ano'] == 2023.08) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_08 = df_map[(df_map['Semana do ano'] == 2023.08) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_08 = df_map[(df_map['Semana do ano'] == 2023.08) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_07 = df_map[(df_map['Semana do ano'] == 2023.07) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_07 = df_map[(df_map['Semana do ano'] == 2023.07) & (df_map['Semana'] == 'Terça')]['ID']
    qua_07 = df_map[(df_map['Semana do ano'] == 2023.07) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_07 = df_map[(df_map['Semana do ano'] == 2023.07) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_07 = df_map[(df_map['Semana do ano'] == 2023.07) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_07 = df_map[(df_map['Semana do ano'] == 2023.07) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_07 = df_map[(df_map['Semana do ano'] == 2023.07) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_06 = df_map[(df_map['Semana do ano'] == 2023.06) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_06 = df_map[(df_map['Semana do ano'] == 2023.06) & (df_map['Semana'] == 'Terça')]['ID']
    qua_06 = df_map[(df_map['Semana do ano'] == 2023.06) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_06 = df_map[(df_map['Semana do ano'] == 2023.06) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_06 = df_map[(df_map['Semana do ano'] == 2023.06) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_06 = df_map[(df_map['Semana do ano'] == 2023.06) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_06 = df_map[(df_map['Semana do ano'] == 2023.06) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_05 = df_map[(df_map['Semana do ano'] == 2023.05) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_05 = df_map[(df_map['Semana do ano'] == 2023.05) & (df_map['Semana'] == 'Terça')]['ID']
    qua_05 = df_map[(df_map['Semana do ano'] == 2023.05) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_05 = df_map[(df_map['Semana do ano'] == 2023.05) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_05 = df_map[(df_map['Semana do ano'] == 2023.05) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_05 = df_map[(df_map['Semana do ano'] == 2023.05) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_05 = df_map[(df_map['Semana do ano'] == 2023.05) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_04 = df_map[(df_map['Semana do ano'] == 2023.04) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_04 = df_map[(df_map['Semana do ano'] == 2023.04) & (df_map['Semana'] == 'Terça')]['ID']
    qua_04 = df_map[(df_map['Semana do ano'] == 2023.04) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_04 = df_map[(df_map['Semana do ano'] == 2023.04) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_04 = df_map[(df_map['Semana do ano'] == 2023.04) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_04 = df_map[(df_map['Semana do ano'] == 2023.04) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_04 = df_map[(df_map['Semana do ano'] == 2023.04) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_03 = df_map[(df_map['Semana do ano'] == 2023.03) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_03 = df_map[(df_map['Semana do ano'] == 2023.03) & (df_map['Semana'] == 'Terça')]['ID']
    qua_03 = df_map[(df_map['Semana do ano'] == 2023.03) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_03 = df_map[(df_map['Semana do ano'] == 2023.03) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_03 = df_map[(df_map['Semana do ano'] == 2023.03) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_03 = df_map[(df_map['Semana do ano'] == 2023.03) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_03 = df_map[(df_map['Semana do ano'] == 2023.03) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_02 = df_map[(df_map['Semana do ano'] == 2023.02) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_02 = df_map[(df_map['Semana do ano'] == 2023.02) & (df_map['Semana'] == 'Terça')]['ID']
    qua_02 = df_map[(df_map['Semana do ano'] == 2023.02) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_02 = df_map[(df_map['Semana do ano'] == 2023.02) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_02 = df_map[(df_map['Semana do ano'] == 2023.02) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_02 = df_map[(df_map['Semana do ano'] == 2023.02) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_02 = df_map[(df_map['Semana do ano'] == 2023.02) & (df_map['Semana'] == 'Sábado')]['ID']

    seg_01 = df_map[(df_map['Semana do ano'] == 2023.01) & (df_map['Semana'] == 'Segunda')]['ID']
    ter_01 = df_map[(df_map['Semana do ano'] == 2023.01) & (df_map['Semana'] == 'Terça')]['ID']
    qua_01 = df_map[(df_map['Semana do ano'] == 2023.01) & (df_map['Semana'] == 'Quarta')]['ID']
    qui_01 = df_map[(df_map['Semana do ano'] == 2023.01) & (df_map['Semana'] == 'Quinta')]['ID']
    sex_01 = df_map[(df_map['Semana do ano'] == 2023.01) & (df_map['Semana'] == 'Sexta')]['ID']
    sab_01 = df_map[(df_map['Semana do ano'] == 2023.01) & (df_map['Semana'] == 'Sábado')]['ID']
    dom_01 = df_map[(df_map['Semana do ano'] == 2023.01) & (df_map['Semana'] == 'Sábado')]['ID']



    seg_31 = 0 if len(seg_31) == 0 else seg_31.values[0]
    ter_31 = 0 if len(ter_31) == 0 else ter_31.values[0]
    qua_31 = 0 if len(qua_31) == 0 else qua_31.values[0]
    qui_31 = 0 if len(qui_31) == 0 else qui_31.values[0]
    sex_31 = 0 if len(sex_31) == 0 else sex_31.values[0]
    sab_31 = 0 if len(sab_31) == 0 else sab_31.values[0]
    dom_31 = 0 if len(dom_31) == 0 else dom_31.values[0]

    seg_30 = 0 if len(seg_30) == 0 else seg_30.values[0]
    ter_30 = 0 if len(ter_30) == 0 else ter_30.values[0]
    qua_30 = 0 if len(qua_30) == 0 else qua_30.values[0]
    qui_30 = 0 if len(qui_30) == 0 else qui_30.values[0]
    sex_30 = 0 if len(sex_30) == 0 else sex_30.values[0]
    sab_30 = 0 if len(sab_30) == 0 else sab_30.values[0]
    dom_30 = 0 if len(dom_30) == 0 else dom_30.values[0]

    seg_29 = 0 if len(seg_29) == 0 else seg_29.values[0]
    ter_29 = 0 if len(ter_29) == 0 else ter_29.values[0]
    qua_29 = 0 if len(qua_29) == 0 else qua_29.values[0]
    qui_29 = 0 if len(qui_29) == 0 else qui_29.values[0]
    sex_29 = 0 if len(sex_29) == 0 else sex_29.values[0]
    sab_29 = 0 if len(sab_29) == 0 else sab_29.values[0]
    dom_29 = 0 if len(dom_29) == 0 else dom_29.values[0]

    seg_28 = 0 if len(seg_28) == 0 else seg_28.values[0]
    ter_28 = 0 if len(ter_28) == 0 else ter_28.values[0]
    qua_28 = 0 if len(qua_28) == 0 else qua_28.values[0]
    qui_28 = 0 if len(qui_28) == 0 else qui_28.values[0]
    sex_28 = 0 if len(sex_28) == 0 else sex_28.values[0]
    sab_28 = 0 if len(sab_28) == 0 else sab_28.values[0]
    dom_28 = 0 if len(dom_28) == 0 else dom_28.values[0]

    seg_27 = 0 if len(seg_27) == 0 else seg_27.values[0]
    ter_27 = 0 if len(ter_27) == 0 else ter_27.values[0]
    qua_27 = 0 if len(qua_27) == 0 else qua_27.values[0]
    qui_27 = 0 if len(qui_27) == 0 else qui_27.values[0]
    sex_27 = 0 if len(sex_27) == 0 else sex_27.values[0]
    sab_27 = 0 if len(sab_27) == 0 else sab_27.values[0]
    dom_27 = 0 if len(dom_27) == 0 else dom_27.values[0]

    seg_26 = 0 if len(seg_26) == 0 else seg_26.values[0]
    ter_26 = 0 if len(ter_26) == 0 else ter_26.values[0]
    qua_26 = 0 if len(qua_26) == 0 else qua_26.values[0]
    qui_26 = 0 if len(qui_26) == 0 else qui_26.values[0]
    sex_26 = 0 if len(sex_26) == 0 else sex_26.values[0]
    sab_26 = 0 if len(sab_26) == 0 else sab_26.values[0]
    dom_26 = 0 if len(dom_26) == 0 else dom_26.values[0]

    seg_25 = 0 if len(seg_25) == 0 else seg_25.values[0]
    ter_25 = 0 if len(ter_25) == 0 else ter_25.values[0]
    qua_25 = 0 if len(qua_25) == 0 else qua_25.values[0]
    qui_25 = 0 if len(qui_25) == 0 else qui_25.values[0]
    sex_25 = 0 if len(sex_25) == 0 else sex_25.values[0]
    sab_25 = 0 if len(sab_25) == 0 else sab_25.values[0]
    dom_25 = 0 if len(dom_25) == 0 else dom_25.values[0]

    seg_24 = 0 if len(seg_24) == 0 else seg_24.values[0]
    ter_24 = 0 if len(ter_24) == 0 else ter_24.values[0]
    qua_24 = 0 if len(qua_24) == 0 else qua_24.values[0]
    qui_24 = 0 if len(qui_24) == 0 else qui_24.values[0]
    sex_24 = 0 if len(sex_24) == 0 else sex_24.values[0]
    sab_24 = 0 if len(sab_24) == 0 else sab_24.values[0]
    dom_24 = 0 if len(dom_24) == 0 else dom_24.values[0]

    seg_23 = 0 if len(seg_23) == 0 else seg_23.values[0]
    ter_23 = 0 if len(ter_23) == 0 else ter_23.values[0]
    qua_23 = 0 if len(qua_23) == 0 else qua_23.values[0]
    qui_23 = 0 if len(qui_23) == 0 else qui_23.values[0]
    sex_23 = 0 if len(sex_23) == 0 else sex_23.values[0]
    sab_23 = 0 if len(sab_23) == 0 else sab_23.values[0]
    dom_23 = 0 if len(dom_23) == 0 else dom_23.values[0]

    seg_22 = 0 if len(seg_22) == 0 else seg_22.values[0]
    ter_22 = 0 if len(ter_22) == 0 else ter_22.values[0]
    qua_22 = 0 if len(qua_22) == 0 else qua_22.values[0]
    qui_22 = 0 if len(qui_22) == 0 else qui_22.values[0]
    sex_22 = 0 if len(sex_22) == 0 else sex_22.values[0]
    sab_22 = 0 if len(sab_22) == 0 else sab_22.values[0]
    dom_22 = 0 if len(dom_22) == 0 else dom_22.values[0]

    seg_21 = 0 if len(seg_21) == 0 else seg_21.values[0]
    ter_21 = 0 if len(ter_21) == 0 else ter_21.values[0]
    qua_21 = 0 if len(qua_21) == 0 else qua_21.values[0]
    qui_21 = 0 if len(qui_21) == 0 else qui_21.values[0]
    sex_21 = 0 if len(sex_21) == 0 else sex_21.values[0]
    sab_21 = 0 if len(sab_21) == 0 else sab_21.values[0]
    dom_21 = 0 if len(dom_21) == 0 else dom_21.values[0]

    seg_20 = 0 if len(seg_20) == 0 else seg_20.values[0]
    ter_20 = 0 if len(ter_20) == 0 else ter_20.values[0]
    qua_20 = 0 if len(qua_20) == 0 else qua_20.values[0]
    qui_20 = 0 if len(qui_20) == 0 else qui_20.values[0]
    sex_20 = 0 if len(sex_20) == 0 else sex_20.values[0]
    sab_20 = 0 if len(sab_20) == 0 else sab_20.values[0]
    dom_20 = 0 if len(dom_20) == 0 else dom_20.values[0]

    seg_19 = 0 if len(seg_19) == 0 else seg_19.values[0]
    ter_19 = 0 if len(ter_19) == 0 else ter_19.values[0]
    qua_19 = 0 if len(qua_19) == 0 else qua_19.values[0]
    qui_19 = 0 if len(qui_19) == 0 else qui_19.values[0]
    sex_19 = 0 if len(sex_19) == 0 else sex_19.values[0]
    sab_19 = 0 if len(sab_19) == 0 else sab_19.values[0]
    dom_19 = 0 if len(dom_19) == 0 else dom_19.values[0]

    seg_18 = 0 if len(seg_18) == 0 else seg_18.values[0]
    ter_18 = 0 if len(ter_18) == 0 else ter_18.values[0]
    qua_18 = 0 if len(qua_18) == 0 else qua_18.values[0]
    qui_18 = 0 if len(qui_18) == 0 else qui_18.values[0]
    sex_18 = 0 if len(sex_18) == 0 else sex_18.values[0]
    sab_18 = 0 if len(sab_18) == 0 else sab_18.values[0]
    dom_18 = 0 if len(dom_18) == 0 else dom_18.values[0]

    seg_17 = 0 if len(seg_17) == 0 else seg_17.values[0]
    ter_17 = 0 if len(ter_17) == 0 else ter_17.values[0]
    qua_17 = 0 if len(qua_17) == 0 else qua_17.values[0]
    qui_17 = 0 if len(qui_17) == 0 else qui_17.values[0]
    sex_17 = 0 if len(sex_17) == 0 else sex_17.values[0]
    sab_17 = 0 if len(sab_17) == 0 else sab_17.values[0]
    dom_17 = 0 if len(dom_17) == 0 else dom_17.values[0]

    seg_16 = 0 if len(seg_16) == 0 else seg_16.values[0]
    ter_16 = 0 if len(ter_16) == 0 else ter_16.values[0]
    qua_16 = 0 if len(qua_16) == 0 else qua_16.values[0]
    qui_16 = 0 if len(qui_16) == 0 else qui_16.values[0]
    sex_16 = 0 if len(sex_16) == 0 else sex_16.values[0]
    sab_16 = 0 if len(sab_16) == 0 else sab_16.values[0]
    dom_16 = 0 if len(dom_16) == 0 else dom_16.values[0]

    seg_15 = 0 if len(seg_15) == 0 else seg_15.values[0]
    ter_15 = 0 if len(ter_15) == 0 else ter_15.values[0]
    qua_15 = 0 if len(qua_15) == 0 else qua_15.values[0]
    qui_15 = 0 if len(qui_15) == 0 else qui_15.values[0]
    sex_15 = 0 if len(sex_15) == 0 else sex_15.values[0]
    sab_15 = 0 if len(sab_15) == 0 else sab_15.values[0]
    dom_15 = 0 if len(dom_15) == 0 else dom_15.values[0]

    seg_14 = 0 if len(seg_14) == 0 else seg_14.values[0]
    ter_14 = 0 if len(ter_14) == 0 else ter_14.values[0]
    qua_14 = 0 if len(qua_14) == 0 else qua_14.values[0]
    qui_14 = 0 if len(qui_14) == 0 else qui_14.values[0]
    sex_14 = 0 if len(sex_14) == 0 else sex_14.values[0]
    sab_14 = 0 if len(sab_14) == 0 else sab_14.values[0]
    dom_14 = 0 if len(dom_14) == 0 else dom_14.values[0]

    seg_13 = 0 if len(seg_13) == 0 else seg_13.values[0]
    ter_13 = 0 if len(ter_13) == 0 else ter_13.values[0]
    qua_13 = 0 if len(qua_13) == 0 else qua_13.values[0]
    qui_13 = 0 if len(qui_13) == 0 else qui_13.values[0]
    sex_13 = 0 if len(sex_13) == 0 else sex_13.values[0]
    sab_13 = 0 if len(sab_13) == 0 else sab_13.values[0]
    dom_13 = 0 if len(dom_13) == 0 else dom_13.values[0]

    seg_12 = 0 if len(seg_12) == 0 else seg_12.values[0]
    ter_12 = 0 if len(ter_12) == 0 else ter_12.values[0]
    qua_12 = 0 if len(qua_12) == 0 else qua_12.values[0]
    qui_12 = 0 if len(qui_12) == 0 else qui_12.values[0]
    sex_12 = 0 if len(sex_12) == 0 else sex_12.values[0]
    sab_12 = 0 if len(sab_12) == 0 else sab_12.values[0]
    dom_12 = 0 if len(dom_12) == 0 else dom_12.values[0]

    seg_11 = 0 if len(seg_11) == 0 else seg_11.values[0]
    ter_11 = 0 if len(ter_11) == 0 else ter_11.values[0]
    qua_11 = 0 if len(qua_11) == 0 else qua_11.values[0]
    qui_11 = 0 if len(qui_11) == 0 else qui_11.values[0]
    sex_11 = 0 if len(sex_11) == 0 else sex_11.values[0]
    sab_11 = 0 if len(sab_11) == 0 else sab_11.values[0]
    dom_11 = 0 if len(dom_11) == 0 else dom_11.values[0]

    seg_10 = 0 if len(seg_10) == 0 else seg_10.values[0]
    ter_10 = 0 if len(ter_10) == 0 else ter_10.values[0]
    qua_10 = 0 if len(qua_10) == 0 else qua_10.values[0]
    qui_10 = 0 if len(qui_10) == 0 else qui_10.values[0]
    sex_10 = 0 if len(sex_10) == 0 else sex_10.values[0]
    sab_10 = 0 if len(sab_10) == 0 else sab_10.values[0]
    dom_10 = 0 if len(dom_10) == 0 else dom_10.values[0]

    seg_09 = 0 if len(seg_09) == 0 else seg_09.values[0]
    ter_09 = 0 if len(ter_09) == 0 else ter_09.values[0]
    qua_09 = 0 if len(qua_09) == 0 else qua_09.values[0]
    qui_09 = 0 if len(qui_09) == 0 else qui_09.values[0]
    sex_09 = 0 if len(sex_09) == 0 else sex_09.values[0]
    sab_09 = 0 if len(sab_09) == 0 else sab_09.values[0]
    dom_09 = 0 if len(dom_09) == 0 else dom_09.values[0]

    seg_08 = 0 if len(seg_08) == 0 else seg_08.values[0]
    ter_08 = 0 if len(ter_08) == 0 else ter_08.values[0]
    qua_08 = 0 if len(qua_08) == 0 else qua_08.values[0]
    qui_08 = 0 if len(qui_08) == 0 else qui_08.values[0]
    sex_08 = 0 if len(sex_08) == 0 else sex_08.values[0]
    sab_08 = 0 if len(sab_08) == 0 else sab_08.values[0]
    dom_08 = 0 if len(dom_08) == 0 else dom_08.values[0]

    seg_07 = 0 if len(seg_07) == 0 else seg_07.values[0]
    ter_07 = 0 if len(ter_07) == 0 else ter_07.values[0]
    qua_07 = 0 if len(qua_07) == 0 else qua_07.values[0]
    qui_07 = 0 if len(qui_07) == 0 else qui_07.values[0]
    sex_07 = 0 if len(sex_07) == 0 else sex_07.values[0]
    sab_07 = 0 if len(sab_07) == 0 else sab_07.values[0]
    dom_07 = 0 if len(dom_07) == 0 else dom_07.values[0]

    seg_06 = 0 if len(seg_06) == 0 else seg_06.values[0]
    ter_06 = 0 if len(ter_06) == 0 else ter_06.values[0]
    qua_06 = 0 if len(qua_06) == 0 else qua_06.values[0]
    qui_06 = 0 if len(qui_06) == 0 else qui_06.values[0]
    sex_06 = 0 if len(sex_06) == 0 else sex_06.values[0]
    sab_06 = 0 if len(sab_06) == 0 else sab_06.values[0]
    dom_06 = 0 if len(dom_06) == 0 else dom_06.values[0]

    seg_05 = 0 if len(seg_05) == 0 else seg_05.values[0]
    ter_05 = 0 if len(ter_05) == 0 else ter_05.values[0]
    qua_05 = 0 if len(qua_05) == 0 else qua_05.values[0]
    qui_05 = 0 if len(qui_05) == 0 else qui_05.values[0]
    sex_05 = 0 if len(sex_05) == 0 else sex_05.values[0]
    sab_05 = 0 if len(sab_05) == 0 else sab_05.values[0]
    dom_05 = 0 if len(dom_05) == 0 else dom_05.values[0]

    seg_04 = 0 if len(seg_04) == 0 else seg_04.values[0]
    ter_04 = 0 if len(ter_04) == 0 else ter_04.values[0]
    qua_04 = 0 if len(qua_04) == 0 else qua_04.values[0]
    qui_04 = 0 if len(qui_04) == 0 else qui_04.values[0]
    sex_04 = 0 if len(sex_04) == 0 else sex_04.values[0]
    sab_04 = 0 if len(sab_04) == 0 else sab_04.values[0]
    dom_04 = 0 if len(dom_04) == 0 else dom_04.values[0]

    seg_03 = 0 if len(seg_03) == 0 else seg_03.values[0]
    ter_03 = 0 if len(ter_03) == 0 else ter_03.values[0]
    qua_03 = 0 if len(qua_03) == 0 else qua_03.values[0]
    qui_03 = 0 if len(qui_03) == 0 else qui_03.values[0]
    sex_03 = 0 if len(sex_03) == 0 else sex_03.values[0]
    sab_03 = 0 if len(sab_03) == 0 else sab_03.values[0]
    dom_03 = 0 if len(dom_03) == 0 else dom_03.values[0]

    seg_02 = 0 if len(seg_02) == 0 else seg_02.values[0]
    ter_02 = 0 if len(ter_02) == 0 else ter_02.values[0]
    qua_02 = 0 if len(qua_02) == 0 else qua_02.values[0]
    qui_02 = 0 if len(qui_02) == 0 else qui_02.values[0]
    sex_02 = 0 if len(sex_02) == 0 else sex_02.values[0]
    sab_02 = 0 if len(sab_02) == 0 else sab_02.values[0]
    dom_02 = 0 if len(dom_02) == 0 else dom_02.values[0]

    seg_01 = 0 if len(seg_01) == 0 else seg_01.values[0]
    ter_01 = 0 if len(ter_01) == 0 else ter_01.values[0]
    qua_01 = 0 if len(qua_01) == 0 else qua_01.values[0]
    qui_01 = 0 if len(qui_01) == 0 else qui_01.values[0]
    sex_01 = 0 if len(sex_01) == 0 else sex_01.values[0]
    sab_01 = 0 if len(sab_01) == 0 else sab_01.values[0]
    dom_01 = 0 if len(dom_01) == 0 else dom_01.values[0]




    matriz = [
        [sab_01, sab_02, sab_03, sab_04, sab_05, sab_06, sab_07, sab_08, sab_09,
         sab_10, sab_11, sab_12, sab_13, sab_14, sab_15, sab_16, sab_17, sab_18, sab_19,
         sab_20, sab_21, sab_22, sab_23, sab_24, sab_25, sab_26, sab_27, sab_28, sab_29, sab_30, sab_31],

        [sex_01, sex_02, sex_03, sex_04, sex_05, sex_06, sex_07, sex_08, sex_09,
         sex_10, sex_11, sex_12, sex_13, sex_14, sex_15, sex_16, sex_17, sex_18, sex_19,
         sex_20, sex_21, sex_22, sex_23, sex_24, sex_25, sex_26, sex_27, sex_28, sex_29, sex_30, sex_31],

        [qui_01, qui_02, qui_03, qui_04, qui_05, qui_06, qui_07, qui_08, qui_09,
         qui_10, qui_11, qui_12, qui_13, qui_14, qui_15, qui_16, qui_17, qui_18, qui_19,
         qui_20, qui_21, qui_22, qui_23, qui_24, qui_25, qui_26, qui_27, qui_28, qui_29, qui_30, qui_31],

        [qua_01, qua_02, qua_03, qua_04, qua_05, qua_06, qua_07, qua_08, qua_09,
         qua_10, qua_11, qua_12, qua_13, qua_14, qua_15, qua_16, qua_17, qua_18, qua_19,
         qua_20, qua_21, qua_22, qua_23, qua_24, qua_25, qua_26, qua_27, qua_28, qua_29, qua_30, qua_31],

        [ter_01, ter_02, ter_03, ter_04, ter_05, ter_06, ter_07, ter_08, ter_09,
         ter_10, ter_11, ter_12, ter_13, ter_14, ter_15, ter_16, ter_17, ter_18, ter_19,
         ter_20, ter_21, ter_22, ter_23, ter_24, ter_25, ter_26, ter_27, ter_28, ter_29, ter_30, ter_31],

        [seg_01, seg_02, seg_03, seg_04, seg_05, seg_06, seg_07, seg_08, seg_09,
         seg_10, seg_11, seg_12, seg_13, seg_14, seg_15, seg_16, seg_17, seg_18, seg_19,
         seg_20, seg_21, seg_22, seg_23, seg_24, seg_25, seg_26, seg_27, seg_28, seg_29, seg_30, seg_31],

        [dom_01, dom_02, dom_03, dom_04, dom_05, dom_06, dom_07, dom_08, dom_09,
         dom_10, dom_11, dom_12, dom_13, dom_14, dom_15, dom_16, dom_17, dom_18, dom_19,
         dom_20, dom_21, dom_22, dom_23, dom_24, dom_25, dom_26, dom_27, dom_28, dom_29, dom_30, dom_31],


    ]

    x = [
        'S-01', 'S-02', 'S-03', 'S-04', 'S-05', 'S-06', 'S-07', 'S-08', 'S-09',
        'S-10', 'S-11', 'S-12', 'S-13', 'S-14', 'S-15', 'S-16', 'S-17', 'S-18', 'S-19',
        'S-20', 'S-21', 'S-22', 'S-23', 'S-24', 'S-25', 'S-26', 'S-27', 'S-28', 'S-29','S-30','S-31']

    fig = go.Figure(data=go.Heatmap(
        z=matriz, name="", text=matriz,
        x=x, y=['Sábado', 'Sexta', 'Quinta', 'Quarta', 'Terça' , 'Segunda', 'Domingo'],
        texttemplate="%{text:,.0f}",
        hovertemplate="</br><b>Semana do ano:</b> %{x}" +
                      "</br><b>Dia da semana:</b> %{y}" +
                      "</br><b>Pedidos:</b> %{z:,.0f}",
        colorscale='Portland'))
    fig.update_layout(height=200, margin=dict(l=20, r=20, b=20, t=20),
                      paper_bgcolor="#F8F8FF", font={'size': 16})

    return fig


def linha_diff_data(df, metrica):


    media_total = df[metrica].median()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Data'], y=df[metrica],
        mode='lines+markers',
        name='',
        line=dict(width=1, color='#000080'),
        marker=dict(line=dict(width=0.5),size=3, symbol='diamond', color='#000080'),
        stackgroup='one',
        hovertemplate="</br>Data:<b> %{x}</b>" +
                      "</br>"+metrica+":<b> %{y}</b>"+
                      "</br>cliente:<b> "+df['Nome do contato']+"</b>",
    ))

    fig.update_layout(
        showlegend=False, xaxis_type='category',
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        legend=dict(font_size=10, orientation="h", yanchor="top", y=1.10, xanchor="center", x=0.50),
        height=250, hovermode='closest', autosize=False, dragmode=False, margin=dict(l=20, r=20, b=20, t=20)
    )

    fig.add_trace(go.Scatter(x=[df['Data'].min(), df['Data'].max()],
                             y=[media_total, media_total],
                             mode='lines',
                             name='',
                             showlegend=False,
                             hovertemplate="</br>Mediana de Todo o Período:<b>"+
                                           "</br>"+str(media_total)+"</b>",
                             line=dict(color='#800000', dash='dash')))

    fig.update_yaxes(
        title_text=metrica, title_font=dict(family='Sans-serif', size=12), zeroline=False,
        tickfont=dict(family='Sans-serif', size=12), nticks=5, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )
    fig.update_xaxes(
        tickfont=dict(family='Sans-serif', size=12),
        showgrid=False, gridwidth=0.5, gridcolor='#D3D3D3'
    )

    return fig


def classificar_cliente(row):
    if row['Mediana Diff dias [todos pedidos]'] <= 7:
        return 'Semanal'
    elif row['Mediana Diff dias [todos pedidos]'] <= 15:
        return 'Quinzenal'
    elif row['Mediana Diff dias [todos pedidos]'] <= 30:
        return 'Mensal'
    elif row['Mediana Diff dias [todos pedidos]'] <= 90:
        return 'Trimestral'
    elif row['Mediana Diff dias [todos pedidos]'] <= 180:
        return 'Semestral'
    elif row['Mediana Diff dias [todos pedidos]'] <= 365:
        return 'Anual'
    elif row['Mediana Diff dias [todos pedidos]'] <= 720:
        return 'Bianual'
    else:
        return 'Visitante'

def difrenca_dias_transform_df(df, df2):

    df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d')
    df2['Data'] = pd.to_datetime(df2['Data'], format='%Y-%m-%d')

    df1 = df.groupby(['Data','Nome do contato']).agg({'Quantidade': 'sum', 'Número do pedido':'count'}).reset_index()
    df1 = df1.sort_values(by=['Nome do contato', 'Data'])

    df1 = df1.rename(columns={"Número do pedido": "Pedidos"})

    df1['Diff dias [pedido anterior]'] = df1.groupby('Nome do contato')['Data'].diff().dt.days

    hoje = df2['Data'].max()
    df1['Diff dias [data mais recente]'] = df1.groupby('Nome do contato')['Data'].transform(
        lambda x: (hoje - x.max()).days
    )


    df0 = df1.groupby(['Nome do contato'])['Diff dias [pedido anterior]'].median().reset_index().round(1)
    df0 = df0.rename(columns={"Diff dias [pedido anterior]": "Mediana Diff dias [todos pedidos]"})

    df1 = pd.merge(df1, df0, on='Nome do contato', how='outer')

    df1['Rotina'] = df1.apply(classificar_cliente, axis=1)

    df1['Data'] = pd.to_datetime(df1['Data'], format='%Y-%m-%d').dt.date

    return df1





def classificar_cliente_fidelidade(row):
    if (row['Diff dias [pedido anterior]'] == 0):
        return 'Novo'
    elif (row['Diff dias [pedido anterior]'] > row['Diff dias [data mais recente]']):
        return 'Recente'
    elif (row['Diff dias [pedido anterior]'] <= row['Diff dias [data mais recente]']) and \
            row['Diff dias [pedido anterior]'] * 2 >= row['Diff dias [data mais recente]']:
        return 'Aberto'
    elif row['Diff dias [pedido anterior]'] * 5 <= row['Diff dias [data mais recente]']:
        return 'Ausente'
    elif row['Diff dias [pedido anterior]'] * 2 < row['Diff dias [data mais recente]']:
        return 'Atrasado'
    else:
        return 'Novo'


def diff_dias_transform_df_fidelidade(df1):

    df1['Consumo'] = df1.apply(classificar_cliente_fidelidade, axis=1)


    return df1





def plot_point_diff(df, varx, vary, colorscales, vermedia, vart, varz):

    fig = go.Figure()

    if varz == 'Nome do contato' or varz == 'Código (SKU)':
        fig.add_trace(go.Scatter(x=df[varx], y=df[vary], customdata=df[varz],
                                 mode='markers', name='',
                                 hovertemplate="</br>" + varx + ":<b> %{x:,.0f}</b>" +
                                               "</br>" + vary + ":<b> %{y:,.0f}</b>" +
                                               "</br>" + varz + ":<b> %{customdata}</b>",
                                 showlegend=False,
                                 marker=dict(
                                     size=5,
                                     color=((df[vary])),
                                     colorscale=colorscales)
                                 ))

    elif varz == 'Rotina':
        color_mapping = {
            'Visitante': '#19715b',
            'Semanal':'#191970',
            'Mensal': '#5b1971',
            'Quinzenal': '#71195b',
            'Trimestral': '#714519',
            'Semestral': '#712f19',
            'Anual': '#711919',
            'Bianual': '#711919',
            # Adicione mais mapeamentos conforme necessário
        }

        symbol_mapping = {
            'Visitante': 'circle',
            'Semanal':'hexagram',
            'Mensal': 'diamond',
            'Quinzenal': 'square',
            'Trimestral': 'cross',
            'Semestral':'star-square',
            'Anual': 'x',
            'Bianual': 'star',
        }

        for categoria in df[varz].unique():
            categoria_df = df[df[varz] == categoria]

            # Obtenha a cor correspondente a partir do mapeamento
            cor = color_mapping.get(categoria,'#000000')

            # Obtenha o símbolo correspondente a partir do mapeamento
            simbolo = symbol_mapping.get(categoria, 'circle')
            fig.add_trace(go.Scatter(
                x=categoria_df[varx],
                y=categoria_df[vary],
                customdata=categoria_df[varz],
                text=categoria_df[vart],
                mode='markers',
                name=categoria,  # Adicione a categoria como o nome para a legenda
                hovertemplate="</br>" + varx + ":<b> %{x:,.0f}</b>" +
                              "</br>" + vary + ":<b> %{y:,.0f}</b>" +
                              "</br>Consumo:<b> %{customdata}</b>" +
                              "</br>Cliente:<b> %{text}</b>",
                showlegend=True,
                marker=dict(
                    size=5,
                    symbol=simbolo,
                    color=cor)))

    elif varz == 'Consumo':
        color_mapping = {
            'Novo': '#191970',
            'Recente': '#5b1971',
            'Aberto': '#71195b',
            'Atrasado': '#714519',
            'Ausente': '#711919',
        }

        symbol_mapping = {
            'Novo': 'circle',
            'Recente': 'hexagram',
            'Aberto': 'diamond',
            'Atrasado': 'cross',
            'Ausente': 'x',
        }

        for categoria in df[varz].unique():
            categoria_df = df[df[varz] == categoria]

            # Obtenha a cor correspondente a partir do mapeamento
            cor = color_mapping.get(categoria,'#000000')

            # Obtenha o símbolo correspondente a partir do mapeamento
            simbolo = symbol_mapping.get(categoria, 'circle')

            fig.add_trace(go.Scatter(
                x=categoria_df[varx],
                y=categoria_df[vary],
                customdata=categoria_df[varz],
                text=categoria_df[vart],
                mode='markers',
                name=categoria,  # Adicione a categoria como o nome para a legenda
                hovertemplate="</br>" + varx + ":<b> %{x:,.0f}</b>" +
                              "</br>" + vary + ":<b> %{y:,.0f}</b>" +
                              "</br>Consumo:<b> %{customdata}</b>" +
                              "</br>Cliente:<b> %{text}</b>",
                showlegend=True,
                    marker=dict(
                    size=5,
                    symbol=simbolo,
                    color=cor)))



    if vermedia == 'Sim':
        media_metrica = df[vary].median()
        fig.add_trace(
            go.Scatter(
                x=[df[varx].min(), df[varx].max()],
                y=[media_metrica, media_metrica],
                mode='lines',
                name='Mediana Geral',
                showlegend=False,
                line=dict(color='black', dash='dash')
            )
        )

    fig.update_layout(
        paper_bgcolor="#F8F8FF",
        plot_bgcolor="#F8F8FF",
        font={'color': "#000000", 'family': "sans-serif"},
        height=250,
        hovermode="closest",
        autosize=False,
        margin=dict(l=20, r=20, b=20, t=20)
    )
    fig.update_xaxes(
        title_text=varx,
        title_font=dict(family='Sans-serif', size=12),
        tickfont=dict(family='Sans-serif', size=12),
        dtick=30,
        nticks=10,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#D3D3D3'
    )
    fig.update_yaxes(
        title_text=vary,
        title_font=dict(family='Sans-serif', size=10),
        tickfont=dict(family='Sans-serif', size=12),
        dtick=30,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#D3D3D3'
    )

    return fig



def difrenca_dias_transform_produto(df, df2):

    df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d')
    df2['Data'] = pd.to_datetime(df2['Data'], format='%Y-%m-%d')

    df1 = df.groupby(['Data','Código (SKU)']).agg({'ID': 'count',
                                                   'Quantidade':'sum'}).reset_index()
    df1 = df1.sort_values(by=['Código (SKU)', 'Data'], ascending=[False, True]).rename(columns={"ID": "Pedidos"})

    df1['Diff dias [pedido anterior]'] = df1.groupby('Código (SKU)')['Data'].diff().dt.days

    hoje = df2['Data'].max()
    df1['Diff dias [data mais recente]'] = df1.groupby('Código (SKU)')['Data'].transform(
        lambda x: (hoje - x.max()).days)


    df0 = df1.groupby(['Código (SKU)'])['Diff dias [pedido anterior]'].median().reset_index().round(1)
    df0 = df0.rename(columns={"Diff dias [pedido anterior]": "Mediana Diff dias [todos pedidos]"})

    df1 = pd.merge(df1, df0, on='Código (SKU)', how='outer')

    df1['Rotina'] = df1.apply(classificar_cliente, axis=1)

    df1 = df1.sort_values(by=['Código (SKU)', 'Data', 'Quantidade'], ascending=[False, False, False])

    df1['Data'] = pd.to_datetime(df1['Data'], format='%Y-%m-%d').dt.date


    return df1




def linha_produto_data(df, df2, selected_metrica, top):

    df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d').dt.date

    data_range = pd.date_range(start=df2['Data'].min(), end=df2['Data'].max(), freq='D')
    df_datas = pd.DataFrame({'Data': data_range})
    df_datas['Data'] = pd.to_datetime(df_datas['Data'], format='%Y-%m-%d').dt.date

    fig = go.Figure()

    for produto, grupo1 in df.groupby(top):

        grupo1 = pd.merge(df_datas, grupo1, on='Data', how='left').fillna({top: produto, selected_metrica: 0})
        grupo = grupo1.groupby(['Data', top])[selected_metrica].sum().reset_index().sort_values('Data', ascending=True)

        fig.add_trace(go.Scatter(
            x=grupo['Data'], y=grupo[selected_metrica],
            mode='lines+markers', name=f'{produto}',
            marker=dict(symbol='diamond'),
            hovertemplate="</br>Data:<b> %{x}</b>" +
                          "</br>"+selected_metrica+":<b> %{y:,.0f}</b>"+
                          "</br>Produto:<b> "+produto+"</b>",
        ))

    # Atualize o layout do gráfico
    fig.update_layout(
        showlegend=True,
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        height=300, hovermode="closest", autosize=False, dragmode=False, margin=dict(l=20, r=20, b=20, t=20)
    )
    fig.update_yaxes(
        tickfont=dict(family='Sans-serif', size=12), nticks=5, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )
    fig.update_xaxes(
        tickfont=dict(family='Sans-serif', size=12),
        showgrid=False, gridwidth=0.5, gridcolor='#D3D3D3'
    )

    return fig

def linha_produto_data_mes(df, df2, selected_metrica, tempo, selected_linha, top):

    if selected_linha == 'Independete':
        stackgroup = ''
    elif selected_linha == 'Agrupado':
        stackgroup = 'one'

    if selected_metrica == 'Pedidos':
        df2 = (df2.groupby([tempo])['Quantidade'].count()
                     .reset_index().sort_values('Quantidade', ascending=False))

    elif selected_metrica == 'Quantidade':
        df2 = (df2.groupby([tempo])['Quantidade'].sum()
                     .reset_index().sort_values(tempo, ascending=False))

    df2 = df2[[tempo]]

    fig = go.Figure()

    if selected_linha == 'Normalizado':
        for produto, grupo1 in df.groupby(top):
            grupo1 = pd.merge(df2, grupo1, on=tempo, how='outer').fillna({top: produto, selected_metrica: 0})
            grupo = grupo1.groupby([tempo, top])[selected_metrica].sum().reset_index().sort_values(tempo, ascending=True)
            fig.add_trace(go.Scatter(
                x=grupo[tempo], y=grupo[selected_metrica],
                mode='lines+markers', name=f'{produto}',
                marker=dict(symbol='diamond'), stackgroup='one', groupnorm='percent',
                hovertemplate="</br>Data:<b> %{x}</b>" +
                              "</br>" + selected_metrica + ":<b> %{y:.2f}%</b>" +
                              "</br>"+ top + ":<b> " + produto + "</b>",
            ))


        fig.update_layout(
            showlegend=True,
            xaxis_type='category',
            yaxis=dict(
                type='linear',
                range=[1, 100],
                ticksuffix='%'))

    else:
        for produto, grupo1 in df.groupby(top):

            grupo1 = pd.merge(df2, grupo1, on=tempo, how='outer').fillna({top: produto, selected_metrica: 0})
            grupo = grupo1.groupby([tempo, top])[selected_metrica].sum().reset_index().sort_values(tempo, ascending=True)
            fig.add_trace(go.Scatter(
                x=grupo[tempo], y=grupo[selected_metrica],
                mode='lines+markers', name=f'{produto}',
                marker=dict(symbol='diamond'), stackgroup=stackgroup,
                hovertemplate="</br>Data:<b> %{x}</b>" +
                              "</br>" + selected_metrica + ":<b> %{y:,.0f}</b>" +
                              "</br>"+ top +":<b> " + produto + "</b>",
            ))

    # Atualize o layout do gráfico
    fig.update_layout(
        showlegend=True, xaxis_type='category',
        paper_bgcolor="#F8F8FF", plot_bgcolor="#F8F8FF", font={'color': "#000000", 'family': "sans-serif"},
        height=300, hovermode="closest", autosize=False, dragmode=False, margin=dict(l=20, r=20, b=20, t=20)
    )
    fig.update_yaxes(
        tickfont=dict(family='Sans-serif', size=12), nticks=5, showgrid=True, gridwidth=0.5, gridcolor='#D3D3D3'
    )
    fig.update_xaxes(
        tickfont=dict(family='Sans-serif', size=12),
        showgrid=False, gridwidth=0.5, gridcolor='#D3D3D3'
    )
    return fig


def merge_data_produto(df, df2, tempo):

    merge1 = df.groupby([tempo, 'Código (SKU)']).count().reset_index()

    merge2 = df2.groupby([tempo, 'Código (SKU)']).count().reset_index()
    merge2['Quantidade'] = 0

    merged_df = pd.merge(merge1, merge2, on=[tempo, 'Código (SKU)'], how='outer')
    merged_df = merged_df[[tempo, 'Código (SKU)', 'Quantidade_x']]
    merged_df = merged_df.fillna(0)
    merged_df = merged_df.rename(columns={"Quantidade_x": "Quantidade"})
    merged_df = merged_df.sort_values(tempo, ascending=True)

    return merged_df


def generate_summary(solicitado, temperature, api_key, rotina, frequencia, caract):

    prompt = f"""
    Você, como representante comercial, está redigindo uma mensagem pós-atendimento para os clientes da loja Setor da Embalagem. 
    Ao personalizar a abordagem para cada cliente, considere duas características importantes: 
    a frequência de pedidos, classificando o cliente como ("{rotina}"), e o histórico do último pedido, 
    classificando-o como ("{frequencia}"). Utilize essas informações apenas como guia para a mensagem.

    Ao entrar em contato com o cliente, cujo nome é: "Cliente", via WhatsApp, o seu objetivo é manter um 
    relacionamento contínuo e positivo de compras. Evite mencionar diretamente essas características na mensagem, 
    mas use-as para moldar seu conteúdo e atendimneto ao cliente.

    Crie uma mensagem persuasiva e impactante em um único parágrafo, e com exatamente {caract} letras ao todo.
    Aborde o cliente com cordialidade, não pode ser uma mensagem robotica, sem oferecer promoções. 

    Analise a solicitação fornecida ("{solicitado}") e integre o conteúdo de forma a criar uma 
    mensagem envolvente que cativará o cliente."""


    openai.api_key = api_key

    max_context_length = 2049  # tamanho máximo do contexto do GPT-3
    max_completion_length = 2024  # tamanho máximo para a conclusão

    if len(prompt) > max_context_length:
        prompt = prompt[:max_context_length]

    if max_completion_length > 2024:
        max_completion_length = 2024

    completion = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=max_completion_length,
        n=1,
        stop=None,
        temperature=temperature,
    )

    if len(completion.choices[0].text) > max_completion_length:
        completion.choices[0].text = completion.choices[0].text[:max_completion_length]

    return completion.choices[0].text


def classificar_cliente(respostas):
    if respostas == 0:
        return 'Ignorado'
    elif respostas <= 4 and respostas != 0:
        return 'Interação'
    elif respostas >= 5:
        return 'Comércio'
    else:
        return 'Outro'


def metrica_respostas(df):

    count = (df.groupby('Nome do contato').count()
             .sort_values('Numero de Respostas', ascending=False).reset_index())

    total = int(len(count) * 10)

    meta = int(len(count) * 3)

    valor = df['Numero de Respostas'].sum()

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=valor,
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': meta, 'increasing': {'color': "Purple"}},
        gauge={
            'axis': {'range': [0, total], 'tickwidth': 2, 'tickcolor': "#4169E1"},
            'bordercolor': "#4169E1",
            'bar': {'color': "#4169E1"},
            'bgcolor': "lightgray",
            'borderwidth': 2,
            'steps': [
                {'range': [0, meta], 'color': "#ADD8E6"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.7,
                'value': meta}}))
    fig.update_layout(
        paper_bgcolor="#F8F8FF", font={'size': 20},
        height=300, autosize=True, margin=dict(l=20, r=20, b=20, t=30))

    return fig


def unir_palavras(lista):
    return ' '.join(lista)

def wordcloud(df):

    df['words'] = df['Respostas Whatsapp'].apply(unir_palavras)
    words = ' '.join(df['words'])

    stop_words = STOPWORDS.update(["da", "do", "a", "e", "o", "em", "para", "um",
                                   "que", "por", "como", "uma", "de", "onde", "são",
                                   "sim", "não", "mas", "mais", "então", "das", "dos", "nas", "nos",
                                   "bio", "link", "isso", "tem", "até"])

    fig, ax = plt.subplots()
    wordcloud = WordCloud(
        height=240,
        min_font_size=8,
        scale=2.5,
        background_color='#F8F8FF',
        max_words=100,
        stopwords=stop_words,
        min_word_length=3).generate(words)

    plt.imshow(wordcloud)
    plt.axis('off')  # to off the axis of x and

    return fig