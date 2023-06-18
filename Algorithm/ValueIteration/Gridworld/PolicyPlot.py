import plotly.offline as py
import plotly.graph_objects as go


def draw_policy(policy, title_name, x_range=5, y_range=5):
    fig = go.Figure()
    fig.update_xaxes(range=[0, x_range])  # Set axes ranges
    fig.update_yaxes(range=[y_range, 0])
    # Add shapes
    s = 0
    x_take = []
    y_take = []
    for i in range(x_range):
        x_take.append(i + 0.5)
    for j in range(y_range):
        y_take.append(j + 0.5)
    for y in y_take:
        for x in x_take:
            if policy[s][0] == 1:
                fig.add_shape(type='line', x0=x, y0=y, x1=x, y1=y - 0.4, line=dict(color="RoyalBlue", width=3))
                fig.add_shape(type='line', x0=x, y0=y - 0.4, x1=x - 0.1, y1=y - 0.3,
                              line=dict(color="RoyalBlue", width=3))
                fig.add_shape(type='line', x0=x, y0=y - 0.4, x1=x + 0.1, y1=y - 0.3,
                              line=dict(color="RoyalBlue", width=3))
            if policy[s][1] == 1:
                fig.add_shape(type='line', x0=x, y0=y, x1=x + 0.4, y1=y, line=dict(color="RoyalBlue", width=3))
                fig.add_shape(type='line', x0=x + 0.4, y0=y, x1=x + 0.3, y1=y + 0.1,
                              line=dict(color="RoyalBlue", width=3))
                fig.add_shape(type='line', x0=x + 0.4, y0=y, x1=x + 0.3, y1=y - 0.1,
                              line=dict(color="RoyalBlue", width=3))
            if policy[s][2] == 1:
                fig.add_shape(type='line', x0=x, y0=y, x1=x, y1=y + 0.4, line=dict(color="RoyalBlue", width=3))
                fig.add_shape(type='line', x0=x, y0=y + 0.4, x1=x - 0.1, y1=y + 0.3,
                              line=dict(color="RoyalBlue", width=3))
                fig.add_shape(type='line', x0=x, y0=y + 0.4, x1=x + 0.1, y1=y + 0.3,
                              line=dict(color="RoyalBlue", width=3))
            if policy[s][3] == 1:
                fig.add_shape(type='line', x0=x, y0=y, x1=x - 0.4, y1=y, line=dict(color="RoyalBlue", width=3))
                fig.add_shape(type='line', x0=x - 0.4, y0=y, x1=x - 0.3, y1=y + 0.1,
                              line=dict(color="RoyalBlue", width=3))
                fig.add_shape(type='line', x0=x - 0.4, y0=y, x1=x - 0.3, y1=y - 0.1,
                              line=dict(color="RoyalBlue", width=3))
            s += 1
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)  # 添加上下边界
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)  # 添加左右边界
    fig.update_xaxes(showgrid=True, gridwidth=1.5, gridcolor='LightPink')
    fig.update_yaxes(showgrid=True, gridwidth=1.5, gridcolor='LightPink')
    fig.update_layout(height=1000, width=1000, title_text=title_name)  # 画布大小
    py.plot(fig, filename=title_name+'.html')