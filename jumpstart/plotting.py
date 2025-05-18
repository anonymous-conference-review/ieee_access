import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from PIL import Image


def add_margin(pil_img, top, right, bottom, left, color=(255, 255, 255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.jr_mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def plot_status_ploty(status, labels=None):
    dead_color = '#db5e56'
    linear_color = '#dbc256'
    nonlinear_color = '#5ea28d'
    colorscale = [[0.0, dead_color], [0.15, dead_color], [0.33, dead_color],
                  [0.33, linear_color], [0.66, linear_color],
                  [0.66, nonlinear_color], [1.0, nonlinear_color]]
    colorbar = {'tickvals': [0.66, 0, -0.66],
                'ticktext': ['Non-Linear', 'Linear', 'Dead']}
    fig = go.Figure(go.Heatmap(z=status, colorscale=colorscale, colorbar=colorbar,
                               x=labels, hoverongaps=False, zmin=-1, zmax=1))
    return fig


def plot_unit_status_plotly(unit_status, labels):
    return plot_status_ploty(unit_status, labels)


def plot_point_status_plotly(point_status, labels):
    return plot_status_ploty(point_status, labels)


def plot_unit_status(ax, network_unit_status, square=True, xticklabels=True, yticklabels=True,
                     cbar_ax=None):
    sns.heatmap(network_unit_status.fillna(0).astype(int), ax=ax,
                vmin=-1, vmax=1,
                cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["#db5e56", "#5ea28d", "#dbc256", ], N=3),
                mask=network_unit_status.isnull(),
                square=square,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                cbar_ax=cbar_ax,
                )

    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + r / 3 * (0.5 + i) for i in range(3)])
    colorbar.set_ticklabels(['Dead', 'Non-linear', 'Linear'])
    plt.tight_layout()


def plot_point_status(ax, network_point_status, square=True, xticklabels=True, yticklabels=True,
                      cbar=True, cbar_ax=None):
    sns.heatmap(network_point_status.fillna(0).astype(int), ax=ax,
                vmin=-1, vmax=1,
                cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["#db5e56", "#5ea28d", "#dbc256", ], N=3),
                mask=network_point_status.isnull(),
                square=square,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                cbar=cbar,
                cbar_ax=cbar_ax
                )
    if cbar:
        colorbar = ax.collections[0].colorbar
        r = colorbar.vmax - colorbar.vmin
        colorbar.set_ticks([colorbar.vmin + r / 3 * (0.5 + i) for i in range(3)])
        colorbar.set_ticklabels(['Dead', 'Non-linear', 'Linear'])
    plt.tight_layout()

    # ax.set_xlim([0, len(shattering_metrics.layer_sets)])
    # ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    # ax.axis('off')

def plot_lr_schedule(lr_scheduler, epochs):
    lrs = []
    for epoch in range(epochs):
        lr_scheduler.step()
        lrs.append(lr_scheduler.get_last_lr()[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(epochs)), y=lrs, mode='lines'))
    fig.update_layout(title='Learning Rate Schedule', xaxis_title='Epoch', yaxis_title='Learning Rate')
    fig.show()
