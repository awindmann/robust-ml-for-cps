import os
import numpy as np
import torch

import plotly.graph_objects as go
import plotly.subplots as subplots
import plotly.io as pio



pio.renderers.default = "browser"


def plot_sample_forecast(sample, fcast, title=None, display=True):
    """Plots forecast of tank levels and settings for one sample.
    Args:
        sample: torch.Tensor, shape (seq_len, 3), sample from dataloader
        fcast: torch.Tensor, shape (pred_len, 3), forecast of sample
        title: str, title of plot
        display: bool, if True, plot is displayed, else returned
    """
    x1, x2 = sample
    pred_x2 = np.squeeze(fcast)
    x = np.concatenate((x1, x2))
    colors = [
        '#1f77b4',  # muted blue
        '#d62728',  # brick red
        '#2ca02c',  # cooked asparagus green
        '#17becf',  # blue-teal
        '#ff7f0e',  # safety orange
        '#bcbd22',  # curry yellow-green
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
    ]

    fig = go.Figure()
    for sig, name, c in zip([x[:, 0], x[:, 1], x[:, 2]],
                            ['h1', 'h2', 'h3'],
                            colors[:3]):
        fig.add_trace(go.Scatter(x=np.array(range(x.shape[0])), y=sig, name=name,
                      mode="lines", opacity=1, line=dict(color=c)))
    for sig, name, c in zip([pred_x2[:, 0], pred_x2[:, 1], pred_x2[:, 2]],
                            ['pred_h1', 'pred_h2', 'pred_h3'],
                            colors[3:7]):
        fig.add_trace(go.Scatter(x=np.array(range(x1.shape[0], x1.shape[0] + x2.shape[0])), y=sig, name=name,
                      mode="lines", opacity=1, line=dict(color=c, dash="dot")))

    fig.add_vline(x=len(x1), line_dash="dash")
    fig.update_xaxes(tick0=0, dtick=200)
    fig.update_xaxes(title_text=r'time')
    fig.update_layout(width=800, height=500,
                      font_family="Serif", font_size=14,
                      margin_l=5, margin_t=50, margin_b=5, margin_r=5)
    if title is not None:
        fig.update_layout(title=title)
    if display:
        fig.show()
    else:
        return fig


def fcast_overview(datamodule, model, idx=0, title=None, save_path=None):
    """Plots forecast of tank levels and settings.
    All scenarios are plotted in two figures. Combines plot_sample_forecast() and fcast_overview_separate().
    Args:
        datamodule: DataModule
        model: Model
        idx: int, index of sample to plot
        title: str, title of plot
        save_path: str, path to save plot
    """
    model = model.to(model.visualization_device)
    model.eval()
    datasets = datamodule.ds_dict

    # plot water levels
    n_rows = 3
    n_cols = 3
    fig = subplots.make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True, vertical_spacing=0.02)
    for i, (scenario, ds) in enumerate(datasets.items()):
        if i >= n_rows * n_cols:
            break
        sample = ds[idx]
        x = torch.tensor(sample[0]).unsqueeze(0)
        fcast = model(x).cpu().detach().numpy()
        fcast_plot = plot_sample_forecast(sample, fcast, title=scenario, display=False)
        for j in range(6):
            fig.add_trace(fcast_plot.data[j], row=(i//n_cols)+1, col=(i%n_cols)+1)
        fig.update_xaxes(tick0=0, dtick=50)
        fig.update_yaxes(title_text=scenario + f" [DataLoader {i}]", row=(i//n_cols)+1, col=(i%n_cols)+1)

    fig.add_vline(x=x.size(1), line_dash="dash")
    for col in range(n_cols):
        fig.update_xaxes(title_text=r'time', row=n_rows, col=col+1)
    fig.update_layout(showlegend=False)
    fig.update_layout(title=f"Water Level Predictions by {title}")
    # export figure
    if save_path is not None:
        # if the path does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.write_image(save_path + f"water_levels_{title}.png", width=1200, height=800)
    else:
        fig.show()
