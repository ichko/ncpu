import panel as pn
import torch
from ncpu.model import NeuralCA

pn.extension()


def nca_dashboard():
    width = pn.widgets.IntInput(name="Width", value=64, start=3, end=256, width=100)
    height = pn.widgets.IntInput(name="Height", value=64, start=3, end=256, width=100)
    num_channels = pn.widgets.IntInput(
        name="State Channels", value=16, start=1, end=128, width=100
    )
    hidden_channels = pn.widgets.IntInput(
        name="Hidden Channels", value=128, start=1, end=1024, width=100
    )
    fire_rate = pn.widgets.FloatSlider(
        name="Fire Rate",
        value=0.5,
        start=0.0,
        end=1.0,
        step=0.01,
        width=100,
        align="center",
    )
    num_steps = pn.widgets.IntInput(
        name="Number of Steps", value=100, start=1, end=1000, width=100
    )
    batch_size = pn.widgets.IntInput(
        name="Batch Size", value=8, start=1, end=30, width=100
    )

    reload_model_button = pn.widgets.Button(
        name="Reload Model", width=100, button_type="default", align="end"
    )
    predict_button = pn.widgets.Button(name="Predict", width=100, button_type="primary")
    gifs_pane = pn.pane.HTML()

    model = None

    def load_model(_):
        nonlocal model
        model = NeuralCA(
            channels=num_channels.value,
            hidden_channels=hidden_channels.value,
            fire_rate=fire_rate.value,
        )

    def predict(_):
        nonlocal model
        predict_controls.loading = True
        inp = torch.randn(
            batch_size.value, num_channels.value, width.value, height.value
        )
        out = model.forward(inp, steps=num_steps.value)
        gifs_pane.object = sequence_batch_to_html_gifs(out)
        predict_controls.loading = False

    reload_model_button.on_click(load_model)
    predict_button.on_click(predict)
    load_model(None)

    predict_controls = pn.Column(
        pn.pane.HTML(f"<b>Predict</b>"),
        num_steps,
        batch_size,
        predict_button,
    )

    return pn.Row(
        pn.Column(
            pn.pane.HTML(f"<b>Load config</b>"),
            width,
            height,
            num_channels,
            hidden_channels,
            fire_rate,
            reload_model_button,
        ),
        predict_controls,
        gifs_pane,
    )
