from dataclasses import dataclass
import threading
import time

from ncpu.dataset import NCPUDataset, PoolDataset
from ncpu.model import NeuralCA
from ncpu.trainer import NCPUTrainer

from matplotlib import pyplot as plt
from IPython.display import clear_output, display
from tqdm.auto import tqdm
from datetime import datetime
from ncpu.utils import print_tensor, sequence_batch_to_html_gifs
import mediapy as media
import torch
import panel as pn

pn.extension()


def setup_trainer(config):
    dataset = NCPUDataset(
        W=config.W,
        H=config.H,
        r=config.r,
        small_r=config.small_r,
        spacing=config.spacing,
        margin=config.margin,
        sampler=config.sampler,
        balanced=config.balanced,
    )
    if config.pool_size > 0:
        dataset = PoolDataset(dataset, pool_size=config.pool_size)

    nca = NeuralCA(
        channels=config.channels,
        hidden_channels=config.hidden_channels,
        fire_rate=config.fire_rate,
        alive_masking=config.alive_masking,
        zero_initialization=config.zero_initialization,
    ).to(config.device)

    trainer = NCPUTrainer(
        nca,
        dataset.get_dataloader(batch_size=config.batch_size),
        lr=config.lr,
        apply_gaussian_noise=config.apply_gaussian_noise,
    )
    trainer.sanity_check()

    return trainer


class TrainRunnerUI:
    def __init__(self, config):
        self.config = config
        self.trainer = setup_trainer(config)

    def render(self):
        is_training = False
        start_pause_training_btn = pn.widgets.Button(
            name="Start training", button_type="primary"
        )
        training_progress_bar = pn.widgets.Progress(
            name="Training progress", value=0, width=200, align="end"
        )

        def toggle_training(event=None):
            start_pause_training_btn.name = "Pause training"

        def training():
            while is_training:
                info = self.trainer.optim_step(steps=40)
                loss = info["loss"].item()
                training_progress_bar.value = training_progress_bar.value + 1
                pn.io.push_notebook(hard=True)
                time.sleep(0.01)

        # training_thread = threading.Thread(target=training)
        # training_thread.daemon = True
        # training_thread.start()

        start_pause_training_btn.on_click(toggle_training)

        return pn.Column(
            pn.Row(start_pause_training_btn, training_progress_bar),
        )

    def generate_pred_plots(self):
        pbar = tqdm(range(self.config.its))
        for i in pbar:
            info = self.trainer.optim_step(steps=self.config.nca_steps)
            loss = info["loss"].item()
            pbar.set_description(f"loss={loss:.6f}")

            if i % 250 == 0:
                clear_output(wait=False)
                display(pbar.container)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(range(len(self.trainer.history)), self.trainer.history, s=1)
                ax.set_yscale("log")
                plt.close(fig)
                display(fig)

                print_tensor("inp    ", info["inp"])
                print_tensor("out    ", info["out"])
                print_tensor("nca_out", info["nca_out"])

                to_show = 8
                inp = info["inp"][:to_show]
                out = info["out"][:to_show]
                nca_out = info["nca_out"][:to_show]
                rollout = info["rollout"][:to_show]
                io = torch.cat([inp, out, nca_out], dim=0)

                media.show_images(
                    io.detach().cpu(), columns=to_show, width=100, height=100
                )
                sequence_batch_to_html_gifs(rollout, columns=to_show, fps=10)

                it = len(self.trainer.history)
                path = f"notebooks/checkpoints/ncpu_{it}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                torch.save(self.trainer.nca.state_dict(), path)
