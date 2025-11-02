import io
import sys
import cma
import copy
import torch
import warnings
import numpy as np
import torch.multiprocessing as mp

from cma.optimization_tools import EvalParallel2

from tqdm import tqdm
from src.nca.utils import *
from src.nca.nca import NCA
from matplotlib import pyplot as plt
from IPython.display import clear_output

import plotille

from typing import List, Callable
# from cell.evolution.ga import mixAndMutate

warnings.simplefilter(action="ignore", category=FutureWarning)
torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()  # Frees unused cached memory

class NCADish:

    SEED = 0
    TARGET = 1

    def __init__(
        self,
        width : int,
        height : int,
        channels : int = 8,
        hidden_channels : int = 32,
        life_masking_channels : int = 3,
        life_masking : bool = True,
        model_path : str = None,
        device : str = "cpu"
    ):
        self.width = width
        self.height = height
        self.channels = channels
        self.seeds = None
        self.targets = None
        self.device = device
        if model_path:
            self.model = self.load(model_path)
        else:
            self.model = NCA(
                n_channels=channels,
                hidden_channels=hidden_channels,
                life_masking=life_masking,
                lmc=life_masking_channels,
                device=self.device,
            )
        self.device = self.model.device

    def seed(self, seeds : np.array):
        self.seeds = np.zeros((1, self.channels,self.width,self.height))
        self.seeds[
            :,
            :seeds.shape[1],
            :seeds.shape[2],
            :seeds.shape[3]] = seeds
        self.seeds = torch.from_numpy(self.seeds).float().to(self.device)
        return self.seeds

    def target(self, targets : np.array):
        self.targets = np.zeros((1, targets.shape[1], self.width, self.height))
        targets = np.array(targets)
        self.targets[
            :targets.shape[0],
            :targets.shape[1],
            :targets.shape[2],
            :targets.shape[3]] = targets
        self.targets = torch.from_numpy(self.targets).float().to(self.device)
        return self.targets

    def grow(self, 
        env : Callable, 
        epochs : int, 
        learnign_rate : float  = 0.002,
        pool_size : int = 1024,
        batch_size : int = 8,
        damage : bool = False,
        loss_callback : Callable = None):

        assert not damage, "EMOTIONAL DAMAGE. Damage is not functional yet."
        self.pool_size = pool_size

        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learnign_rate
        )

        basic_seeds_pool = self._prepare_pool(self.seeds)
        pool_seeds = self._prepare_pool(self.seeds)

        targets_pool = self._prepare_pool(self.targets)

        losses = []
        for epoch in (pbar := tqdm(range(epochs))):

            batch_idxs = self._random_sample_pool(
                basic_seeds_pool, batch_size
            )

            cs = pool_seeds[batch_idxs]
            target = targets_pool[batch_idxs]

            loss_batch, total_loss, cs = env(
                self.model, cs, target,
            )

            # Backpropagate intermediate loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # index of cell state with highest loss in batch
            argmax_batch = loss_batch.argmax().item()
            # index of cell state with highest loss in pool
            argmax_pool = batch_idxs[argmax_batch]
            # indices of cell states in batch that are not the cell state with highest loss
            remaining_batch = [i for i in range(batch_size) if i != argmax_batch]
            # indices of cell states in pool that are not the cell state with highest loss
            remaining_pool = [i for i in batch_idxs if i != argmax_pool]
            # replace cell state with highest loss in pool with seed image
            pool_seeds[argmax_pool] = basic_seeds_pool[argmax_pool].clone()
            # update cell states of selected batch with cell states from model output
            pool_seeds[remaining_pool] = cs[remaining_batch].clone().detach()

            pbar.set_description(
                f"Epoch [{epoch+1}], Loss: {total_loss.item():.4f}"
            )

            losses.append(total_loss.cpu().detach())
            if loss_callback:
                loss_callback(losses)

        avg_loss = np.sum(losses)/len(losses)

        return self.model, avg_loss

    def save(self, path = "checkpoint.nca"):
        torch.save(self.model.state_dict(), path)

    def load(self, path = "checkpoint.nca"):
        self.model.load_state_dict(torch.load("checkpoint.nca"))
        return self.model

    def show_best(self):
        pass
        # env(
        #     self.model,
        #     cs[argmax_batch : argmax_batch + 1, ...],
        #     target[argmax_batch : argmax_batch + 1, ...],
        #     epoch,
        #     True,
        #     self.device,
        # )

    def _prepare_pool(self, source):
        if source is not None:
            pool = (
                source.repeat(self.pool_size, 1, 1, 1).to(self.device).clone()
            )
        else:
            pool = np.array([None] * self.pool_size)
        return pool

    def _random_sample_pool(self, pool, batch_size):
        batch_idxs = np.random.choice(
            pool.shape[0], batch_size, replace=False
        ).tolist()
        return batch_idxs



######################################################### EVOLUTIONARY LEARNING METHODS #########################################################

    # def cmaes_eval(self, solution, cs, shapes, target, config, epoch, env):

    #     device = self.device
    #     if torch.cuda.is_available() and device.type == "cuda":
    #         torch.cuda.init()  # Explicit CUDA initialization
    #         device = torch.device("cuda:0")

    #     if self.debug:
    #         start = time.time()

    #     model = copy.deepcopy(self.model)

    #     model.set(unflatten(solution, shapes))

    #     _, total_loss, _ = env(
    #         model, cs[...], target[...], config, epoch, False, device
    #     )

    #     if self.debug:
    #         print(f"timeit: {time.time() - start}")

    #     return total_loss.item()

    # @no_grad
    # def grow_cmaes(self, env, epochs, seeds, targets, config):
    #     mp.set_start_method("spawn", force=True)
    #     # Create the CMA-ES optimizer
    #     initial_params, shapes = flatten(self.model.get())  # 10D search space
    #     sigma = self.training_step
    #     popsize = 32
    #     es = cma.CMAEvolutionStrategy(initial_params, sigma, {"popsize": popsize})

    #     basic_seeds_pool = seeds.repeat(self.pool_size, 1, 1, 1).to(self.device).clone()
    #     pool_seeds = seeds.repeat(self.pool_size, 1, 1, 1).to(self.device).clone()
    #     if targets is not None:
    #         pool_targets = (
    #             targets.repeat(self.pool_size, 1, 1, 1).to(self.device).clone()
    #         )
    #     else:
    #         pool_targets = np.array([None] * self.pool_size)
    #     # pool = [pair for pair in paired_seeds_targets for _ in range(self.pool_size)]

    #     losses = []
    #     epoch = 0

    #     # INFO: using -1 for number of processes speeds up calculations so much!
    #     with EvalParallel2(self.cmaes_eval, number_of_processes=-1) as eval_all:
    #         while epoch < epochs:
    #             solutions = es.ask()  # Generate candidate solutions
    #             args = (
    #                 pool_seeds[0:1, ...],
    #                 shapes,
    #                 pool_targets[0:1, ...],
    #                 config,
    #                 epoch,
    #                 env,
    #             )

    #             fitvals = eval_all(solutions, args=args)
    #             es.tell(solutions, fitvals)
    #             es.disp()

    #             minfit = np.min(fitvals)
    #             best_solution = solutions[np.argmin(fitvals)]

    #             if (
    #                 minfit < self.best_recorded_loss
    #                 and self.checkpoint_path is not None
    #             ):
    #                 self.model_checkpoint = self.model
    #                 torch.save(
    #                     self.model_checkpoint.state_dict(),
    #                     f"{self.checkpoint_path}.nca",
    #                 )

    #             self.model.set(unflatten(best_solution, shapes))

    #             losses.append(minfit)
    #             epoch += 1

    #             # execute best
    #             _, _, _ = env(
    #                 self.model,
    #                 pool_seeds[0:1, ...],
    #                 pool_targets[0:1, ...],
    #                 config,
    #                 epoch,
    #                 True,
    #                 self.device,
    #             )

    #             if config["plot"]:
    #                 # Show the figure in terminal
    #                 fig = plotille.Figure()
    #                 plotille.plot(
    #                     range(len(losses)),
    #                     losses,
    #                     height=30,
    #                     width=60,
    #                     interp="linear",
    #                     lc="cyan",
    #                 )

    #     # Plot the data
    #     clear_output(wait=False)  # Clear previous plot
    #     plt.figure(111)
    #     plt.clf()  # Clear the current figure
    #     plt.plot(losses)  # Add markers to make points visible
    #     plt.title(f"Training Loss popsize: {popsize} sigma:{sigma} epochs: {epochs}")
    #     plt.xlabel("Index")
    #     plt.ylabel("Value")
    #     plt.grid(True)

    #     # Save the plot as a PNG image
    #     plt.savefig("losses.png", dpi=300, bbox_inches="tight")
    #     print("Learning finished")

    #     return self.model
