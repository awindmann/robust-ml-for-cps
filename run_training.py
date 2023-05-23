import glob
import os
import itertools
import shutil
from datetime import datetime
from argparse import ArgumentParser

import torch
import torchinfo
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.profilers import SimpleProfiler
import train_arguments as args
from data.data_module import ThreeTankDataModule
from visualizations.plots import fcast_overview

from models.LSTM import LSTM
from models.GRU import GRU
from models.MLP import MLP
from models.RIM import RIM
from models.Transformer import Transformer
from models.TcnAe import TcnAe
from models.TCN import TCN


torch.backends.cudnn.benchmark = False

def run_training(model, hparams = None, model_name = None, logdir = None, cli_args = None):

    if hparams is None:
        # load hparams. If hparam is a list, pick first value
        hparams = {key: value[0] if isinstance(value, list) else value for key, value in args.HPARAMS.items()}  # TODO hparam dict structure in args changed
    if model_name is None:
        model_name = '_'.join([model]+ [''.join([k[0] for k in key.split('_')]) + str(v) for key, v in hparams.items()])

    # set seed
    pl.seed_everything(42, workers=True)

    # log the results
    if logdir is None:
        logdir = f"logs/{datetime.today().strftime('%Y-%m-%d')}/{model}"

    # load datamodule
    dm = ThreeTankDataModule(
        batch_size=cli_args.BATCH_SIZE, 
        num_workers=cli_args.NUM_WORKERS
    )

    # configure pl trainer
    callbacks = list()
    callbacks.append(EarlyStopping(monitor="ep_val_loss", patience=50))
    if args.SAVE_CHECKPOINT:
        callbacks.append(
            ModelCheckpoint(
                monitor='ep_val_loss', 
                filename='{epoch}-{ep_val_loss:.4f}',
                save_top_k=1, 
                mode='min'
                )
            )
    version = '' if not args.USE_LOGGER_VERSIONING else None
    if args.USE_LOGGER:
        logger = TensorBoardLogger(logdir, name=model_name, version=version, default_hp_metric=False, log_graph=False)
        callbacks.append(LearningRateMonitor())
        profiler = SimpleProfiler(filename="profiler-results")
    else:
        logger = False
        profiler = None

    trainer_args = args.TRAINER_CONFIG
    trainer_args["max_epochs"] = cli_args.MAX_EPOCHS
    trainer_args["accelerator"] = cli_args.ACCELERATOR

    trainer = pl.Trainer(
        **trainer_args,
        log_every_n_steps=cli_args.LOG_EVERY_N_STEPS,
        callbacks=callbacks,
        logger=logger,
        profiler=profiler
        )
    print("Done.")

    # load model
    if model == "LSTM":
        model = LSTM(**hparams)
    elif model == "GRU":
        model = GRU(**hparams)
    elif model == "MLP":
        model = MLP(**hparams)
    elif model == "RIM":
        model = RIM(**hparams)
    elif model == "Transformer":
        model = Transformer(**hparams)
    elif model == "TcnAe":
        model = TcnAe(**hparams)
    elif model == "TCN":
        model = TCN(**hparams)
    else:
        raise ValueError(f"The model {model} does not exist.")

    # train
    print(f"\nTraining {model_name}.")
    if args.USE_LOGGER:
        print(f"Logging in {logdir}/{model_name}")
        # create directory for model
        os.makedirs(f"{logdir}/{model_name}", exist_ok=True)
        # save model architecture
        summary = torchinfo.summary(
            model, 
            verbose=0
        )
        with open(f"{logdir}/{model_name}/architecture.txt", "w") as f:
            f.write(str(summary))

    if args.CONTINUE_TRAIN:
        raise NotImplementedError("Continuing training is not implemented yet.")
        # if not args.USE_LOGGER_VERSIONING:
        #     raise NotImplementedError
        # latest_version = len([version for version in os.listdir(f"{logdir}/{model_name}")]) - 1
        # ckpt_path = glob.glob(f"{logdir}/{model_name}/version_{latest_version}/checkpoints/*.ckpt"),
        # trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt_path[0][0])
    else:
        trainer.fit(model=model, datamodule=dm)
    
    # test
    trainer.test(model=model, datamodule=dm, ckpt_path="best")

    # visualization
    if args.PLOT_FORECAST:
        # load best model 
        if args.SAVE_CHECKPOINT:
            model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        fcast_overview(dm, model, idx=84, title=model_name, save_path=f"{logdir}/{model_name}/plots/")


def check_hparams(model, hparams, logdir = None, cli_args = None):

    # create a list of keys that have direct values
    direct_keys = [k for k, v in hparams.items() if not isinstance(v, list)]
    # create a list of keys that have list values
    list_keys = [k for k, v in hparams.items() if isinstance(v, list)]
    # get the list values for each key that has a list value
    list_values = [hparams[k] for k in list_keys]

    # log the results
    if logdir is None:
        logdir = f"logs/{datetime.today().strftime('%Y-%m-%d')}/{model}"
    
    if list_keys:
        # calculate the number of combinations
        num_combinations = 1
        for v in list_values:
            num_combinations *= len(v)
        print(f"Training the model with {num_combinations} different hyperparameter combinations.")

        # iterate over every combination of list values
        curr_combination = 0
        failed_runs = list()
        failed_runs_reason = list()
        for combo in itertools.product(*list_values):
            curr_combination += 1
            print("-"*80 + f"\nTraining hyperparameter combination {curr_combination} of {num_combinations}.")
            # create a new dictionary with the combination of values
            combo_dict = dict(zip(list_keys, combo))
            # merge the new dictionary with the direct values
            merged_dict = {**{k: v for k, v in hparams.items() if k in direct_keys}, **combo_dict}
            # get the model name
            model_name = '_'.join([model]+ [''.join([k[0] for k in key.split('_')]) + str(v) for key, v in merged_dict.items()])
            # check whether the model has already been trained
            if os.path.exists(logdir) and model_name in os.listdir(logdir):
                print(f"The model {model_name} has already been trained. Skipping.")
                continue
            # try to run the training function with the merged dictionary
            try:
                run_training(model, merged_dict, model_name, logdir=logdir, cli_args=cli_args)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    failed_runs.append(model_name)
                    failed_runs_reason.append("CUDA out of memory")
                    print("-"*80 + "\nCUDA out of memory. Skipping.\nSkipped models: " + str(failed_runs)+ "\n" + "-"*80)
                    shutil.rmtree(f"{logdir}/{model_name}")
                    continue
                else:
                    failed_runs.append(model_name)
                    failed_runs_reason.append(str(e))
                    print(str(e))
                    print("-"*80 + "\nUnknown error. Skipping.\nSkipped models: " + str(failed_runs) + "\n" + "-"*80)
                    shutil.rmtree(f"{logdir}/{model_name}")
                    continue
        # print the failed runs
        if failed_runs:
            print(f"The following runs failed: {failed_runs}")
            print(f"The reason for the failure was: {failed_runs_reason}")
    else:  # if there are no list values
        # get the model name
        model_name = '_'.join([model]+ [''.join([k[0] for k in key.split('_')]) + str(v) for key, v in hparams.items()])
        # run the training function with the original dictionary
        run_training(model, hparams, logdir=logdir, cli_args=cli_args)

def train_all_models(cli_args):

    if cli_args.LOG_DIR is None:
        logdir = f"logs/{datetime.today().strftime('%Y-%m-%d')}/"
    else:
        logdir = cli_args.LOG_DIR
    
    for model, hparams in args.HPARAMS.items():
        print(f"Training {model}.")
        check_hparams(model, hparams, logdir=logdir+model, cli_args=cli_args)

# helper functions
def none_or_str(value):
    if not value:
        return None
    return value


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--BATCH_SIZE", type=int, default=args.BATCH_SIZE)
    parser.add_argument("--NUM_WORKERS", type=int, default=args.NUM_WORKERS)
    parser.add_argument("--MAX_EPOCHS", type=int, default=args.MAX_EPOCHS)
    parser.add_argument("--ACCELERATOR", type=str, default=args.ACCELERATOR)
    parser.add_argument("--LOG_EVERY_N_STEPS", type=int, default=args.LOG_EVERY_N_STEPS)
    parser.add_argument("--LOG_DIR", type=none_or_str, default=None)
    parser.add_argument("--TRAIN_SCENARIO", type=str, default=args.TRAIN_SCENARIO)
    cli_args = parser.parse_args()

    train_all_models(cli_args)
