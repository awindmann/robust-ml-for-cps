import glob
import os
import yaml
import shutil
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



def run_finetuning(model, hparams, logdir, cli_args=None, model_name = None):
    """Trains a model with the given hyperparameters and logs the results.
    """
    print("--- Starting Training ---" + "-"*55)
    pl.seed_everything(42, workers=True)

    new_logdir = f"{logdir}/ftune_on_{cli_args.TRAIN_SCENARIO}/for_{cli_args.MAX_EPOCHS}_epochs"

    if cli_args is None:
        cli_args = dict(
            BATCH_SIZE=args.BATCH_SIZE,
            NUM_WORKERS=args.NUM_WORKERS,
            MAX_EPOCHS=args.MAX_EPOCHS,
            ACCELERATOR=args.ACCELERATOR,
            LOG_EVERY_N_STEPS=args.LOG_EVERY_N_STEPS
        )

    # load datamodule
    dm = ThreeTankDataModule(
        train_scenario=cli_args.TRAIN_SCENARIO,
        batch_size=cli_args.BATCH_SIZE, 
        num_workers=cli_args.NUM_WORKERS
    )

    # configure pl trainer
    callbacks = list()
    if cli_args.EARLY_STOPPING:
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
        logger = TensorBoardLogger(new_logdir, name=model_name, version=version, default_hp_metric=False, log_graph=False)
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

    # load latest checkpoint
    ckpt_path = glob.glob(f"{glob.escape(logdir)}/checkpoints/*.ckpt")
    # latest_version = len([version for version in os.listdir(f"{logdir}/{model_name}")]) - 1
    # ckpt_path = glob.glob(f"{logdir}/{model_name}/version_{latest_version}/checkpoints/*.ckpt")
    model = model.load_from_checkpoint(ckpt_path[0], train_scenario=cli_args.TRAIN_SCENARIO)

    if args.USE_LOGGER:
        print(f"Logging in {new_logdir}")
        # create directory for model
        os.makedirs(f"{new_logdir}", exist_ok=True)
        # save model architecture
        summary = torchinfo.summary(
            model, 
            verbose=0
        )
        with open(f"{new_logdir}/architecture.txt", "w") as f:
            f.write(str(summary))

    model_name = logdir.split("/")[-1]
    # train
    print(f"\nTraining {model_name}.")
    trainer.fit(model=model, datamodule=dm)

    # test
    trainer.test(model=model, datamodule=dm, ckpt_path="best")

    # visualization
    if args.PLOT_FORECAST:
        # load best model 
        if args.SAVE_CHECKPOINT:
            model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        fcast_overview(dm, model, idx=84, title=model_name, save_path=f"{new_logdir}/plots/")

def ftune_all_models(cli_args):

    # get every model in logdir based on the checkpoint files
    logdir = cli_args.LOG_DIR
    if logdir is None:
        raise ValueError("Please specify a log directory. (--LOGDIR)")
    if not os.path.exists(logdir):
        raise ValueError(f"The log directory {logdir} does not exist.")
    
    # get the model paths
    models = [root for root, dirs, files in os.walk(logdir) if "checkpoints" in dirs and not "for_" in root]

    failed_models = list()
    failed_reasons = list()
    print(f"Finetuning {len(models)} models.")
    for i, model_path in enumerate(models):
        print(f"Finetuning model {i+1}/{len(models)}: {model_path.split('/')[-1]}")
        # if there already exists a finetuned model, skip it
        if os.path.exists(f"{model_path}/ftune_on_{cli_args.TRAIN_SCENARIO}/for_{cli_args.MAX_EPOCHS}_epochs"):
            print(f"Skipping {model_path.split('/')[-1]} because it was already finetuned.")
            continue
        # load the hparams yaml file
        with open(model_path + "/hparams.yaml", "r") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)
        
        model_architecture = model_path.split("/")[-2]  # e.g. "MLP"

        if cli_args.DEBUG:  # throw errors
            torch.autograd.set_detect_anomaly(True)
            run_finetuning(model=model_architecture, hparams=hparams, cli_args=cli_args, logdir=model_path)
        else:
            try:
                run_finetuning(model=model_architecture, hparams=hparams, cli_args=cli_args, logdir=model_path)
            except Exception as e:
                print(e)
                failed_models.append(model_path.split("/")[-1])
                failed_reasons.append(e)
                # remove the finetuned model directory if it exists
                if os.path.exists(f"{model_path}/ftune_on_{cli_args.TRAIN_SCENARIO}/for_{cli_args.MAX_EPOCHS}"):
                    shutil.rmtree(f"{model_path}/ftune_on_{cli_args.TRAIN_SCENARIO}/for_{cli_args.MAX_EPOCHS}")
                continue
    if len(failed_models) > 0:
        print("-"*80)
        print(f"Failed models: {failed_models}")
        print(f"Reasons: {failed_reasons}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--TRAIN_SCENARIO", type=str, default=args.TRAIN_SCENARIO)
    parser.add_argument("--LOG_DIR", type=str)
    parser.add_argument("--BATCH_SIZE", type=int, default=args.BATCH_SIZE)
    parser.add_argument("--NUM_WORKERS", type=int, default=args.NUM_WORKERS)
    parser.add_argument("--MAX_EPOCHS", type=int, default=args.MAX_EPOCHS)
    parser.add_argument("--ACCELERATOR", type=str, default=args.ACCELERATOR)
    parser.add_argument("--LOG_EVERY_N_STEPS", type=int, default=args.LOG_EVERY_N_STEPS)
    parser.add_argument("--ALL_EPOCHS", action="store_true")
    parser.add_argument("--ALL_SCENARIOS", action="store_true")
    parser.add_argument("--EARLY_STOPPING", action="store_true")
    parser.add_argument("--DEBUG", action="store_true")
    cli_args = parser.parse_args()


    if cli_args.ALL_EPOCHS:
        epochs = [1, 5, 10, 20, 50]
    else:
        epochs = [cli_args.MAX_EPOCHS]
    if cli_args.ALL_SCENARIOS:
        scenarios = [
            # "fault",
            # "noise",
            # "duration",
            "scale",
            # "switch", 
            "q1+v3",
            "q1+v3+rest",
            "v12+v23",
            # "standard+",
            # "standard++",
            # "frequency",
            "time_warp"
        ]
    else:
        scenarios = [cli_args.TRAIN_SCENARIO]

    for scenario in scenarios:
        for epoch in epochs:
            cli_args.TRAIN_SCENARIO = scenario
            cli_args.MAX_EPOCHS = epoch
            print(f"Finetuning on {cli_args.TRAIN_SCENARIO} for {cli_args.MAX_EPOCHS} epochs.")
            ftune_all_models(cli_args=cli_args)
