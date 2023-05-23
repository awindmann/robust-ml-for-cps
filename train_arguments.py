TRAIN_SCENARIO = "standard"
USE_LOGGER = True
USE_LOGGER_VERSIONING = False
SAVE_CHECKPOINT = True
CONTINUE_TRAIN = False
LOG_EVERY_N_STEPS = 1

EVAL_OOD = True
LOSS_FCT = "MSE"
BATCH_SIZE = 64
NUM_WORKERS = 8
MAX_EPOCHS = 1000
ACCELERATOR = "gpu"

TRAINER_CONFIG = dict(
    accelerator=ACCELERATOR,
    devices=1,  # for benchmarking for research papers, use 1 device only
    max_epochs=MAX_EPOCHS,
    enable_checkpointing=True if SAVE_CHECKPOINT else False,
    # precision=16
)

PLOT_FORECAST = True

HPARAMS = dict(
    MLP = dict(
        d_hidden_layers=[
            [256, 512, 256]
            ],
        batch_norm=[False],
    ),
    TCN = dict(
        kernel_size=[9],
        num_channels= [(64, 128, 64)],
        dropout= [0]
        ),
    GRU = dict(
        d_hidden=[256],
        n_layers=[1],
        autoregressive=[False, True],
    ),
    TcnAe = dict(
        latent_dim=[16],
        enc_tcn1_in_dims=[(3, 50, 40, 30)],
        enc_tcn1_out_dims=[(50, 40, 30, 10)],
        enc_tcn2_in_dims=[(10, 8, 6, 3)],
        enc_tcn2_out_dims=[(8, 6, 3, 1)],
        kernel_size=[15],
    ),
    Transformer = dict(
        d_model=[16],
        d_ff=[256],
        n_layers_enc=[4],
        n_layers_dec=[4],
        n_heads=[4],
        dropout = [0],
        norm_first = [False],
        conv_embed = [False, True],
        conv_kernel_size = [3],
    )
)
