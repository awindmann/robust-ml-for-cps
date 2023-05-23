from typing import Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, Subset, DataLoader
from scipy.integrate import odeint
from tqdm import tqdm



class ThreeTankSimulation:
    """Simulates the three tank system.
    The system is simulated using the scipy odeint function.
    """
    def __init__(self, tank_1_lvl=0, tank_2_lvl=0, tank_3_lvl=0, seed=42):
        self.tank_levels = np.array([tank_1_lvl, tank_2_lvl, tank_3_lvl])
        self.seed = seed
        self.state_df = pd.DataFrame(columns=["q1", "q3", "kv1", "kv2", "kv3", "duration"])

    def add_state(self, q1: float, q3: float, kv1: float, kv2: float, kv3: float, duration: int, name=None) -> None:
        """Add a state to the state dataframe.
        A state consists of specific settings to the system's parameters.
        Args:
            q1 (float): inflow tank 1
            q3 (float): inflow tank 3
            kv1 (float): coefficient of the valve between tank 1 and 2
            kv2 (float): coefficient of the valve between tank 2 and 3
            kv3 (float): coefficient of the outgoing valve on tank 3
            duration (int): number of time steps of the state
            name (string): the name of the state
        """
        if name is not None:
            self.state_df.loc[name] = [q1, q3, kv1, kv2, kv3, duration]
        else:
            self.state_df.append(dict(q1=q1, q3=q3, kv1=kv1, kv2=kv2, kv3=kv3, duration=duration),
                                 ignore_index=True)

    @staticmethod
    def _system_dynamics_function(x, t, q1, q3, kv1, kv2, kv3):
        # ensure non-negative tank levels
        x1, x2, x3 = x * (x > 0)
        # ODE
        dh1_dt = q1 - kv1 * np.sign(x1 - x2) * np.sqrt(np.abs(x1 - x2))
        dh2_dt = kv1 * np.sign(x1 - x2) * np.sqrt(np.abs(x1 - x2)) \
                 - kv2 * np.sign(x2 - x3) * np.sqrt(np.abs(x2 - x3))
        dh3_dt = q3 + kv2 * np.sign(x2 - x3) * np.sqrt(np.abs(x2 - x3)) - kv3 * np.sqrt(x3)

        return dh1_dt, dh2_dt, dh3_dt

    def _compute_section(self, duration: int = 10, x0: np.array = np.array([30, 10, 50]),
                         kv1: float = 1, kv2: float = 1, kv3: float = 1,
                         q1: float = 1, q3: float = 1):
        t = np.array(range(duration))
        y = odeint(self._system_dynamics_function, x0, t, (q1, q3, kv1, kv2, kv3))
        # non-negativity
        y = y * (y > 0)
        y_stop = y[-1, :]
        return y, y_stop

    @staticmethod
    def _duplicate_row(row, factor):
        return pd.concat([row.copy()] * factor, axis = 1)

    def _configuration_seq(self, cycle: list, nb_of_cycles: int,
                           sd_q: float, sd_kv: float, sd_dur: float,
                           leaky: bool, periodic_inflow: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """generates configuration dataframes
        The configuration dataframe describes the state at every time step.
        Outputs original state configuration and configuration with noise.
        """
        # generate cycle of states
        seq = list()
        for i in range(nb_of_cycles):
            for state in cycle:
                if type(state) is str:
                    seq.append(self.state_df.loc[state])
                else:
                    seq.append((self.state_df.iloc[state, :]))
        seq_df = pd.concat(seq, axis=1).T.astype({"duration": int})
        seq_len = seq_df.shape[0]

        # add periodic inflow
        if periodic_inflow:
            amplitude = 0.5 * seq_df["q1"].max()
            wave = amplitude * np.cos(np.linspace(np.pi, 40*np.pi, 10000))

            q1_mask = seq_df["q1"] > 0
            q3_mask = seq_df["q3"] > 0
            
            seq_df.loc[q1_mask, "q1"] += wave[:q1_mask.sum()]
            seq_df.loc[q3_mask, "q3"] += wave[:q3_mask.sum()]

        # add noise
        np.random.seed(self.seed)
        seq_df_noise = seq_df.copy()
        if sd_q is not None:
            q_noise = np.random.normal(0, sd_q, 2 * seq_len)
            seq_df_noise["q1"] = seq_df["q1"] + q_noise[:seq_len]
            seq_df_noise["q3"] = seq_df["q3"] + q_noise[seq_len:]
            if not leaky:
                seq_df_noise["q1"].where(seq_df["q1"] > 0, other=0, inplace=True)  # no leaky inflow
                seq_df_noise["q3"].where(seq_df["q3"] > 0, other=0, inplace=True)  # (set back to 0 if no inflow)
        if sd_kv is not None:
            kv_noise = np.random.normal(0, sd_kv, 3 * seq_len)
            seq_df_noise["kv1"] = seq_df["kv1"] + kv_noise[:seq_len]
            seq_df_noise["kv2"] = seq_df["kv2"] + kv_noise[seq_len:2*seq_len]
            seq_df_noise["kv3"] = seq_df["kv3"] + kv_noise[2*seq_len:]
            if not leaky:
                seq_df_noise["kv1"].where(seq_df["kv1"] > 0, other=0, inplace=True)  # no leaky valve
                seq_df_noise["kv2"].where(seq_df["kv2"] > 0, other=0, inplace=True)
                seq_df_noise["kv3"].where(seq_df["kv3"] > 0, other=0, inplace=True)
        if sd_dur is not None:
            dur_noise = np.random.normal(0, sd_dur, seq_len)
            seq_df_noise["duration"] = round(seq_df["duration"] + dur_noise).astype(int)
        # no negative inflow etc.
        seq_df = seq_df.where(seq_df >= 0, 0)
        seq_df_noise = seq_df_noise.where(seq_df_noise >= 0, 0) 

        return seq_df, seq_df_noise

    @staticmethod
    def _export_config(seq_df, seq_df_noise, export_path):
        """exports state configuration dataframe
        Transforms the dataframe so that the state at every time step is exported.
        """
        seq0 = list()
        seq0_noise = list()
        for (_, row), (_, row_noise) in zip(seq_df.iterrows(), seq_df_noise.iterrows()):
            duration = int(row_noise.duration)  # actual duration
            seq0 += [row] * duration
            seq0_noise += [row_noise] * duration
        seq0_df = pd.concat(seq0_noise, axis=1).T
        seq0_df.to_csv(f"{export_path[:-4]}_config.csv", index=False)

    def simulate(self, cycle: list, nb_of_cycles: int = 10,
                 sd_q: float = None, sd_kv: float = None, sd_dur: float = None, sd_white_noise: float = None, 
                 leaky: bool = False, periodic_inflow = False,
                 export_path: str = None) -> np.array:
        """Simulates the dynamics in the three-tank system
        Args:
            cycle (list): sequence of states that compose a typical cycle.
                          Either list of integers or list of state names.
            nb_of_cycles (int): number of successive cycles to simulate
            sd_q (float): if set, white noise with this standard deviation is added to the inflow
            sd_kv (float): if set, white noise with this standard deviation is added to the valve coefficients
            sd_dur (float): if set, white noise with this standard deviation is added to the duration
            leaky (bool): if true, add noise on closed valves or stopped inflow
            periodic_inflow (bool): if true, add periodic variation to the inflow
            export_path (str): if set, save simulation data at export path
        """
        seq_denoised, seq = self._configuration_seq(cycle, nb_of_cycles, sd_q, sd_kv, sd_dur, leaky, periodic_inflow)

        y_ls = []
        y_stop = self.tank_levels
        for config in tqdm(seq.itertuples(), total=len(seq)):
            y, y_stop = self._compute_section(duration=config.duration, x0=y_stop,
                                            kv1=config.kv1, kv2=config.kv2, kv3=config.kv3,
                                            q1=config.q1, q3=config.q3)
            y_ls.append(y)
        y_out = np.concatenate(y_ls)

        if sd_white_noise is not None:
            np.random.seed(self.seed)
            y_out += np.random.normal(0, sd_white_noise, y_out.shape)

        if export_path is not None:
            y_df = pd.DataFrame(y_out, columns=['h1', 'h2', 'h3'])
            y_df.to_csv(export_path, index=False)
            self._export_config(seq_denoised, seq, export_path)

        return y_out



class ThreeTankDataset(Dataset):
    """Three tank dataset
    A sample consists of a random time window + consecutive time window
    Args:
        file (str): path to csv file containing the data
        input_len (int): length of input sequence
        pred_len (int): length of prediction sequence
        nb_of_samples (int): number of samples to draw
        ordered_samples (bool): if true, samples are arranged in order of time
        faulty_input (bool): if true, input sequence is faulty
        seed (int): random seed
    """
    def __init__(self,
                 file,
                 input_len=250,  # should contain at least 4 phases (one standard cycle)
                 pred_len=50,
                 nb_of_samples=1000,
                 ordered_samples=True,
                 faulty_input=False,
                 seed=42
                 ):
        super().__init__()
        # read data
        self.X = pd.read_csv(file)

        self.input_len = input_len
        self.pred_len = pred_len
        self.phase_len = 50  # defined in simulation as duration of one phase
        self.nb_of_samples = nb_of_samples
        self.nb_of_features = self.X.shape[1]

        self.ordered_samples = ordered_samples
        self.faulty_input = faulty_input
        self.sample_idxs = self._create_samples(seed)
        self.fault_mask = self._create_faults(seed)

    def _create_samples(self, seed):
        """Create array of random start numbers"""
        np.random.seed(seed)
        start_idxs = np.random.randint(0, self.X.shape[0] - self.input_len - self.pred_len, self.nb_of_samples)
        if self.ordered_samples:  # for now important due to train/test split in datamodule
            start_idxs = np.sort(start_idxs)
        return start_idxs
    
    def _create_faults(self, seed):
        """Create a mask that simulates faulty sensors"""
        fault_mask = np.ones((self.nb_of_samples, self.input_len, self.nb_of_features), dtype=np.float32)
        if self.faulty_input:
            np.random.seed(seed)
            for i in range(self.nb_of_samples):
                # choose a random sensor and a random position in the sequence
                sensor = np.random.randint(0, self.nb_of_features)
                # either add a point anomaly or simulate a dead sensor
                if np.random.rand() < 0.5:
                    pos = np.random.randint(0, self.input_len)
                    fault_mask[i, pos, sensor] = np.random.rand() * 8 + 2  # random value between 2 and 10
                else:
                    # make sure the fault is not at the end of the sequence. The last {phase_len} steps should not be faulty
                    pos = np.random.randint(0, self.input_len - self.phase_len - 1)
                    # choose a random number of consecutive timesteps to be faulty
                    # no longer than a phase and no longer than the remaining sequence
                    nb_of_steps = np.random.randint(1, min(self.phase_len, self.input_len - self.phase_len - pos))
                    fault_mask[i, pos:pos+nb_of_steps, sensor] = 0
        return fault_mask

    def __len__(self):
        """Size of dataset"""
        return self.nb_of_samples

    def __getitem__(self, index):
        """Get one sample
        Simple setup: always yield two samples, x1(t) and concurrent sample x2(t+input_len) (without configurations).
        Note that the model can effectively see x2 via a different x1 if dataloader is not chronological.
        """
        start_idx = self.sample_idxs[index]
        x1 = self.X.iloc[start_idx: start_idx + self.input_len].to_numpy(dtype=np.float32)
        x2 = self.X.iloc[start_idx + self.input_len: start_idx + self.input_len + self.pred_len].to_numpy(dtype=np.float32)
        if self.faulty_input:
            # element-wise multiplication of input with fault mask
            x1 = x1 * self.fault_mask[index]
        return x1, x2
    

class ThreeTankDataModule(pl.LightningDataModule):
    """Data module for three tank dataset
    Args:
        train_scenario (str): scenario used for training
        batch_size (int): batch size
        num_workers (int): number of workers for dataloader
        pin_memory (bool): pin memory for dataloader
        train_split (float): fraction of data used for training
        val_split (float): fraction of data used for validation
        seed (int): random seed
    """
    def __init__(self,
                 train_scenario="standard",
                 batch_size=64, num_workers=8, pin_memory=False,
                 train_split=0.5, val_split=0.25,
                 seed=42):
        super(ThreeTankDataModule, self).__init__()

        self.scenarios = [
            "standard",
            "fault",
            "noise",
            "duration",
            "scale",
            "switch",
            "q1+v3",
            "q1+v3+rest",
            "v12+v23",
            "standard+",
            "standard++",
            "frequency",
            "time_warp"
        ]
        self.train_scenario = train_scenario
        assert self.train_scenario in self.scenarios, f"train_scenario must be one of {self.scenarios}"

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_split = train_split
        self.val_split = val_split

        self.ds_dict = {}  # popoulated in setup()

        self.seed = seed

    def setup(self, stage=None) -> None:
        # (in distribution) dataset to train/finetune on
        for scenario in self.scenarios:

            if scenario == self.train_scenario:
                nb_of_samples = 1000
            else:
                nb_of_samples = 100

            if scenario == "fault":
                self.ds_dict[scenario] = ThreeTankDataset(
                    f"data/processed/simulation_standard.csv",  # TODO remove simulation naming or separate from augmentation
                    nb_of_samples=nb_of_samples,
                    ordered_samples=True,
                    faulty_input=True,  # simulate faulty sensors
                    seed=self.seed + 1234  # sample from training dataset but use different seed to avoid overfitting
                )
            else:
                self.ds_dict[scenario] = ThreeTankDataset(
                    f"data/processed/simulation_{scenario}.csv", 
                    nb_of_samples=nb_of_samples,
                    ordered_samples=True,
                    seed=self.seed
                )

    def _get_val_start_idx(self, ds):
        return int(len(ds) * self.train_split)
    
    def _get_test_start_idx(self, ds):
        return int(len(ds) * (self.train_split + self.val_split))

    def train_dataloader(self) -> DataLoader:
        # output is a tuple of two tensors ([batch_size, seq_len, features], [batch_size, features])
        # [train | val | test] split
        ds_train = Subset(
            self.ds_dict[self.train_scenario], 
            range(0, self._get_val_start_idx(self.ds_dict[self.train_scenario]))
            )
        return DataLoader(
            ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self) -> DataLoader:
        val_dl_list = [
            DataLoader(
                Subset(
                    self.ds_dict[scenario],
                    range(
                        self._get_val_start_idx(self.ds_dict[scenario]),
                        self._get_test_start_idx(self.ds_dict[scenario])
                        )
                    ),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
                )
            for scenario in self.scenarios
        ]
        return val_dl_list

    def test_dataloader(self) -> DataLoader:
        test_dl_list = [
            DataLoader(
                Subset(
                    self.ds_dict[scenario],
                    range(
                        self._get_test_start_idx(self.ds_dict[scenario]),
                        len(self.ds_dict[scenario])
                        )
                    ),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
                )
            for scenario in self.scenarios
        ]
        return test_dl_list
