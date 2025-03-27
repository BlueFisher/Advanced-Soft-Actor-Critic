import importlib.util
import logging
import shutil
from pathlib import Path

import numpy as np

from algorithm import config_helper

from .. import sac_main
from ..sac_main import Main
from ..utils import UnifiedElapsedTimer
from ..utils.enums import *
from .oc_agent import OC_MultiAgentsManager
from .option_selector_base import OptionSelectorBase


class OC_Main(Main):
    ma_manager: OC_MultiAgentsManager

    def __init__(self,
                 root_dir: Path | str,
                 config_dir: Path | str,
                 config_cat: str | None = None,
                 override: list[tuple[list[str], str]] | None = None,
                 train_mode: bool = True,
                 inference_ma_names: set[str] | None = None,
                 copy_model: str | None = None,
                 logger_in_file: bool = False,

                 render: bool = False,
                 env_args: dict | None = None,
                 envs: int | None = None,
                 max_iter: int | None = None,

                 unity_port: int | None = None,
                 unity_run_in_editor: bool = False,
                 unity_quality_level: int = 2,
                 unity_time_scale: float | None = None,

                 name: str | None = None,
                 disable_sample: bool = False,
                 use_env_nn: bool = False,
                 device: str | None = None,
                 last_ckpt: str | None = None,
                 nn: str | None = None):
        """
        root_dir: the root directory of asac
        config_path: the directory of config file
        """
        sac_main.MultiAgentsManager = OC_MultiAgentsManager

        self.root_dir = Path(root_dir)

        self.train_mode = train_mode
        self.inference_ma_names = inference_ma_names if inference_ma_names is not None else set()
        self.render = render

        self.unity_run_in_editor = unity_run_in_editor
        self.unity_quality_level = unity_quality_level
        self.unity_time_scale = unity_time_scale

        self.disable_sample = disable_sample
        self.force_env_nn = use_env_nn
        self.device = device
        self.last_ckpt = last_ckpt

        self._logger = logging.getLogger('oc')

        self._profiler = UnifiedElapsedTimer(self._logger)

        self.model_abs_dir, self._config_abs_dir = self._init_config(config_dir,
                                                                     config_cat,
                                                                     override,

                                                                     env_args,
                                                                     envs,
                                                                     max_iter,

                                                                     unity_port,

                                                                     name,
                                                                     nn)

        if logger_in_file:
            config_helper.add_file_logger(self.model_abs_dir.joinpath(f'log.log'))

        self._handle_copy_model(copy_model)

        self._init_env()
        self._init_oc()

        self._run()

    def _init_oc(self):
        for n, mgr in self.ma_manager:
            # If nn models exists, load saved model, or copy a new one
            saved_nn_abs_dir = mgr.model_abs_dir / 'nn'
            if not self.force_env_nn and saved_nn_abs_dir.exists():
                nn_abs_path = saved_nn_abs_dir / f'{mgr.config["sac_config"]["nn"]}.py'
                spec = importlib.util.spec_from_file_location(f'{self._get_relative_package(nn_abs_path)}.{mgr.config["sac_config"]["nn"]}',
                                                              nn_abs_path)
                self._logger.info(f'Loaded nn from existed {nn_abs_path}')
            else:
                nn_abs_path = self._config_abs_dir / f'{mgr.config["sac_config"]["nn"]}.py'

                spec = importlib.util.spec_from_file_location(f'{self._get_relative_package(nn_abs_path)}.{mgr.config["sac_config"]["nn"]}',
                                                              nn_abs_path)
                self._logger.info(f'Loaded nn in env dir: {nn_abs_path}')
                if not self.force_env_nn:
                    shutil.copytree(self._config_abs_dir, saved_nn_abs_dir,
                                    ignore=lambda _, names: [name for name in names
                                                             if name == '__pycache__'])

            nn = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(nn)
            mgr.config['sac_config']['nn'] = nn

            mgr.set_rl(OptionSelectorBase(obs_names=mgr.obs_names,
                                          obs_shapes=mgr.obs_shapes,
                                          d_action_sizes=mgr.d_action_sizes,
                                          c_action_size=mgr.c_action_size,
                                          model_abs_dir=mgr.model_abs_dir,
                                          device=self.device,
                                          ma_name=None if len(self.ma_manager) == 1 else n,
                                          train_mode=self.train_mode and n not in self.inference_ma_names,
                                          last_ckpt=self.last_ckpt,

                                          nn_config=mgr.config['nn_config'],
                                          **mgr.config['sac_config'],

                                          **mgr.config['oc_config'],

                                          replay_config=mgr.config['replay_config']))

    def _extra_step(self,
                    ma_d_action: dict[str, np.ndarray],
                    ma_c_action: dict[str, np.ndarray]):
        if not self.train_mode:
            pass
            # ma_option = self.ma_manager.get_option()

            # # TODO: multiple agent options
            # ma_option = {n: int(option[0]) for n, option in ma_option.items()}
            # self.env.send_option(ma_option)
