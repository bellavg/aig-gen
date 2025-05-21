#!/usr/bin/env python3
try:
    import graph_tool as gt
except ModuleNotFoundError:
    print("Graph tool not found")
import os
import pathlib
import warnings

import torch
import sys
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict  # Added OmegaConf for merging
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

# Path setup
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT)) # Add project root to Python path

# Corrected imports:
from src import utils # Changed from 'import utils'
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from src.diffusion_model import LiftedDenoisingDiffusion
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
#
import torch# Set matmul precision for Tensor Cores
# Options: 'highest' (default), 'high', 'medium'
# 'high' or 'medium' can leverage Tensor Cores for float32 matrix multiplications
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7: # Check for Volta or newer
    print("Setting float32 matmul precision to 'high' for Tensor Cores.")
    torch.set_float32_matmul_precision('high')

import torch._dynamo
torch._dynamo.config.suppress_errors = True

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    # Load the config from the checkpoint's directory, relative to the checkpoint file itself
    saved_cfg_path = pathlib.Path(cfg.general.test_only).parents[1] / '.hydra' / 'config.yaml'
    if not os.path.exists(saved_cfg_path):
        raise FileNotFoundError(f"Saved config not found at {saved_cfg_path}. "
                                f"Ensure the checkpoint path '{cfg.general.test_only}' is correct and "
                                f"the '.hydra' directory exists in its parent's parent.")
    saved_cfg_base = OmegaConf.load(saved_cfg_path)

    # Start with a copy of the currently passed cfg, then override with saved_cfg_base,
    # then selectively apply specific overrides for testing.
    # This allows new command-line args to be respected if they aren't part of saved_cfg_base.
    # And ensures critical saved params are used.

    # Create a new config object, starting with the loaded saved_cfg_base
    # Then, merge the current cfg into it. This way, current cfg can override some
    # general settings or add new ones not present in the saved config.
    # However, for model and dataset, we usually want to stick to what was saved.

    # Let's adopt the strategy from your `main_v1.py` for `get_resume` as it's clearer
    # for ensuring the original experiment's config is mostly used.

    original_test_only_path = cfg.general.test_only  # Save the new test_only path

    # Load the config associated with the checkpoint
    cfg_for_model_load = saved_cfg_base.copy()  # Use the config stored with the checkpoint
    cfg_for_model_load.general.test_only = original_test_only_path  # Use the provided path for loading
    cfg_for_model_load.general.resume = None  # We are in test_only mode

    if cfg_for_model_load.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(original_test_only_path, cfg=cfg_for_model_load,
                                                                **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(original_test_only_path, cfg=cfg_for_model_load,
                                                              **model_kwargs)

    # The config to return for the rest of the script should be based on the saved one,
    # but with the current test_only path and a modified name.
    # Also, allow updates from the current `cfg` for keys not in `saved_cfg_base`.
    final_cfg = utils.update_config_with_new_keys(saved_cfg_base.copy(), cfg)  # cfg can add new keys to saved_cfg_base
    final_cfg.general.test_only = original_test_only_path
    final_cfg.general.resume = None  # Explicitly ensure not in resume mode
    final_cfg.general.name = saved_cfg_base.general.name + '_test'

    return final_cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    # Assuming 'src' is one level down from project root, adjust if main.py is elsewhere
    project_root_guess = pathlib.Path(current_path).parent

    # If cfg.general.abs_path_to_project_root is defined and valid, prefer it
    if hasattr(cfg.general, 'abs_path_to_project_root') and cfg.general.abs_path_to_project_root:
        root_dir = pathlib.Path(cfg.general.abs_path_to_project_root)
    else:
        # Fallback: try to infer root relative to 'outputs' if that's where resume path might be relative to
        # This part is tricky without knowing where cfg.general.resume is anchored.
        # A common pattern is that cfg.general.resume is like "outputs/run_name/checkpoints/model.ckpt"
        # If so, root_dir would be the parent of "outputs".
        # For robustness, it's better if cfg.general.resume is an absolute path or resolvable
        # from a well-defined project root.
        # The original fixed_bug code had: root_dir = current_path.split('outputs')[0]
        # This assumes 'outputs' is a subdir of root_dir.
        # Let's assume cfg.general.resume is relative to the original hydra output dir structure.
        # The actual checkpoint path for loading is cfg.general.resume.
        # The config path is relative to that.

        # Path to the checkpoint file itself
        resume_checkpoint_path_str = cfg.general.resume
        if not os.path.isabs(resume_checkpoint_path_str):
            # If resume_path is relative, it's typically relative to the original run's output dir,
            # or the CWD if re-running from a different location.
            # This is a common source of issues. Best if `cfg.general.resume` is absolute or easily made so.
            # For now, assume it's relative to the project root if not absolute.
            resume_checkpoint_path_str = os.path.join(project_root_guess, resume_checkpoint_path_str)
            print(f"Interpreted relative resume path as: {resume_checkpoint_path_str}")

    resume_checkpoint_path = pathlib.Path(resume_checkpoint_path_str)
    if not resume_checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")

    # Path to the .hydra config of the run being resumed
    saved_cfg_path = resume_checkpoint_path.parents[1] / '.hydra' / 'config.yaml'
    if not saved_cfg_path.exists():
        raise FileNotFoundError(f"Saved config not found at {saved_cfg_path} for checkpoint {resume_checkpoint_path}")

    print(f"Loading saved config from: {saved_cfg_path}")
    saved_cfg_base = OmegaConf.load(saved_cfg_path)

    # Preserve critical parts from the saved config
    preserved_model_cfg = saved_cfg_base.model
    preserved_dataset_name = saved_cfg_base.dataset.name
    # Potentially preserve dataset-specific parameters if they are critical
    preserved_dataset_params = saved_cfg_base.dataset.copy()

    # Start with the new `cfg` (from current command/defaults), then override with `saved_cfg_base`.
    # Then, apply specific overrides from the *new* `cfg` that are meant for adaptation (e.g., epochs, LR).

    # Create the new config: Start with the new cfg, then selectively update from saved_cfg
    # This allows current command line args to take precedence for things like number of epochs, LR, etc.
    # while restoring model architecture, dataset type etc. from the saved run.

    new_cfg = cfg.copy()  # Current command-line/default config

    # Override model and dataset sections entirely from the saved config
    # to ensure consistency with the loaded checkpoint.
    new_cfg.model = preserved_model_cfg
    # new_cfg.dataset = preserved_dataset_params # Use the whole dataset block from saved
    # Or, more carefully:
    new_cfg.dataset.name = preserved_dataset_name
    # Copy other dataset params from saved_cfg if they are not in current cfg or if they should be preserved
    for key, value in preserved_dataset_params.items():
        if key not in new_cfg.dataset:
            OmegaConf.set_struct(new_cfg.dataset, True)  # Allow adding new keys
            with open_dict(new_cfg.dataset):
                new_cfg.dataset[key] = value
    OmegaConf.set_struct(new_cfg.dataset, False)  # Disallow adding new keys after this

    # Update general section: use saved name, set resume path, allow new abs_path
    new_cfg.general.name = saved_cfg_base.general.name + '_resume'
    new_cfg.general.resume = str(resume_checkpoint_path)  # Use the full path to the checkpoint
    if hasattr(cfg.general, 'abs_path_to_project_root'):  # If new config specifies a root, use it
        new_cfg.general.abs_path_to_project_root = cfg.general.abs_path_to_project_root
    elif hasattr(saved_cfg_base.general, 'abs_path_to_project_root'):  # Else, use saved one
        new_cfg.general.abs_path_to_project_root = saved_cfg_base.general.abs_path_to_project_root

    # Allow specific training parameters (like n_epochs, lr) to be overridden by the current `cfg`
    # These are already in `new_cfg` if they were in the initial `cfg`.
    # Example: new_cfg.train.n_epochs = cfg.train.n_epochs (already done by new_cfg = cfg.copy())

    # Load model using the new_cfg which is now a blend
    # The model's internal self.hparams will be set from new_cfg.
    if new_cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(
            str(resume_checkpoint_path), cfg=new_cfg, strict=False, **model_kwargs
            # strict=False might be needed if cfg slightly differs
        )
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(
            str(resume_checkpoint_path), cfg=new_cfg, strict=False, **model_kwargs
        )

    # Ensure the model object itself uses this carefully constructed new_cfg
    # The load_from_checkpoint should handle setting model.cfg or model.hparams
    # If not, uncomment:
    # model.cfg = new_cfg
    # model.hparams.cfg = new_cfg # Or however PL stores it

    return new_cfg, model


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    model_cfg = cfg["model"]  # Get model config for convenience

    # Common initializations
    # Adjusted import paths for metrics
    train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
    visualization_tools = None
    extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()
    sampling_metrics = None  # Initialize sampling_metrics

    if dataset_config["name"] in ['sbm', 'comm20', 'planar']:
        from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos  # Adjusted path
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, \
            Comm20SamplingMetrics  # Adjusted path
        from analysis.visualization import NonMolecularVisualization  # Adjusted path

        datamodule = SpectreGraphDataModule(cfg)
        if dataset_config['name'] == 'sbm':
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config['name'] == 'comm20':
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        else:  # planar
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        # train_metrics already initialized
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.get('extra_features') is not None:  # Use .get for safety
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        # domain_features remains DummyExtraFeatures for these

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

    elif dataset_config["name"] == 'aig':
        # BEGIN AIG-SPECIFIC BLOCK
        from datasets.aig_custom_dataset import AIGDataModule, AIGDatasetInfos  # Adjusted path
        from metrics.aig_metrics import AIGSamplingMetrics  # Adjusted path
        from analysis.visualization import AIGVisualization  # Using AIGSpecific Visualization

        datamodule = AIGDataModule(cfg)
        dataset_infos = AIGDatasetInfos(datamodule, dataset_config)  # dataset_infos computes its own dims

        print(f"INFO: Instantiating AIGSamplingMetrics for dataset: {dataset_config['name']}")
        sampling_metrics = AIGSamplingMetrics(datamodule)
        visualization_tools = AIGVisualization(cfg=cfg, dataset_infos=dataset_infos)

        # For AIGs, extra_features and domain_features are typically DummyExtraFeatures
        # as per original DiGress setup for AIGs.
        # If specific extra features were intended for AIG in discrete mode:
        # if cfg.model.type == 'discrete' and cfg.model.get('extra_features') is not None:
        #     extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        # else:
        #     extra_features = DummyExtraFeatures()
        # domain_features = DummyExtraFeatures()

        # Note: AIGDatasetInfos calls compute_input_output_dims internally,
        # so no explicit call here is needed if that's the case.
        # If it doesn't, then:
        # dataset_infos.compute_input_output_dims(datamodule=datamodule,
        #                                         extra_features=extra_features,
        #                                         domain_features=domain_features)
        # END AIG-SPECIFIC BLOCK

    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}

    # Model instantiation / Resuming logic
    model = None
    if cfg.general.test_only:
        cfg, model = get_resume(cfg, model_kwargs)
        # Important: chdir to the original experiment's output directory.
        # This is where results, sub-folders like 'graphs', 'chains' should go.
        # The path should be relative to the checkpoint file.
        # Example: if test_only is 'outputs/my_run/checkpoints/epoch=10.ckpt',
        # we want to chdir to 'outputs/my_run'.
        original_hydra_dir = pathlib.Path(cfg.general.test_only).parents[1]
        os.chdir(original_hydra_dir)
        print(f"Changed CWD to: {os.getcwd()} for test_only mode.")

    elif cfg.general.resume is not None:
        cfg, model = get_resume_adaptive(cfg, model_kwargs)
        # Similar to test_only, chdir to the resumed experiment's output directory.
        original_hydra_dir = pathlib.Path(cfg.general.resume).parents[1]
        os.chdir(original_hydra_dir)
        print(f"Changed CWD to: {os.getcwd()} for resume mode.")

    # If not resuming or testing, create model from scratch
    # The CWD will be the new Hydra output directory (e.g., outputs/YYYY-MM-DD/HH-MM-SS)
    if model is None:
        if cfg.model.type == 'discrete':
            model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
        else:
            model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    utils.create_folders(cfg)  # Creates folders in the current working directory

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints",  # Saves inside the current Hydra run directory
            filename='best-val_NLL-{epoch:03d}',  # Changed from '{epoch}'
            monitor='val/epoch_NLL',
            save_top_k=cfg.train.get('save_top_k', 5),  # Use config or default
            mode='min',
            every_n_epochs=1)  # Check every epoch

        last_ckpt_save = ModelCheckpoint(
            dirpath=f"checkpoints",
            filename='last-{epoch:03d}',  # Changed from 'last'
            every_n_epochs=cfg.general.check_val_every_n_epochs,  # Save at validation frequency
            save_top_k=-1  # Keep all such 'last' checkpoints, or 1 for only the most recent
        )
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.get('ema_decay', 0) > 0:  # Use .get for safety
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()


    trainer_strategy = None
    if use_gpu:
        if cfg.general.gpus > 1:
            # ddp_find_unused_parameters_true might be slow, consider ddp if no issues.
            trainer_strategy = "ddp_find_unused_parameters_true"
        else:
            trainer_strategy = "auto"
    else:
        trainer_strategy = "auto"

    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy=trainer_strategy,
        accelerator='gpu' if use_gpu else 'cpu',
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=cfg.general.name == 'debug',
        enable_progress_bar=cfg.train.get('progress_bar', False),  # Use .get
        callbacks=callbacks,
        log_every_n_steps=cfg.general.get('log_every_steps', 50) if name != 'debug' else 1,  # Use .get
        logger=[]  # DiGress handles WandB logging internally
    )

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule,
                    ckpt_path=cfg.general.resume if cfg.general.resume is not None else None)
        if cfg.general.name not in ['debug', 'test'] and not cfg.general.get('skip_final_test', False):  # Use .get
            print("Training finished. Starting final testing...")
            best_ckpt_path = checkpoint_callback.best_model_path if 'checkpoint_callback' in locals() and hasattr(
                checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path else None
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                print(f"Loading best model for final test: {best_ckpt_path}")
                # Need to load the model with the specific checkpoint for testing
                # The config used here should be the one associated with the *best_ckpt_path*
                # or ensure that current cfg + model_kwargs is compatible.
                # For simplicity, assuming current cfg is compatible for re-loading.
                if model_cfg.type == 'discrete':
                    test_model = DiscreteDenoisingDiffusion.load_from_checkpoint(best_ckpt_path, cfg=cfg,
                                                                                 **model_kwargs)
                else:
                    test_model = LiftedDenoisingDiffusion.load_from_checkpoint(best_ckpt_path, cfg=cfg, **model_kwargs)
                trainer.test(test_model, datamodule=datamodule)
            else:
                print(
                    "No best model checkpoint found to test, or checkpoint_callback not used. Testing with last model state from training.")
                trainer.test(model, datamodule=datamodule)
    else:  # Test only mode
        print(f"Starting testing with checkpoint: {cfg.general.test_only}")
        # model is already loaded with cfg.general.test_only by get_resume
        trainer.test(model, datamodule=datamodule)

        if cfg.general.evaluate_all_checkpoints:
            print("Evaluating all checkpoints in the directory...")
            # directory should be where cfg.general.test_only (the specific checkpoint) is located
            # which is original_hydra_dir / "checkpoints"
            checkpoints_directory = pathlib.Path(cfg.general.test_only).parent

            if not checkpoints_directory.is_dir():
                print(f"Error: Checkpoints directory not found: {checkpoints_directory}")
                return

            print(f"Evaluating checkpoints in: {checkpoints_directory}")
            files_list = sorted(os.listdir(checkpoints_directory))  # Sort for consistent order

            for file_name in files_list:
                # Flexible check for checkpoint files, e.g. 'best-val_NLL...' or 'last-...'
                if file_name.endswith('.ckpt'):
                    ckpt_path = os.path.join(checkpoints_directory, file_name)
                    if ckpt_path == cfg.general.test_only:  # Already tested this one
                        continue
                    print(f"Loading checkpoint for evaluation: {ckpt_path}")
                    # Important: When loading other checkpoints, ensure they use their *own* configs
                    # or that the current `cfg` is compatible. For simplicity, reusing current `cfg` and `model_kwargs`.
                    # A more robust way would be to load the config associated with each checkpoint.
                    if model_cfg.type == 'discrete':  # Use model_cfg from the initial config for type
                        eval_model = DiscreteDenoisingDiffusion.load_from_checkpoint(ckpt_path, cfg=cfg, **model_kwargs)
                    else:
                        eval_model = LiftedDenoisingDiffusion.load_from_checkpoint(ckpt_path, cfg=cfg, **model_kwargs)
                    trainer.test(eval_model, datamodule=datamodule)


if __name__ == '__main__':
    main()
