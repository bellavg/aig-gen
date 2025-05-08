import os
import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig, OmegaConf # Added OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

# Assuming utils is in the same directory or accessible via PYTHONPATH
try:
    from . import utils # Use relative import if main.py is inside src
except ImportError:
    import utils # Fallback to direct import if main.py is at the root or src is added to path

# --- Metrics Imports ---
# Keep abstract and training metrics
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete # Only need discrete for AIG

# --- Model & Diffusion Imports ---
# from diffusion_model import LiftedDenoisingDiffusion # Not needed if only using discrete
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from datasets.aig_dataset import AIGDataModule, AIGDatasetInfos
# --- AIG Dataset Imports ---
# Import your custom AIG classes directly


warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    # Assuming AIG uses the discrete model
    if cfg.model.type != 'discrete':
        print(f"WARNING: test_only expects discrete model for AIG, but found {cfg.model.type}. Attempting load anyway.")
        # Add logic for other model types if needed, otherwise raise error or default to discrete
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    current_path = os.path.dirname(os.path.realpath(__file__))
    try:
        root_dir = current_path.split('outputs')[0]
    except IndexError:
        root_dir = current_path

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if not os.path.exists(resume_path):
        print(f"ERROR: Checkpoint path for resuming not found: {resume_path}")
        print("Starting training from scratch.")
        cfg.general.resume = None
        return cfg, None

    print(f"Attempting to resume from: {resume_path}")
    try:
        # Assuming AIG uses the discrete model
        if cfg.model.type != 'discrete':
             print(f"WARNING: resume expects discrete model for AIG, but found {cfg.model.type}. Attempting load anyway.")
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
        print("Successfully loaded model from checkpoint.")
        new_cfg = model.cfg

        # Update loaded config with current config values
        for category in cfg:
            if category not in new_cfg: new_cfg[category] = {}
            if isinstance(cfg[category], DictConfig):
                for arg in cfg[category]:
                    OmegaConf.set_struct(new_cfg[category], True)
                    new_cfg[category][arg] = cfg[category][arg]
                    OmegaConf.set_struct(new_cfg[category], False)
            else:
                 new_cfg[category] = cfg[category]

        new_cfg.general.resume = resume_path
        if not new_cfg.general.name.endswith('_resume'):
             new_cfg.general.name = new_cfg.general.name + '_resume'

        new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
        return new_cfg, model

    except Exception as e:
        print(f"ERROR: Failed to load checkpoint from {resume_path}: {e}")
        print("Starting training from scratch.")
        cfg.general.resume = None
        return cfg, None


@hydra.main(version_base='1.3', config_path='./', config_name='aig_full.yaml')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]

    # --- Specific AIG Dataset Setup ---
    if dataset_config["name"] != 'aig':
        raise ValueError(f"This main.py is configured only for the 'aig' dataset, found '{dataset_config['name']}'")

    print("Setting up AIG dataset...")
    datamodule = AIGDataModule(cfg)
    dataset_infos = AIGDatasetInfos(datamodule, cfg)

    # Use abstract metrics for training loss calculation
    train_metrics = TrainAbstractMetricsDiscrete()

    # Use Dummy features unless specific AIG features are defined and configured
    if cfg.model.type == 'discrete' and getattr(cfg.model, 'extra_features', None) is not None:
        # If you implement AIG-specific extra features, import and use them here
        # from diffusion.extra_features_aig import AigExtraFeatures # Example
        # extra_features = AigExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        print("WARNING: cfg.model.extra_features is set, but no specific AIG feature extractor is defined. Using DummyExtraFeatures.")
        extra_features = DummyExtraFeatures() # Placeholder
    else:
        extra_features = DummyExtraFeatures()
    # Domain features are typically molecular, use Dummy for AIGs
    domain_features = DummyExtraFeatures()

    # Compute input/output dimensions based on dataset and features
    if dataset_infos:
         dataset_infos.compute_input_output_dims(datamodule=datamodule,
                                                 extra_features=extra_features,
                                                 domain_features=domain_features)
    else:
         raise ValueError("AIG Dataset Infos not initialized correctly.")

    # --- Prepare Model Arguments ---
    # Removed sampling_metrics and visualization_tools
    model_kwargs = {'dataset_infos': dataset_infos,
                    'train_metrics': train_metrics,
                    'extra_features': extra_features,
                    'domain_features': domain_features}

    model_to_load = None # Variable to hold the loaded model during resume

    # --- Resume/Test Logic ---
    if cfg.general.test_only:
        print(f"Attempting to load model for testing from: {cfg.general.test_only}")
        cfg, model_to_load = get_resume(cfg, model_kwargs)
        if model_to_load is None: print("ERROR: Failed to load model for testing. Exiting."); return
        if os.path.exists(cfg.general.test_only):
             try:
                 checkpoint_dir = os.path.dirname(os.path.abspath(cfg.general.test_only))
                 output_dir = os.path.dirname(checkpoint_dir) if os.path.basename(checkpoint_dir) == 'checkpoints' else checkpoint_dir
                 if os.path.isdir(output_dir): os.chdir(output_dir); print(f"Changed directory to: {output_dir}")
                 else: print(f"Warning: Could not determine valid output directory from checkpoint path {cfg.general.test_only}")
             except Exception as e: print(f"Warning: Error changing directory based on checkpoint path: {e}")
        else: print(f"ERROR: Checkpoint path for testing not found: {cfg.general.test_only}"); return

    elif cfg.general.resume is not None:
        print(f"Attempting to resume training from: {cfg.general.resume}")
        cfg, model_to_load = get_resume_adaptive(cfg, model_kwargs)
        if model_to_load is not None: # Resume successful
            if os.path.exists(cfg.general.resume):
                try:
                     checkpoint_dir = os.path.dirname(os.path.abspath(cfg.general.resume))
                     output_dir = os.path.dirname(checkpoint_dir) if os.path.basename(checkpoint_dir) == 'checkpoints' else checkpoint_dir
                     if os.path.isdir(output_dir): os.chdir(output_dir); print(f"Changed directory to: {output_dir}")
                     else: print(f"Warning: Could not determine valid output directory from resume path {cfg.general.resume}")
                except Exception as e: print(f"Warning: Error changing directory based on resume path: {e}")
        # else: Resume failed, continue to train from scratch (handled in get_resume_adaptive)

    # --- Setup Folders and Initialize Model ---
    utils.create_folders(cfg)

    if model_to_load is None:
        print("Initializing new model instance...")
        # Assuming AIG uses the discrete model
        if cfg.model.type != 'discrete':
             raise ValueError(f"AIG dataset requires model.type='discrete', but found '{cfg.model.type}'")
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        print("Using loaded model instance.")
        model = model_to_load

    # --- Callbacks ---
    callbacks = []
    if cfg.train.save_model:
        ckpt_dir = f"checkpoints/{cfg.general.name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='{epoch}',
            monitor='val/epoch_NLL',
            save_top_k=cfg.train.save_top_k,
            mode='min',
            every_n_epochs=cfg.general.save_every_n_epochs,
            save_last=True
        )
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        try:
            from ema import EMA
            ema_callback = EMA(decay=cfg.train.ema_decay)
            callbacks.append(ema_callback)
        except ImportError:
            print("Warning: EMA callback specified but EMA class could not be imported.")

    # --- Trainer Setup ---
    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run=True. ")

    logger_list = []
    if cfg.general.wandb != 'disabled':
        try:
            from pytorch_lightning.loggers import WandbLogger
            import wandb
            if cfg.general.wandb == 'online':
                 if not wandb.login(key=os.environ.get("WANDB_API_KEY", None), relogin=False):
                     print("Warning: Wandb is set to 'online' but not logged in. Disabling Wandb.")
                     cfg.general.wandb = 'disabled'

            if cfg.general.wandb != 'disabled':
                wandb_logger = WandbLogger(
                    name=cfg.general.name,
                    project=f'graph_ddm_{cfg.dataset.name}', # Project name uses 'aig'
                    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                    save_dir="logs/",
                    mode=cfg.general.wandb
                )
                logger_list.append(wandb_logger)
        except ImportError:
            print("Warning: Wandb logger specified but 'wandb' or 'WandbLogger' could not be imported.")


    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    accelerator_strategy = 'gpu' if use_gpu else 'cpu'
    devices_strategy = cfg.general.gpus if use_gpu else 1
    strategy = "ddp_find_unused_parameters_true" if use_gpu and cfg.general.gpus > 1 else "auto"

    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy=strategy,
                      accelerator=accelerator_strategy,
                      devices=devices_strategy,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=(cfg.general.name == 'debug'),
                      enable_progress_bar=cfg.general.progress_bar,
                      callbacks=callbacks,
                      log_every_n_steps=cfg.general.log_every_steps if name != 'debug' else 1,
                      logger=logger_list)

    # --- Training/Testing Execution ---
    if not cfg.general.test_only:
        print(f"Starting training for {cfg.general.name}...")
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume if model_to_load else None)

        if cfg.general.name not in ['debug', 'test'] and cfg.general.run_test_after_train:
            print("Training finished. Starting testing using best checkpoint...")
            trainer.test(model, datamodule=datamodule, ckpt_path='best')
        else:
            print("Training finished. Skipping final test run based on config.")
    else:
        print(f"Starting testing for checkpoint: {cfg.general.test_only}")
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)

        if cfg.general.evaluate_all_checkpoints:
            try:
                directory = pathlib.Path(cfg.general.test_only).parent
                if os.path.isdir(directory):
                    print(f"Evaluating all checkpoints in directory: {directory}")
                    ckpt_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.ckpt')])
                    for ckpt_path in ckpt_files:
                        if ckpt_path == cfg.general.test_only: continue
                        print(f"\n--- Loading and testing checkpoint: {os.path.basename(ckpt_path)} ---")
                        # Create a new model instance for each checkpoint test
                        if cfg.model.type != 'discrete':
                             raise ValueError("evaluate_all_checkpoints assumes discrete model for AIG")
                        test_model = DiscreteDenoisingDiffusion.load_from_checkpoint(ckpt_path, **model_kwargs)
                        trainer.test(test_model, datamodule=datamodule, ckpt_path=None)
                else:
                     print(f"Warning: Directory for evaluating all checkpoints not found: {directory}")
            except Exception as e:
                 print(f"Error during evaluation of all checkpoints: {e}")

    # --- Clean up Wandb ---
    if cfg.general.wandb != 'disabled' and 'wandb_logger' in locals():
         wandb.finish()

if __name__ == '__main__':
    main()

