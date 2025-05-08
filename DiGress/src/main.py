import os
import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
# import tqdm # No longer used directly in main
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

# Assuming utils is in the same directory (src) or accessible via PYTHONPATH
try:
    from . import utils # Use relative import if main.py is inside src
except ImportError:
    import utils # Fallback to direct import

# Abstract Metrics are always needed
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

# Import models - Adjust path if they are elsewhere
# Assuming diffusion_model_discrete is in the same directory (src)
try:
    # from .diffusion_model import LiftedDenoisingDiffusion # Original continuous model (commented out)
    from .diffusion_model_discrete import DiscreteDenoisingDiffusion
except ImportError:
    # from diffusion_model import LiftedDenoisingDiffusion
    from diffusion_model_discrete import DiscreteDenoisingDiffusion

# Import feature extractors - Adjust path if needed
try:
    from .diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
except ImportError:
    from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures


warnings.filterwarnings("ignore", category=PossibleUserWarning)


# --- Resume Functions (Unchanged from your AIG version) ---
def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    # Assuming AIG uses the discrete model
    if cfg.model.type != 'discrete':
        print(f"WARNING: test_only expects discrete model for AIG, but found {cfg.model.type}. Attempting load anyway.")
        # Add logic for other model types if needed, otherwise raise error or default to discrete

    # Load checkpoint using the specified model type (only Discrete needed for AIG focus)
    print(f"Loading checkpoint: {resume}")
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)

    # Update config from loaded model
    loaded_cfg = model.cfg # Use the config stored in the checkpoint
    loaded_cfg.general.test_only = resume # Set the test path correctly
    loaded_cfg.general.name = name # Update name for the resumed run
    # Ensure keys from the *current* cfg override the loaded one if necessary (or update if missing)
    # Note: update_config_with_new_keys merges missing keys, might need refinement
    # if you want current cfg to always override loaded keys.
    final_cfg = utils.update_config_with_new_keys(loaded_cfg, saved_cfg) # Merge missing keys from saved_cfg to loaded_cfg
    return final_cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy() # Keep a copy of the current cfg for merging missing keys later
    current_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Try to find the project root relative to the 'outputs' directory if resuming from there
        # This assumes a standard structure where outputs is parallel to src, configs etc.
        project_root_parts = current_path.split('outputs')
        if len(project_root_parts) > 1:
             root_dir = project_root_parts[0]
        else:
             # Fallback: Assume script is run from src, project root is one level up
             root_dir = os.path.dirname(current_path)
             # Or assume project root is the current dir if neither above applies
             if not os.path.basename(root_dir): root_dir = "."

    except IndexError:
        # Fallback if 'outputs' isn't in the path (e.g., running from root)
        root_dir = "." # Assume current directory is root

    # Construct absolute path for resume
    resume_path = os.path.abspath(os.path.join(root_dir, cfg.general.resume))

    if not os.path.exists(resume_path):
        print(f"ERROR: Checkpoint path for resuming not found: {resume_path}")
        print("Starting training from scratch.")
        cfg.general.resume = None # Clear resume path in config
        return cfg, None # Return original cfg and None model

    print(f"Attempting to resume from: {resume_path}")
    try:
        # Load model (assuming discrete for AIG)
        if cfg.model.type != 'discrete':
             print(f"WARNING: resume expects discrete model for AIG, but found {cfg.model.type}. Attempting load anyway.")
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
        print("Successfully loaded model from checkpoint.")

        # Get config from the loaded model
        loaded_cfg = model.cfg

        # Adaptively update the loaded config with values from the *current* config (cfg)
        # This allows overriding parameters for the resumed run
        for category in cfg:
            if category not in loaded_cfg: loaded_cfg[category] = {} # Add category if missing
            if isinstance(cfg[category], DictConfig):
                # Iterate through args in the current config's category
                for arg in cfg[category]:
                    # Ensure the category in loaded_cfg is modifiable
                    OmegaConf.set_struct(loaded_cfg[category], True)
                    # Update or add the argument from current cfg to loaded cfg
                    loaded_cfg[category][arg] = cfg[category][arg]
                    # Restore struct flag if needed (optional)
                    OmegaConf.set_struct(loaded_cfg[category], False)
            else:
                 # If the category is not a DictConfig, just update the value directly
                 loaded_cfg[category] = cfg[category]

        # Ensure resume path and name are updated in the config being returned
        loaded_cfg.general.resume = resume_path # Store the absolute path used
        if hasattr(loaded_cfg.general, 'name') and isinstance(loaded_cfg.general.name, str) and not loaded_cfg.general.name.endswith('_resume'):
            loaded_cfg.general.name = loaded_cfg.general.name + '_resume'

        # Merge any keys that were in the *original* saved_cfg but missing in the updated loaded_cfg
        final_cfg = utils.update_config_with_new_keys(loaded_cfg, saved_cfg)

        return final_cfg, model

    except Exception as e:
        print(f"ERROR: Failed to load or process checkpoint from {resume_path}: {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed error
        print("Starting training from scratch.")
        cfg.general.resume = None # Clear resume path in config
        return cfg, None # Return original cfg and None model

# --- Main Function ---
# Restored original config path and name convention
@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]

    # Initialize variables to None
    datamodule = None
    dataset_infos = None
    train_metrics = None
    sampling_metrics = None
    visualization_tools = None
    extra_features = None
    domain_features = None
    model_kwargs = {}

    # --- Dataset Specific Setup ---
    # Using conditional logic similar to original
    if dataset_config["name"] == 'aig':
        print("Setting up AIG dataset...")
        # Import AIG specific modules
        try:
             from .datasets.aig_dataset import AIGDataModule, AIGDatasetInfos
        except ImportError:
             from datasets.aig_dataset import AIGDataModule, AIGDatasetInfos

        datamodule = AIGDataModule(cfg)
        # We need dataset_infos to determine model dims etc.
        dataset_infos = AIGDatasetInfos(datamodule, cfg) # Pass cfg here

        # Train metrics: Use abstract discrete version
        train_metrics = TrainAbstractMetricsDiscrete()

        # Sampling metrics and visualization: Not used for AIG
        sampling_metrics = None
        visualization_tools = None

        # Features: Use Dummy for AIG unless specific ones are implemented
        if cfg.model.type == 'discrete' and getattr(cfg.model, 'extra_features', None) is not None:
            # Check if 'extra_features' is set and not None/null in config
            print(f"WARNING: cfg.model.extra_features is set ({cfg.model.extra_features}), but no specific AIG feature extractor is defined. Using DummyExtraFeatures.")
            # Example placeholder if you were to implement AIG features:
            # try:
            #     from .diffusion.extra_features_aig import AigExtraFeatures
            #     extra_features = AigExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            # except ImportError:
            #     print("Could not import AigExtraFeatures. Using Dummy.")
            extra_features = DummyExtraFeatures()
        else:
            extra_features = DummyExtraFeatures() # Default to Dummy

        # Domain features are typically molecular, use Dummy for AIG
        domain_features = DummyExtraFeatures()

        # Compute input/output dimensions (needs datamodule and features)
        if dataset_infos:
            print("Computing input/output dimensions...")
            dataset_infos.compute_input_output_dims(datamodule=datamodule,
                                                    extra_features=extra_features,
                                                    domain_features=domain_features)
            print("Dimensions computed.")
        else:
            raise ValueError("AIG Dataset Infos not initialized correctly.")

        # Prepare model arguments specifically for AIG
        model_kwargs = {'dataset_infos': dataset_infos,
                        'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics,       # Pass None explicitly
                        'visualization_tools': visualization_tools, # Pass None explicitly
                        'extra_features': extra_features,
                        'domain_features': domain_features}

    # --- Add elif blocks here for other datasets from the original if needed ---
    # Example:
    # elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
    #     # ... import molecular specific modules ...
    #     # ... setup datamodule, dataset_infos, metrics, features ...
    #     # ... set model_kwargs ...
    #     pass # Replace with original logic

    else:
        raise NotImplementedError(f"Dataset {dataset_config['name']} not implemented in main.py")

    # --- Resume/Test Logic (Mostly Unchanged) ---
    model_to_load = None # Variable to hold the loaded model during resume
    if cfg.general.test_only:
        print(f"Attempting to load model for testing from: {cfg.general.test_only}")
        # Pass the correctly constructed model_kwargs
        cfg, model_to_load = get_resume(cfg, model_kwargs)
        if model_to_load is None:
            print("ERROR: Failed to load model for testing. Exiting.")
            return

        # Change directory logic (handle potential errors)
        if os.path.exists(cfg.general.test_only):
             try:
                 checkpoint_dir = os.path.dirname(os.path.abspath(cfg.general.test_only))
                 # Check if the parent is 'checkpoints' to find the run directory
                 if os.path.basename(checkpoint_dir) == 'checkpoints':
                     output_dir = os.path.dirname(checkpoint_dir)
                 else: # Assume checkpoint is directly in the run directory
                     output_dir = checkpoint_dir

                 if os.path.isdir(output_dir):
                     # Change directory to the run output directory for relative paths (logs, graphs etc.)
                     os.chdir(output_dir)
                     print(f"Changed working directory to run output directory: {output_dir}")
                 else:
                     print(f"Warning: Could not determine valid output directory from checkpoint path {cfg.general.test_only}")
             except Exception as e:
                 print(f"Warning: Error changing directory based on checkpoint path: {e}")
        else:
            print(f"ERROR: Checkpoint path for testing not found: {cfg.general.test_only}")
            return

    elif cfg.general.resume is not None:
        print(f"Attempting to resume training from: {cfg.general.resume}")
         # Pass the correctly constructed model_kwargs
        cfg, model_to_load = get_resume_adaptive(cfg, model_kwargs)
        if model_to_load is not None: # Resume successful
            if os.path.exists(cfg.general.resume):
                try:
                     # Similar logic to change directory to the run output directory
                     checkpoint_dir = os.path.dirname(os.path.abspath(cfg.general.resume))
                     if os.path.basename(checkpoint_dir) == 'checkpoints':
                          output_dir = os.path.dirname(checkpoint_dir)
                     else:
                          output_dir = checkpoint_dir

                     if os.path.isdir(output_dir):
                          os.chdir(output_dir)
                          print(f"Changed working directory to run output directory: {output_dir}")
                     else:
                          print(f"Warning: Could not determine valid output directory from resume path {cfg.general.resume}")
                except Exception as e:
                     print(f"Warning: Error changing directory based on resume path: {e}")
        # else: Resume failed, model_to_load is None, continue to train from scratch

    # --- Setup Folders and Initialize Model ---
    # Create folders in the *current working directory* (which might have been changed by resume logic)
    utils.create_folders(cfg)

    if model_to_load is None:
        print("Initializing new model instance...")
        # Ensure model type is discrete for AIG
        if cfg.model.type != 'discrete':
             raise ValueError(f"AIG dataset requires model.type='discrete', but found '{cfg.model.type}'")
        # Initialize the correct model type with the prepared kwargs
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        print("Using loaded model instance.")
        model = model_to_load # Use the model loaded during resume/test

    # --- Callbacks (Mostly Unchanged) ---
    callbacks = []
    if cfg.train.save_model:
        # Save checkpoints relative to the current working directory
        ckpt_dir = "checkpoints" # Save directly in 'checkpoints' subdirectory
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='{epoch}', # Simple filename, epoch number
            monitor='val/epoch_NLL', # Monitor validation NLL
            save_top_k=cfg.train.save_top_k if hasattr(cfg.train, 'save_top_k') else 3, # Default 3 if not set
            mode='min',
            every_n_epochs=cfg.general.save_every_n_epochs if hasattr(cfg.general, 'save_every_n_epochs') else 1, # Default 1
            save_last=True # Always save the last checkpoint
        )
        # last_ckpt_save = ModelCheckpoint(dirpath=ckpt_dir, filename='last', every_n_epochs=1) # Redundant if save_last=True
        callbacks.append(checkpoint_callback)
        # callbacks.append(last_ckpt_save)

    if hasattr(cfg.train, 'ema_decay') and cfg.train.ema_decay > 0:
        try:
            from ema import EMA # Assuming ema.py is available
            ema_callback = EMA(decay=cfg.train.ema_decay)
            callbacks.append(ema_callback)
        except ImportError:
            print("Warning: EMA callback specified but EMA class could not be imported.")

    # --- Trainer Setup (Mostly Unchanged) ---
    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run=True.")

    # Logger Setup (WandB)
    logger_list = []
    if hasattr(cfg.general, 'wandb') and cfg.general.wandb != 'disabled':
        try:
            from pytorch_lightning.loggers import WandbLogger
            import wandb

            # Attempt login only if online mode is specified
            if cfg.general.wandb == 'online':
                 # Use relogin=False to avoid prompting if already logged in
                 if not wandb.login(key=os.environ.get("WANDB_API_KEY", None), relogin=False):
                     print("Warning: Wandb is set to 'online' but login failed (maybe key missing?). Disabling Wandb.")
                     cfg.general.wandb = 'disabled' # Disable if login fails

            # Proceed if wandb is not disabled after the login check
            if cfg.general.wandb != 'disabled':
                # Ensure save_dir exists relative to current working directory
                log_save_dir = "logs"
                os.makedirs(log_save_dir, exist_ok=True)

                wandb_logger = WandbLogger(
                    name=cfg.general.name,
                    project=f'graph_ddm_{cfg.dataset.name}', # Project name uses dataset name
                    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                    save_dir=log_save_dir, # Save logs locally as well
                    mode=cfg.general.wandb # 'online', 'offline', or 'disabled'
                )
                logger_list.append(wandb_logger)
                print(f"Wandb logger initialized in '{cfg.general.wandb}' mode.")
        except ImportError:
            print("Warning: Wandb logger specified but 'wandb' or 'WandbLogger' could not be imported.")

    # Determine accelerator and devices
    use_gpu = hasattr(cfg.general, 'gpus') and cfg.general.gpus > 0 and torch.cuda.is_available()
    accelerator_strategy = 'gpu' if use_gpu else 'cpu'
    devices_strategy = cfg.general.gpus if use_gpu else 1
    # Strategy for DDP (same as your version)
    strategy = "ddp_find_unused_parameters_true" if use_gpu and cfg.general.gpus > 1 else "auto"

    # Initialize Trainer
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad if hasattr(cfg.train, 'clip_grad') else None,
        strategy=strategy,
        accelerator=accelerator_strategy,
        devices=devices_strategy,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs if hasattr(cfg.general, 'check_val_every_n_epochs') else 1,
        fast_dev_run=(cfg.general.name == 'debug'),
        enable_progress_bar=cfg.general.progress_bar if hasattr(cfg.general, 'progress_bar') else True,
        callbacks=callbacks,
        log_every_n_steps=cfg.general.log_every_steps if hasattr(cfg.general, 'log_every_steps') and name != 'debug' else 1,
        logger=logger_list # Pass the configured logger list
    )

    # --- Training/Testing Execution (Mostly Unchanged) ---
    if not cfg.general.test_only:
        print(f"Starting training for {cfg.general.name}...")
        # Pass ckpt_path only if resuming and model wasn't loaded manually already
        resume_ckpt_path = cfg.general.resume if hasattr(cfg.general, 'resume') and cfg.general.resume is not None else None
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt_path)

        # Optional: Run test after training
        run_test = getattr(cfg.general, 'run_test_after_train', False) # Default False if not set
        if cfg.general.name not in ['debug', 'test'] and run_test:
            print("Training finished. Starting testing using best checkpoint...")
            # trainer.test loads the best checkpoint automatically if ckpt_path='best'
            # Ensure a checkpoint callback is monitoring a value (like val/epoch_NLL)
            best_path = checkpoint_callback.best_model_path if 'checkpoint_callback' in locals() and checkpoint_callback.best_model_path else None
            if best_path and os.path.exists(best_path):
                 trainer.test(model, datamodule=datamodule, ckpt_path=best_path)
            elif cfg.train.save_model and os.path.exists(os.path.join(ckpt_dir, "last.ckpt")):
                 print("Best checkpoint not found, testing last checkpoint...")
                 trainer.test(model, datamodule=datamodule, ckpt_path=os.path.join(ckpt_dir, "last.ckpt"))
            else:
                 print("No best or last checkpoint found to test.")
        else:
            print("Training finished. Skipping final test run based on config.")
    else:
        # Test mode
        print(f"Starting testing for checkpoint: {cfg.general.test_only}")
        # The model was already loaded in the get_resume logic
        trainer.test(model, datamodule=datamodule, ckpt_path=None) # Pass ckpt_path=None as model is already loaded

        # Optional: Evaluate all checkpoints in the directory
        evaluate_all = getattr(cfg.general, 'evaluate_all_checkpoints', False) # Default False
        if evaluate_all:
            try:
                # Use the path from the loaded config (which should be absolute now)
                test_ckpt_path = cfg.general.test_only
                directory = pathlib.Path(test_ckpt_path).parent # Directory of the specified checkpoint
                if os.path.isdir(directory):
                    print(f"Evaluating all checkpoints in directory: {directory}")
                    ckpt_files = sorted([
                        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.ckpt')
                    ])
                    for ckpt_path in ckpt_files:
                        if ckpt_path == test_ckpt_path: continue # Skip the one already tested
                        print(f"\n--- Loading and testing checkpoint: {os.path.basename(ckpt_path)} ---")
                        # Load each checkpoint into a fresh model instance
                        if cfg.model.type != 'discrete':
                             print("WARNING: evaluate_all_checkpoints currently assumes discrete model for AIG")
                        # Pass original kwargs, they are needed for model init
                        test_model_instance = DiscreteDenoisingDiffusion.load_from_checkpoint(ckpt_path, **model_kwargs)
                        # Run test on this specific instance
                        trainer.test(test_model_instance, datamodule=datamodule, ckpt_path=None)
                else:
                     print(f"Warning: Directory for evaluating all checkpoints not found: {directory}")
            except Exception as e:
                 print(f"Error during evaluation of all checkpoints: {e}")

    # --- Clean up Wandb ---
    if 'wandb_logger' in locals() and wandb.run:
         print("Finishing Wandb run...")
         wandb.finish()

if __name__ == '__main__':
    main()