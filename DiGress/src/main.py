#!/usr/bin/env python3
import os
import pathlib
import warnings
import pickle  # For potential use in custom sampling metrics

import torch

torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig, OmegaConf  # Added OmegaConf for merging
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

# Assuming 'src' is the root for these imports from the DiGress structure
from src import utils
from src.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from src.diffusion_model import LiftedDenoisingDiffusion
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg_path = pathlib.Path(cfg.general.test_only).parents[1] / '.hydra' / 'config.yaml'
    saved_cfg = OmegaConf.load(saved_cfg_path)

    # Update paths if the run was moved
    saved_cfg.general.test_only = cfg.general.test_only
    saved_cfg.general.resume = None  # Ensure we are in test_only mode

    # Override with any new keys from the current config (e.g., for new eval parameters)
    cfg = utils.update_config_with_new_keys(saved_cfg, cfg)  # saved_cfg is base, cfg has new keys
    cfg.general.name = cfg.general.name + '_test'

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(cfg.general.test_only, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(cfg.general.test_only, **model_kwargs)

    # model.cfg = cfg # Ensure the model uses the potentially updated cfg for testing
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split(os.sep + 'src')[0]
    resume_path = os.path.join(root_dir, cfg.general.resume)

    # Load the config from the checkpoint's directory
    saved_cfg_path = pathlib.Path(resume_path).parents[1] / '.hydra' / 'config.yaml'
    saved_cfg = OmegaConf.load(saved_cfg_path)

    # Preserve essential parameters from the saved config
    # For example, model architecture, dataset name if it's intrinsically linked to the checkpoint
    preserved_model_cfg = saved_cfg.model
    preserved_dataset_name = saved_cfg.dataset.name

    # Update the loaded config with overrides from the current config
    # This allows changing learning rate, epochs, etc.
    # new_cfg = OmegaConf.merge(saved_cfg, cfg) # cfg overrides saved_cfg
    new_cfg = cfg  # Start with the new config, and selectively pull from saved_cfg if needed

    # Ensure critical parts of the saved model's config are respected if not meant to be changed
    new_cfg.model = preserved_model_cfg
    new_cfg.dataset.name = preserved_dataset_name  # Keep original dataset name

    # Update general section carefully
    new_cfg.general.resume = resume_path  # This is the key part for resuming
    new_cfg.general.name = saved_cfg.general.name + '_resume'  # Name reflects it's a resumed run

    # Update paths and other necessary fields from the new cfg
    new_cfg.general.abs_path_to_project_root = cfg.general.abs_path_to_project_root
    new_cfg.train.n_epochs = cfg.train.n_epochs  # Allow changing n_epochs

    if new_cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, cfg=new_cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, cfg=new_cfg, **model_kwargs)

    # model.cfg = new_cfg # Ensure the model internal config is the merged/updated one
    return new_cfg, model


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    model_cfg = cfg["model"]  # Get model config for convenience

    # Common initializations
    train_metrics = TrainAbstractMetricsDiscrete() if model_cfg.type == 'discrete' else TrainAbstractMetrics()
    visualization_tools = None  # Will be set per dataset type
    extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()
    sampling_metrics = None # Initialize sampling_metrics

    if dataset_config["name"] in ['sbm', 'comm20', 'planar']:
        from src.datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from src.analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        # >>> CHANGE 1: Import AIGSamplingMetrics if it's in spectre_utils.py
        # If AIGSamplingMetrics is in a different file, adjust the import path.
        # For example, if it's in src/analysis/aig_metrics.py:
        # from src.analysis.aig_metrics import AIGSamplingMetrics
        # For this example, assuming it was added to spectre_utils.py:
        from src.analysis.aig_metrics import AIGSamplingMetrics # Ensure this is the correct path
        from src.analysis.visualization import NonMolecularVisualization

        datamodule = SpectreGraphDataModule(cfg)
        visualization_tools = NonMolecularVisualization()

        if dataset_config['name'] == 'sbm':
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config['name'] == 'comm20':
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        else:  # planar
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        if model_cfg.type == 'discrete' and model_cfg.get('extra_features') is not None:
            extra_features = ExtraFeatures(model_cfg.extra_features, dataset_info=dataset_infos)

    elif dataset_config["name"] == 'aig':
        from src.datasets.aig_custom_dataset import AIGDataModule, AIGDatasetInfos
        # >>> CHANGE 2: Import your AIGSamplingMetrics class
        # Ensure this path is correct based on where you defined AIGSamplingMetrics.
        # If it's in spectre_utils.py and already imported above, you don't need a separate import here.
        # If it's in its own file, e.g., src/analysis/aig_metrics.py:
        # from src.analysis.aig_metrics import AIGSamplingMetrics
        # Assuming it's in spectre_utils.py and imported with other spectre_utils metrics:
        from src.analysis.aig_metrics import AIGSamplingMetrics # Or from src.analysis.aig_metrics if separate
        from src.analysis.visualization import NonMolecularVisualization

        datamodule = AIGDataModule(cfg)
        dataset_infos = AIGDatasetInfos(datamodule, dataset_config)
        visualization_tools = NonMolecularVisualization()

        # >>> CHANGE 3: Instantiate AIGSamplingMetrics instead of the placeholder
        print(f"INFO: Instantiating AIGSamplingMetrics for dataset: {dataset_config['name']}")
        sampling_metrics = AIGSamplingMetrics(datamodule)

        # Extra features are not used for AIGs as per your config (model.extra_features: null)
        # extra_features and domain_features remain DummyExtraFeatures

    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
        from src.metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
        from src.analysis.visualization import MolecularVisualization
        # >>> Also ensure AIGSamplingMetrics is imported if it wasn't handled in the first block
        # from src.analysis.spectre_utils import AIGSamplingMetrics # Or from src.analysis.aig_metrics

        if dataset_config["name"] == 'qm9':
            from src.datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            from src.datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None
        elif dataset_config.name == 'moses':
            from src.datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = None
        else:
            raise ValueError(f"Dataset {dataset_config.name} not implemented in molecular block")

        visualization_tools = MolecularVisualization(cfg.dataset.get('remove_h', False), dataset_infos=dataset_infos)
        if model_cfg.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)

        if model_cfg.type == 'discrete' and model_cfg.get('extra_features') is not None:
            extra_features = ExtraFeatures(model_cfg.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        # Ensure AIGSamplingMetrics is imported if no other block handled it
        # from src.analysis.spectre_utils import AIGSamplingMetrics # Or from src.analysis.aig_metrics
        raise NotImplementedError(f"Dataset {dataset_config.name} not implemented.")

    # This call is usually done inside the DatasetInfos constructor or a dedicated method.
    # For AIGCustomDatasetInfos, it's called in its __init__.
    # For SpectreDatasetInfos and molecular ones, it's called after object creation.
    if not dataset_config["name"] == 'aig':  # AIG infos computes this in its init
        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}

    # Model instantiation / Resuming logic
    model = None  # Initialize model variable
    if cfg.general.test_only:
        cfg, model = get_resume(cfg, model_kwargs)
        # In test_only mode, chdir to the experiment's output directory for relative paths
        os.chdir(pathlib.Path(cfg.general.test_only).parents[1])  # Go up two levels from checkpoint file
    elif cfg.general.resume is not None:
        cfg, model = get_resume_adaptive(cfg, model_kwargs)
        # In resume mode, chdir to the experiment's output directory
        os.chdir(pathlib.Path(cfg.general.resume).parents[1])  # Go up two levels from checkpoint file

    # If not resuming or testing, create model from scratch
    if model is None:
        if model_cfg.type == 'discrete':
            model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
        else:
            model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    utils.create_folders(cfg)  # Creates folders in the current working directory (Hydra output dir)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints",  # Save inside the Hydra run directory
            filename='best-val_NLL-{epoch:03d}',
            monitor='val/epoch_NLL',  # Ensure this metric is logged by your model's validation_epoch_end
            save_top_k=3,
            mode='min',
            every_n_epochs=1,  # Check every epoch
            save_last=False  # We use a separate callback for 'last'
        )
        callbacks.append(checkpoint_callback)

        # Save last checkpoint based on check_val_every_n_epochs
        last_ckpt_save = ModelCheckpoint(
            dirpath=f"checkpoints",
            filename='last-{epoch:03d}',
            every_n_epochs=cfg.general.check_val_every_n_epochs,  # Save at validation frequency
            save_top_k=-1  # Keep all such checkpoints (or 1 if you only want the latest 'last')
        )
        callbacks.append(last_ckpt_save)

    if cfg.train.get('ema_decay', 0) > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()

    trainer_strategy = None
    if use_gpu:
        if cfg.general.gpus > 1:
            trainer_strategy = "ddp_find_unused_parameters_true"
        else:
            trainer_strategy = "auto"  # Or "single_device" if you know it's always one GPU
    else:  # CPU
        trainer_strategy = "auto"  # Or "single_device"

    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy=trainer_strategy,
        accelerator='gpu' if use_gpu else 'cpu',
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=cfg.general.name == 'debug',
        enable_progress_bar=cfg.train.get('progress_bar', False), # Default to False if not specified
        callbacks=callbacks,
        log_every_n_steps=cfg.general.get('log_every_steps', 50) if name != 'debug' else 1,
        logger=[]  # DiGress handles WandB logging internally if cfg.general.wandb is 'online' or 'offline'
    )

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule,
                    ckpt_path=cfg.general.resume if cfg.general.resume is not None else None)
        if cfg.general.name not in ['debug', 'test'] and not cfg.general.get('skip_final_test', False):
            print("Training finished. Starting final testing...")
            # Ensure checkpoint_callback was added and has best_model_path
            best_ckpt_path = checkpoint_callback.best_model_path if 'checkpoint_callback' in locals() and hasattr(checkpoint_callback,
                                                                            'best_model_path') and checkpoint_callback.best_model_path else None
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                print(f"Loading best model for final test: {best_ckpt_path}")
                # Need to load the model with the specific checkpoint for testing
                if model_cfg.type == 'discrete':
                    test_model = DiscreteDenoisingDiffusion.load_from_checkpoint(best_ckpt_path, cfg=cfg,
                                                                                 **model_kwargs)
                else:
                    test_model = LiftedDenoisingDiffusion.load_from_checkpoint(best_ckpt_path, cfg=cfg, **model_kwargs)
                trainer.test(test_model, datamodule=datamodule)
            else:
                print(
                    "No best model checkpoint found to test, or checkpoint_callback not used. Testing with last model state from training.")
                trainer.test(model, datamodule=datamodule)  # Test with the final state of the model from trainer.fit
    else:
        print(f"Starting testing with checkpoint: {cfg.general.test_only}")
        trainer.test(model, datamodule=datamodule)  # cfg.general.test_only is already loaded into `model` by get_resume

        if cfg.general.evaluate_all_checkpoints:
            print("Evaluating all checkpoints in the directory...")
            directory = pathlib.Path(cfg.general.test_only).parents[0]  # Checkpoints directory
            for file_name in sorted(os.listdir(directory)):  # Sort for consistent order
                if file_name.startswith('best-val_NLL') and file_name.endswith(
                        '.ckpt'):  # Example: evaluate only 'best' checkpoints
                    ckpt_path = os.path.join(directory, file_name)
                    if ckpt_path == cfg.general.test_only:  # Already tested this one
                        continue
                    print(f"Loading checkpoint for evaluation: {ckpt_path}")
                    if model_cfg.type == 'discrete':
                        eval_model = DiscreteDenoisingDiffusion.load_from_checkpoint(ckpt_path, cfg=cfg, **model_kwargs)
                    else:
                        eval_model = LiftedDenoisingDiffusion.load_from_checkpoint(ckpt_path, cfg=cfg, **model_kwargs)
                    trainer.test(eval_model, datamodule=datamodule)


if __name__ == '__main__':
    main()
