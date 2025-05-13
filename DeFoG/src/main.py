import os
import os.path as osp  # osp is os.path
import pathlib
import warnings

# import graph_tool # graph-tool might not be needed directly in main
import torch

torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

# DeFoG specific imports
from src import utils  # Assuming utils.py is in src
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from src.graph_discrete_flow_model import GraphDiscreteFlowModel
from src.models.extra_features import DummyExtraFeatures, ExtraFeatures

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume_config_for_test(cfg: DictConfig) -> DictConfig:
    """Loads the config from a previous run for test_only mode, allowing current cfg to override some test params."""
    if not cfg.general.test_only:
        return cfg

    # Correctly determine the path to the .hydra directory relative to the checkpoint file
    # Checkpoint is in <output_dir>/checkpoints/<run_name>/epoch=X.ckpt
    # .hydra is in <output_dir>/.hydra/
    checkpoint_path = pathlib.Path(cfg.general.test_only)
    # Assuming <output_dir> is two levels above the specific checkpoint file's directory if run_name is used in dirpath
    # Or one level if dirpath is just 'checkpoints/'
    if checkpoint_path.parent.name == cfg.general.name and checkpoint_path.parent.parent.name == "checkpoints":
        output_dir = checkpoint_path.parents[2]
    elif checkpoint_path.parent.name == "checkpoints":
        output_dir = checkpoint_path.parents[1]
    else:
        warnings.warn(
            f"Could not reliably determine output directory from checkpoint path: {cfg.general.test_only}. Using current config.")
        return cfg

    saved_cfg_path = output_dir / '.hydra' / 'config.yaml'

    if not os.path.exists(saved_cfg_path):
        warnings.warn(f"Saved config not found at {saved_cfg_path}. Using current config for testing.")
        return cfg

    print(f"Loading saved config from {saved_cfg_path} for test_only run.")
    saved_cfg = OmegaConf.load(saved_cfg_path)

    # Preserve original model and dataset config from the saved configuration
    model_arch_cfg = saved_cfg.model
    dataset_original_cfg = saved_cfg.dataset

    # Start with a copy of the current config (which might have specific test parameters)
    merged_cfg = cfg.copy()

    # Overlay the saved model and dataset configurations to ensure consistency
    merged_cfg.model = model_arch_cfg
    merged_cfg.dataset = dataset_original_cfg

    # Update specific general fields from the new (current) cfg for the test run
    merged_cfg.general.test_only = cfg.general.test_only  # The path to the checkpoint itself
    merged_cfg.general.resume = None  # Explicitly set to None in test_only mode
    # Keep the original run name in cfg.general.name for checkpoint path construction, but can log with a modified name
    merged_cfg.general.name_override_for_test_log = saved_cfg.general.name + "_test"
    merged_cfg.general.evaluate_all_checkpoints = cfg.general.evaluate_all_checkpoints

    # Carry over current test-specific sample generation numbers
    if hasattr(cfg.general, 'samples_to_generate'):
        merged_cfg.general.samples_to_generate = cfg.general.samples_to_generate
    if hasattr(cfg.general, 'final_model_samples_to_generate'):
        merged_cfg.general.final_model_samples_to_generate = cfg.general.final_model_samples_to_generate
    if hasattr(cfg.general, 'abs_path_to_project_root'):  # Ensure project root is consistent
        merged_cfg.general.abs_path_to_project_root = cfg.general.abs_path_to_project_root

    print(f"Resumed with config for testing. Original run name: {saved_cfg.general.name}")
    return merged_cfg


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Store original CWD because Hydra changes it
    original_cwd = os.getcwd()
    print(f"Original CWD: {original_cwd}")

    if hasattr(cfg.general, 'abs_path_to_project_root') and cfg.general.abs_path_to_project_root is not None:
        # If abs_path_to_project_root is set, we assume it's the true root.
        # DeFoG's main.py expects to be run from src, and configs are ../configs.
        # If user provides abs_path_to_project_root, we might not need to chdir if paths in configs are relative to it.
        # For now, let's keep the chdir behavior if test_only for loading relative paths from old run.
        # And ensure dataset paths are constructed correctly.
        pass  # Handled later or assumed paths are set correctly in hydra configs
    else:
        # Infer project root: assumes main.py is in src/ and project root is parent of src/
        cfg.general.abs_path_to_project_root = str(pathlib.Path(os.path.realpath(__file__)).parents[1])
        print(f"Project root inferred as: {cfg.general.abs_path_to_project_root}")

    if cfg.general.test_only:
        cfg = get_resume_config_for_test(cfg)
        # For test_only, set CWD to the original experiment's output directory
        # This helps if the loaded config has relative paths for other resources.
        # The checkpoint_path.parents[1] should point to the <output_dir>
        # (e.g., outputs/YYYY-MM-DD/HH-MM-SS-run_name)
        checkpoint_path = pathlib.Path(cfg.general.test_only)
        if checkpoint_path.parent.name == cfg.general.name and checkpoint_path.parent.parent.name == "checkpoints":
            output_dir_for_test = checkpoint_path.parents[2]
        elif checkpoint_path.parent.name == "checkpoints":
            output_dir_for_test = checkpoint_path.parents[1]
        else:  # Fallback or warning
            output_dir_for_test = pathlib.Path(original_cwd)  # Default to where script was launched if path unclear
            warnings.warn(
                f"Could not determine original output directory for test_only. Using CWD: {output_dir_for_test}")

        if os.path.exists(output_dir_for_test):
            os.chdir(output_dir_for_test)
            print(f"Changed CWD for test_only to: {output_dir_for_test}")
        else:
            warnings.warn(f"Original output directory {output_dir_for_test} does not exist. CWD not changed.")

    pl.seed_everything(cfg.train.seed)
    dataset_config = cfg.dataset
    model_config = cfg.model

    # Common initializations
    train_metrics = None
    sampling_metrics = None
    visualization_tools = None
    extra_features = None
    domain_features = None
    datamodule = None
    dataset_infos = None

    if dataset_config.name in ["sbm", "comm20", "planar", "tree"]:
        from src.datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from src.analysis.visualization import NonMolecularVisualization
        from src.analysis.spectre_utils import (
            PlanarSamplingMetrics, SBMSamplingMetrics,
            Comm20SamplingMetrics, TreeSamplingMetrics
        )

        datamodule = SpectreGraphDataModule(cfg)
        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        if dataset_config.name == "sbm":
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config.name == "comm20":
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        elif dataset_config.name == "planar":
            sampling_metrics = PlanarSamplingMetrics(datamodule)
        elif dataset_config.name == "tree":
            sampling_metrics = TreeSamplingMetrics(datamodule)

        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.get("rrwp_steps", 16),
            dataset_info=dataset_infos,
        )
        domain_features = DummyExtraFeatures()

    elif dataset_config.name == "aig":
        from src.datasets.aig_custom_dataset import AIGDataModule, \
            AIGDatasetInfos  # Ensure this is your final AIG dataset script name
        from src.analysis.aig_metrics import AIGSamplingMetrics  # You need to create this
        from src.analysis.visualization import NonMolecularVisualization

        datamodule = AIGDataModule(cfg)
        dataset_infos = AIGDatasetInfos(datamodule=datamodule, dataset_config=dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)
        # Ensure AIGSamplingMetrics can be instantiated correctly. Pass cfg if it needs it.
        sampling_metrics = AIGSamplingMetrics(datamodule=datamodule, cfg=cfg)

        if model_config.get("extra_features") and model_config.extra_features.lower() not in ["null", "none"]:
            extra_features = ExtraFeatures(
                model_config.extra_features,
                model_config.get("rrwp_steps", 16),  # Default if not present
                dataset_info=dataset_infos,
            )
            print(f"Using ExtraFeatures for AIG: {model_config.extra_features}")
        else:
            extra_features = DummyExtraFeatures()
            print("Using DummyExtraFeatures for AIG.")
        domain_features = DummyExtraFeatures()


    # elif dataset_config.name in ["qm9", "guacamol", "moses"]:
    #     from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
    #     from src.metrics.molecular_metrics import SamplingMolecularMetrics
    #     from src.models.extra_features_molecular import ExtraMolecularFeatures
    #     from src.analysis.visualization import MolecularVisualization
    #
    #     dataset_smiles = None
    #     if "qm9" in dataset_config.name:
    #         from src.datasets import qm9_dataset
    #         datamodule = qm9_dataset.QM9DataModule(cfg)
    #         dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
    #         dataset_smiles = qm9_dataset.get_smiles(
    #             cfg=cfg, datamodule=datamodule, dataset_infos=dataset_infos, evaluate_datasets=False
    #         )
    #     elif dataset_config.name == "guacamol":
    #         from src.datasets import guacamol_dataset
    #         datamodule = guacamol_dataset.GuacamolDataModule(cfg)
    #         dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)  # Pass cfg
    #         dataset_smiles = guacamol_dataset.get_smiles(
    #             raw_dir=datamodule.train_dataset.raw_dir, filter_dataset=cfg.dataset.filter,
    #         )
    #     elif dataset_config.name == "moses":  # Changed from dataset_config.name to name
    #         from src.datasets import moses_dataset
    #         datamodule = moses_dataset.MosesDataModule(cfg)
    #         dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)  # Pass cfg
    #         dataset_smiles = moses_dataset.get_smiles(
    #             raw_dir=datamodule.train_dataset.raw_dir, filter_dataset=cfg.dataset.filter,
    #         )
    #     else:
    #         raise ValueError(f"Molecular dataset {dataset_config.name} handling error.")
    #
    #     train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    #     sampling_metrics = SamplingMolecularMetrics(
    #         dataset_infos, dataset_smiles, cfg, add_virtual_states=("absorbing" == cfg.model.transition)
    #     )
    #     visualization_tools = MolecularVisualization(
    #         cfg.dataset.get('remove_h', False), dataset_infos=dataset_infos
    #     )
    #     extra_features = ExtraFeatures(
    #         cfg.model.extra_features,
    #         cfg.model.get("rrwp_steps", 16),
    #         dataset_info=dataset_infos,
    #     )
    #     domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

    # elif dataset_config.name == "tls":
    #     from src.datasets import tls_dataset
    #     from src.metrics.tls_metrics import TLSSamplingMetrics
    #     from src.analysis.visualization import NonMolecularVisualization
    #
    #     datamodule = tls_dataset.TLSDataModule(cfg)
    #     dataset_infos = tls_dataset.TLSInfos(datamodule=datamodule)
    #     train_metrics = TrainAbstractMetricsDiscrete()
    #     if model_config.get("extra_features") and model_config.extra_features.lower() not in ["null", "none"]:
    #         extra_features = ExtraFeatures(
    #             model_config.extra_features,
    #             model_config.get("rrwp_steps", 16),
    #             dataset_info=dataset_infos,
    #         )
    #     else:
    #         extra_features = DummyExtraFeatures()
    #     domain_features = DummyExtraFeatures()
    #     sampling_metrics = TLSSamplingMetrics(datamodule)
    #     visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)
    else:
        raise NotImplementedError(f"Dataset '{dataset_config.name}' is not implemented in main.py.")

    # Common operations for all datasets
    if dataset_infos is None:
        raise ValueError("DatasetInfos was not initialized. Check dataset name and logic.")

    # compute_input_output_dims is now part of AIGDatasetInfos constructor if dataset_config["name"] == 'aig'
    # For other datasets, it's called here.
    if dataset_config.name != 'aig':
        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features
        )

    if hasattr(dataset_infos, 'compute_reference_metrics') and sampling_metrics is not None:
        dataset_infos.compute_reference_metrics(
            datamodule=datamodule,
            sampling_metrics=sampling_metrics,
        )

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
        "extra_features": extra_features,
        "domain_features": domain_features,
        "test_labels": (
            getattr(datamodule, 'test_labels', None)  # Use getattr for safety
            if ("qm9" in cfg.dataset.name and cfg.general.get('conditional', False))
            else None
        ),
    }

    # Model instantiation / Resuming
    model = None
    # cfg.general.name is used for checkpoint paths, so use the original name for resuming.
    # The Hydra output directory will still be unique for the new (resumed) run.
    run_name_for_checkpoints = cfg.general.name_override_for_test_log if cfg.general.test_only and hasattr(cfg.general,
                                                                                                           'name_override_for_test_log') else cfg.general.name

    if cfg.general.test_only:
        print(f"Loading model from checkpoint for testing: {cfg.general.test_only}")
        model = GraphDiscreteFlowModel.load_from_checkpoint(cfg.general.test_only, cfg=cfg, strict=False,
                                                            **model_kwargs)
        # CWD should already be changed if get_resume_config_for_test was effective
    elif cfg.general.resume is not None:
        print(f"Resuming training from checkpoint: {cfg.general.resume}")
        # DeFoG's typical resume loads weights and uses current cfg for new training phase.
        # If cfg.general.resume is relative, it should be relative to the *new* Hydra output dir.
        # Or, it should be an absolute path.
        # For simplicity, we assume it's a path that load_from_checkpoint can find.
        # If the CWD logic is complex (e.g., always relative to original_cwd), adjust path here.
        resume_path = cfg.general.resume
        if not osp.isabs(resume_path) and original_cwd != os.getcwd():  # if relative and CWD changed by Hydra
            # This might be needed if resume path was relative to where script was launched,
            # but Hydra changed CWD. A common pattern is for resume to be absolute or relative
            # to the *new* run's output directory (e.g., if copied there).
            # For now, assume it's findable as is by PyTorch Lightning.
            pass

        model = GraphDiscreteFlowModel.load_from_checkpoint(resume_path, cfg=cfg, strict=False, **model_kwargs)
        print(f"Model loaded for resume. Current working directory: {os.getcwd()}")

    if model is None:
        print("Creating a new GraphDiscreteFlowModel instance.")
        model = GraphDiscreteFlowModel(cfg=cfg, **model_kwargs)

    # Create folders in the current CWD (which is the Hydra output directory)
    # utils.create_folders(cfg) # This function in DeFoG might expect cfg.general.name without "_test" suffix
    # Use a consistent name for folder creation, or ensure create_folders handles it.
    temp_cfg_for_folders = cfg.copy()
    temp_cfg_for_folders.general.name = run_name_for_checkpoints  # Use consistent name for folders
    utils.create_folders(temp_cfg_for_folders)

    callbacks = []
    if cfg.train.save_model:
        # Dirpath should be relative to the current Hydra output directory
        checkpoint_dir = f"checkpoints/{run_name_for_checkpoints}"
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}",
            save_top_k=-1,  # DeFoG's behavior: save all periodic checkpoints
            every_n_epochs=(cfg.general.get('sample_every_val', 1) * cfg.general.check_val_every_n_epochs),
            save_last=True  # Saves last.ckpt
        )
        callbacks.append(checkpoint_callback)

    if cfg.train.get('ema_decay', 0) > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    # Use a logging name that reflects testing or original run name
    effective_run_name = cfg.general.name_override_for_test_log if cfg.general.test_only and hasattr(cfg.general,
                                                                                                     'name_override_for_test_log') else cfg.general.name
    if effective_run_name == "debug":  # Use the original run name from cfg, not the potentially modified one
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run.")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer_strategy_cfg = "ddp_find_unused_parameters_true" if use_gpu and cfg.general.gpus > 1 else None

    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy=trainer_strategy_cfg,
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=(effective_run_name == "debug"),  # Use effective_run_name for this check
        enable_progress_bar=cfg.train.get('progress_bar', False),
        callbacks=callbacks,
        log_every_n_steps=cfg.general.get('log_every_steps', 50) if effective_run_name != "debug" else 1,
        logger=[]  # DeFoG uses utils.setup_wandb called in GraphDiscreteFlowModel
    )

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume if cfg.general.resume else None)
        if effective_run_name not in ['debug', 'test'] and not cfg.general.get('skip_final_test',
                                                                               False):  # Use effective_run_name
            print("Training finished. Starting final testing using the last checkpoint...")
            # Ensure checkpoint_dir is correct
            # The trainer.checkpoint_callback might not be set if no checkpointing was done (e.g. debug)
            last_ckpt_path = osp.join(checkpoint_dir, "last.ckpt") if 'checkpoint_callback' in locals() else None

            if last_ckpt_path and osp.exists(last_ckpt_path):
                print(f"Loading last model checkpoint for final test: {last_ckpt_path}")
                test_model = GraphDiscreteFlowModel.load_from_checkpoint(last_ckpt_path, cfg=cfg, **model_kwargs)
                trainer.test(test_model, datamodule=datamodule)
            else:
                warnings.warn(
                    f"No 'last.ckpt' found at {last_ckpt_path if last_ckpt_path else 'N/A'} or checkpoint_callback not used. Testing with the final model state from training.")
                trainer.test(model, datamodule=datamodule)
    else:
        print(f"Starting testing with loaded model from: {cfg.general.test_only}")
        trainer.test(model, datamodule=datamodule)

        if cfg.general.evaluate_all_checkpoints:
            print("Evaluating all checkpoints in the directory...")
            # CWD is already the original experiment's output dir
            ckpt_eval_dir = pathlib.Path(
                "checkpoints") / saved_cfg.general.name  # Use saved_cfg's name to find original dir
            if not ckpt_eval_dir.exists():
                warnings.warn(f"Checkpoint directory {ckpt_eval_dir} for 'evaluate_all_checkpoints' does not exist.")
            else:
                for file_name in sorted(os.listdir(ckpt_eval_dir)):
                    if file_name.endswith(".ckpt") and file_name != "last.ckpt":  # Avoid re-testing 'last'
                        ckpt_path_eval = osp.join(ckpt_eval_dir, file_name)
                        if ckpt_path_eval == cfg.general.test_only:  # Already tested this one
                            continue
                        print(f"Loading checkpoint for evaluation: {ckpt_path_eval}")
                        eval_model = GraphDiscreteFlowModel.load_from_checkpoint(ckpt_path_eval, cfg=cfg,
                                                                                 **model_kwargs)
                        trainer.test(eval_model, datamodule=datamodule)


if __name__ == "__main__":
    # This setup helps if you run `python src/main.py ...` directly
    # and your custom modules are in `src/`
    current_file_dir = pathlib.Path(__file__).parent.resolve()
    project_root_dir = current_file_dir.parent  # Assumes main.py is in src/
    if str(project_root_dir) not in os.sys.path:
        os.sys.path.insert(0, str(project_root_dir))
        print(f"Added project root to sys.path for script execution: {project_root_dir}")
    main()