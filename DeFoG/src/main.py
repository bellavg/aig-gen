import os
import os.path as osp  # osp is an alias for os.path, used for path manipulations
import pathlib
import warnings

# import graph_tool # graph-tool might not be needed directly in main if only used in specific metrics
import torch

torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig, OmegaConf  # OmegaConf might be used if you implement advanced resume
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # --- Start: Path and Config setup ---
    original_cwd = os.getcwd()  # Store original CWD because Hydra changes it
    print(f"Original CWD: {original_cwd}")

    # Infer project root if not specified, assuming main.py is in src/
    if not hasattr(cfg.general, 'abs_path_to_project_root') or cfg.general.abs_path_to_project_root is None:
        cfg.general.abs_path_to_project_root = str(pathlib.Path(os.path.realpath(__file__)).parents[1])
        print(f"Project root inferred as: {cfg.general.abs_path_to_project_root}")

    # If test_only, load saved config and potentially change CWD
    # This simple version just uses the current cfg if test_only is set,
    # relying on checkpoint path being absolute or resolvable.
    # For more robust test_only config loading, a function like get_resume_config_for_test would be used.
    if cfg.general.test_only and osp.exists(cfg.general.test_only):
        # If you need to load the original config for testing, a helper function would be needed here.
        # For now, DeFoG's original test_only logic directly loads the model with current cfg.
        # If Hydra changes CWD to the output dir, test_only path in config might need to be relative to that.
        # A common practice is to chdir to the checkpoint's parent experiment directory.
        try:
            # Example: checkpoint_path.parents[1] if dirpath=checkpoints/<run_name>
            # checkpoint_path.parents[2] if dirpath=checkpoints (and Hydra makes <run_name> folder)
            # This needs to align with how ModelCheckpoint's dirpath is structured relative to Hydra's run dir.
            # Assuming checkpoint is in <hydra_run_dir>/checkpoints/<cfg.general.name>/<epoch_file>.ckpt
            output_dir_for_test = pathlib.Path(cfg.general.test_only).parents[2]
            if output_dir_for_test.exists():  # output_dir_for_test is <hydra_run_dir>
                os.chdir(output_dir_for_test)
                print(f"Changed CWD for test_only to: {output_dir_for_test}")
            else:
                warnings.warn(
                    f"Original output directory {output_dir_for_test} for test_only not found. CWD not changed from {os.getcwd()}")
        except Exception as e:
            warnings.warn(f"Could not chdir for test_only: {e}. CWD remains {os.getcwd()}")

    # --- End: Path and Config setup ---

    pl.seed_everything(cfg.train.seed)
    dataset_config = cfg.dataset  # Use cfg.dataset directly as in DeFoG
    model_config = cfg.model  # Use cfg.model directly

    # Initialize common variables
    datamodule = None
    dataset_infos = None
    train_metrics = None
    sampling_metrics = None
    visualization_tools = None
    extra_features = None  # Will be initialized based on config
    domain_features = None  # Will be initialized based on config

    if dataset_config.name in ["sbm", "comm20", "planar", "tree"]:
        from src.datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from src.analysis.visualization import NonMolecularVisualization
        from src.analysis.spectre_utils import (
            PlanarSamplingMetrics, SBMSamplingMetrics,
            Comm20SamplingMetrics, TreeSamplingMetrics
        )

        datamodule = SpectreGraphDataModule(cfg)
        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)  # Pass dataset_config
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
            cfg.model.get("rrwp_steps", 12),  # Default from model/discrete.yaml
            dataset_info=dataset_infos,
        )
        domain_features = DummyExtraFeatures()

    # ---------- BEGIN AIG SPECIFIC BLOCK ----------
    elif dataset_config.name == "aig":
        # Imports for your AIG dataset
        from src.datasets.aig_custom_dataset import AIGDataModule, AIGDatasetInfos  # Assuming your file is aig_dataset.py
        from src.analysis.aig_metrics import AIGSamplingMetrics  # You need to create this file and class
        from src.analysis.visualization import NonMolecularVisualization

        print(f"Setting up AIG dataset with name: {dataset_config.name}")
        datamodule = AIGDataModule(cfg)
        # Pass dataset_config (which is cfg.dataset) to AIGDatasetInfos
        dataset_infos = AIGDatasetInfos(datamodule=datamodule, dataset_config=dataset_config)

        train_metrics = TrainAbstractMetricsDiscrete()  # Suitable for discrete graph data
        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        # Instantiate your custom AIG sampling metrics
        # Ensure AIGSamplingMetrics is defined and can be instantiated with datamodule (and cfg if needed)
        sampling_metrics = AIGSamplingMetrics(datamodule=datamodule, cfg=cfg)

        # Handle extra features based on model config, defaulting to Dummy if not specified or 'null'/'none'
        model_extra_features_type = model_config.get("extra_features")
        if model_extra_features_type and model_extra_features_type.lower() not in ["null", "none"]:
            extra_features = ExtraFeatures(
                model_extra_features_type,
                model_config.get("rrwp_steps", 12),  # Default from model/discrete.yaml
                dataset_info=dataset_infos,
            )
            print(f"Using ExtraFeatures for AIG: {model_extra_features_type}")
        else:
            extra_features = DummyExtraFeatures()
            print("Using DummyExtraFeatures for AIG.")

        domain_features = DummyExtraFeatures()  # AIGs typically don't use DeFoG's molecular domain features
    # ---------- END AIG SPECIFIC BLOCK ----------

    # elif dataset_config.name in ["qm9", "guacamol", "moses"]:
    #     from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
    #     from src.metrics.molecular_metrics import \
    #         SamplingMolecularMetrics  # TrainMolecularMetrics not used in DeFoG main
    #     from src.models.extra_features_molecular import ExtraMolecularFeatures
    #     from src.analysis.visualization import MolecularVisualization
    #
    #     dataset_smiles = None
    #     if "qm9" in dataset_config.name:
    #         from src.datasets import qm9_dataset  # DeFoG uses from datasets (plural)
    #         datamodule = qm9_dataset.QM9DataModule(cfg)
    #         dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
    #         dataset_smiles = qm9_dataset.get_smiles(
    #             cfg=cfg, datamodule=datamodule, dataset_infos=dataset_infos, evaluate_datasets=False
    #         )
    #     elif dataset_config.name == "guacamol":
    #         from src.datasets import guacamol_dataset
    #         datamodule = guacamol_dataset.GuacamolDataModule(cfg)
    #         dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
    #         dataset_smiles = guacamol_dataset.get_smiles(
    #             raw_dir=datamodule.train_dataset.raw_dir,  # Accessing train_dataset might require it to be initialized
    #             filter_dataset=cfg.dataset.get("filter", False),  # Use .get for safety
    #         )
    #     elif dataset_config.name == "moses":
    #         from src.datasets import moses_dataset
    #         datamodule = moses_dataset.MosesDataModule(cfg)
    #         dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
    #         dataset_smiles = moses_dataset.get_smiles(
    #             raw_dir=datamodule.train_dataset.raw_dir,
    #             filter_dataset=cfg.dataset.get("filter", False),
    #         )
    #     else:
    #         raise ValueError(f"Molecular dataset {dataset_config.name} not implemented correctly in main.")
    #
    #     train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    #     sampling_metrics = SamplingMolecularMetrics(
    #         dataset_infos, dataset_smiles, cfg,
    #         add_virtual_states=("absorbing" == model_config.transition)  # Use model_config
    #     )
    #     visualization_tools = MolecularVisualization(
    #         cfg.dataset.get('remove_h', False), dataset_infos=dataset_infos
    #     )
    #     extra_features = ExtraFeatures(
    #         model_config.extra_features,  # Use model_config
    #         model_config.get("rrwp_steps", 12),
    #         dataset_info=dataset_infos,
    #     )
    #     domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

    # elif dataset_config.name == "tls":
    #     from src.datasets import tls_dataset
    #     from src.metrics.tls_metrics import TLSSamplingMetrics
    #     from src.analysis.visualization import NonMolecularVisualization
    #
    #     datamodule = tls_dataset.TLSDataModule(cfg)
    #     dataset_infos = tls_dataset.TLSInfos(datamodule=datamodule)  # Pass datamodule as in DeFoG
    #     train_metrics = TrainAbstractMetricsDiscrete()
    #
    #     model_extra_features_type_tls = model_config.get("extra_features")
    #     if model_extra_features_type_tls and model_extra_features_type_tls.lower() not in ["null", "none"]:
    #         extra_features = ExtraFeatures(
    #             model_extra_features_type_tls,
    #             model_config.get("rrwp_steps", 12),
    #             dataset_info=dataset_infos,
    #         )
    #     else:
    #         extra_features = DummyExtraFeatures()
    #     domain_features = DummyExtraFeatures()
    #     sampling_metrics = TLSSamplingMetrics(datamodule)
    #     visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)
    else:
        raise NotImplementedError(f"Dataset '{dataset_config.name}' is not implemented.")

    # Common operations for all datasets
    if dataset_infos is None:  # Should have been initialized in one of the blocks
        raise ValueError(f"DatasetInfos not initialized for dataset: {dataset_config.name}")

    # compute_input_output_dims is called in AIGDatasetInfos constructor.
    # For other DeFoG datasets, it's called here.
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
            getattr(datamodule, 'test_labels', None)
            if ("qm9" in cfg.dataset.name and cfg.general.get('conditional', False))
            else None
        ),
    }

    # Model instantiation / Resuming logic (Simplified for DeFoG's direct approach)
    model = None
    run_name_for_paths = cfg.general.name_override_for_test_log if cfg.general.test_only and hasattr(cfg.general,
                                                                                                     'name_override_for_test_log') else cfg.general.name

    if cfg.general.test_only:
        # cfg should already be potentially modified by get_resume_config_for_test
        print(f"Loading model from checkpoint for testing: {cfg.general.test_only}")
        model = GraphDiscreteFlowModel.load_from_checkpoint(cfg.general.test_only, cfg=cfg, strict=False,
                                                            **model_kwargs)
        # CWD should have been changed by the logic at the start of main for test_only
    elif cfg.general.resume:
        print(f"Resuming training from checkpoint: {cfg.general.resume}")
        # Note: DeFoG's default is to use current cfg for resumed training, only weights are loaded.
        # Path for resume should be resolvable from the *new* Hydra output directory or be absolute.
        model = GraphDiscreteFlowModel.load_from_checkpoint(cfg.general.resume, cfg=cfg, strict=False, **model_kwargs)
        print(f"Model loaded for resume. Current CWD (new run output dir): {os.getcwd()}")

    if model is None:  # If not testing or resuming from a specific path
        print("Creating a new GraphDiscreteFlowModel instance.")
        model = GraphDiscreteFlowModel(cfg=cfg, **model_kwargs)

    # Create folders in the CWD (Hydra output directory)
    # Pass a config where general.name is consistent for folder creation
    cfg_for_folders = cfg.copy()
    if hasattr(cfg.general, 'name_override_for_test_log') and cfg.general.test_only:
        cfg_for_folders.general.name = cfg.general.name_override_for_test_log
    utils.create_folders(cfg_for_folders)

    callbacks = []
    if cfg.train.save_model:
        # Checkpoints are saved in <hydra_run_dir>/checkpoints/<cfg.general.name>/
        checkpoint_dir = f"checkpoints/{cfg_for_folders.general.name}"  # Use name consistent with folder creation

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}",
            save_top_k=-1,  # DeFoG's original behavior saves based on every_n_epochs
            every_n_epochs=int(cfg.general.get('sample_every_val', 1) * cfg.general.check_val_every_n_epochs),
            # Ensure it's int
            save_last=True  # Save a last.ckpt
        )
        callbacks.append(checkpoint_callback)

    if cfg.train.get('ema_decay', 0) > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    effective_run_name_for_debug_check = cfg_for_folders.general.name
    if effective_run_name_for_debug_check == "debug":
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run.")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()

    trainer_strategy = "ddp_find_unused_parameters_true" if use_gpu and cfg.general.gpus > 1 else None

    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy=trainer_strategy,
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=(effective_run_name_for_debug_check == "debug"),
        enable_progress_bar=cfg.train.get('progress_bar', False),  # Use .get
        callbacks=callbacks,
        log_every_n_steps=cfg.general.get('log_every_steps',
                                          50) if effective_run_name_for_debug_check != "debug" else 1,  # Use .get
        logger=[]  # DeFoG uses utils.setup_wandb called in model's on_fit_start
    )

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume if cfg.general.resume else None)
        # Test after training
        if effective_run_name_for_debug_check not in ['debug', 'test'] and not cfg.general.get('skip_final_test',
                                                                                               False):
            print("Training finished. Starting final testing using the last checkpoint...")
            # checkpoint_dir was defined above for ModelCheckpoint
            last_ckpt_path = osp.join(checkpoint_dir, "last.ckpt") if 'checkpoint_callback' in locals() else None

            if last_ckpt_path and osp.exists(last_ckpt_path):
                print(f"Loading last model checkpoint for final test: {last_ckpt_path}")
                # For testing, it's good practice to load into a fresh model instance if config might differ
                # However, DeFoG's load_from_checkpoint handles cfg merging if needed.
                test_model = GraphDiscreteFlowModel.load_from_checkpoint(last_ckpt_path, cfg=cfg, strict=False,
                                                                         **model_kwargs)
                trainer.test(test_model, datamodule=datamodule)
            else:
                warnings.warn(
                    f"No 'last.ckpt' found at '{last_ckpt_path if last_ckpt_path else 'N/A'}' or checkpoint_callback not used. Testing with the final model state from training.")
                trainer.test(model, datamodule=datamodule)
    else:  # Only testing
        print(f"Starting testing with loaded model from: {cfg.general.test_only}")
        # Model is already loaded if test_only
        trainer.test(model, datamodule=datamodule)

        if cfg.general.evaluate_all_checkpoints:
            print("Evaluating all checkpoints in the directory...")
            # CWD is already the original experiment's output dir.
            # Checkpoints are in <original_output_dir>/checkpoints/<original_run_name>/
            # cfg.general.name here would be the *original* run name if get_resume_config_for_test worked.
            # Let's use the name from the loaded (original) config for finding the directory.
            original_run_name = cfg.model.get('_saved_general_name',
                                              run_name_for_paths)  # Hacky, better if get_resume stores it
            if hasattr(cfg,
                       '_raw_hydra_config_') and cfg._raw_hydra_config_:  # If get_resume_config_for_test loaded and merged
                original_run_name_for_eval_all = cfg._raw_hydra_config_.general.name
            else:  # Fallback
                original_run_name_for_eval_all = run_name_for_paths

            ckpt_eval_dir = pathlib.Path("checkpoints") / original_run_name_for_eval_all

            if not ckpt_eval_dir.exists():
                warnings.warn(f"Checkpoint directory {ckpt_eval_dir} for 'evaluate_all_checkpoints' does not exist.")
            else:
                for file_name in sorted(os.listdir(ckpt_eval_dir)):
                    if file_name.endswith(".ckpt") and file_name != "last.ckpt":
                        ckpt_path_eval = osp.join(ckpt_eval_dir, file_name)
                        # Avoid re-testing the cfg.general.test_only checkpoint
                        if pathlib.Path(ckpt_path_eval).resolve() == pathlib.Path(cfg.general.test_only).resolve():
                            continue
                        print(f"Loading checkpoint for evaluation: {ckpt_path_eval}")
                        eval_model = GraphDiscreteFlowModel.load_from_checkpoint(ckpt_path_eval, cfg=cfg, strict=False,
                                                                                 **model_kwargs)
                        trainer.test(eval_model, datamodule=datamodule)


if __name__ == "__main__":
    # This ensures that 'src.' imports work when running 'python src/main.py'
    # Assumes this main.py is in 'src/' and project root is its parent.
    current_file_dir = pathlib.Path(__file__).parent.resolve()
    project_root_dir = current_file_dir.parent
    if str(project_root_dir) not in os.sys.path:
        print(f"Adding project root to sys.path: {project_root_dir}")
        os.sys.path.insert(0, str(project_root_dir))

    # This is important for Hydra to find configs relative to this file's location
    # if config_path is relative (e.g., "../configs")
    # However, Hydra's @hydra.main decorator handles CWD changes internally for config loading.
    # The os.chdir for test_only is for runtime path resolution within the loaded model/config.

    main()