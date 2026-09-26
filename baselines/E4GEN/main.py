import os

import torch
import argparse
import numpy as np

from engine.logger import Logger, DecompTrace, save_trace_npz
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, unnormalize_sample
from Utils.io_utils import load_yaml_config, seed_everything, instantiate_from_config, checkpoint_run_dir
from Models.extreme.extreme_predictor_solver import run_stage2_train_extreme
from Models.extreme.extreme_controlnet_solver import run_stage3_train_controlnet
from Models.extreme.extreme_validation import run_extreme_validation_entry


def parse_args():
    parser = argparse.ArgumentParser("Diffusion-TS-Change")

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--sample", action="store_true")
    mode.add_argument("--train_extreme", action="store_true", help="Train extreme predictor (stage-2) and save checkpoint.")
    mode.add_argument("--train_controlnet", action="store_true", help="Train ControlNet (stage-3) for extreme-conditioned generation.")
    mode.add_argument("--validate_extreme", action="store_true", help="Load diffusion + predictor from checkpoint and run validation only.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--heavy",
        action="store_true",
        help="Enable heavy-tailed (Student-t) noise for diffusion.",
    )
    parser.add_argument(
        "--var_loss_weight",
        type=float,
        default=None,
        help="Override config model.params.var_loss_weight (weight for variance-matching loss).",
    )
    parser.add_argument(
        "--extreme_milestone",
        type=int,
        default=10,
        help="Diffusion checkpoint milestone to load for --train_extreme / --validate_extreme (default: 10).",
    )
    parser.add_argument(
        "--use_controlnet",
        action="store_true",
        help="Use ControlNet for extreme-conditioned sampling (requires predictor and controlnet checkpoints).",
    )
    parser.add_argument(
        "--t_min",
        type=int,
        default=None,
        help="Override extreme timestep window lower bound (for --train_extreme / --validate_extreme / --train_controlnet / --sample with ControlNet).",
    )
    parser.add_argument(
        "--t_max",
        type=int,
        default=None,
        help="Override extreme timestep window upper bound (for --train_extreme / --validate_extreme / --train_controlnet / --sample with ControlNet).",
    )

    args = parser.parse_args()

    args.device = "cuda"
    args.output_dir = os.path.join("OUTPUT", args.name)

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    config = load_yaml_config(args.config_file)
    config.setdefault('model', {}).setdefault('params', {})
    config['model']['params']['mode'] = bool(args.train)
    config['model']['params']['heavy'] = bool(args.heavy)
    if args.var_loss_weight is not None:
        config['model']['params']['var_loss_weight'] = float(args.var_loss_weight)

    # For sample mode with ControlNet: enable use_controlnet before model creation
    if args.sample and (args.use_controlnet or config.get("sample", {}).get("params", {}).get("use_controlnet", False)):
        config['model']['params']['use_controlnet'] = True

    # Optional override: t_min / t_max for extreme predictor / validation / controlnet / sampling
    if args.t_min is not None or args.t_max is not None:
        config.setdefault("extreme_model", {}).setdefault("prediction", {})
        pred_cfg = config["extreme_model"]["prediction"]
        if args.t_min is not None:
            pred_cfg["t_min"] = int(args.t_min)
        if args.t_max is not None:
            pred_cfg["t_max"] = int(args.t_max)
        # Also allow sample.params override when sampling with ControlNet
        config.setdefault("sample", {}).setdefault("params", {})
        if args.t_min is not None:
            config["sample"]["params"]["t_min"] = int(args.t_min)
        if args.t_max is not None:
            config["sample"]["params"]["t_max"] = int(args.t_max)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(args.output_dir)
    logger.save_config(config)

    # .to(device) instead of upstream's .cuda(): identical on a GPU, but also runs on CPU.
    model = instantiate_from_config(config['model']).to(device)

    if args.train:
        dataloader_info = build_dataloader(config, args)
        dataloader = dataloader_info['dataloader']
        normalizer = dataloader_info['normalizer']
        trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader, logger=logger)
        trainer.train()
    elif args.train_extreme:
        run_stage2_train_extreme(config=config, args=args, diffusion_model=model, device=device, logger=logger)
    elif args.train_controlnet:
        run_stage3_train_controlnet(config=config, args=args, diffusion_model=model, device=device, logger=logger)
    elif args.validate_extreme:
        run_extreme_validation_entry(config=config, args=args, diffusion_model=model, device=device, logger=logger)
    else:
        sample_params = config["sample"]["params"]
        num_samples = sample_params["num_samples"]
        dataloader_info = build_dataloader(config, args)
        dataloader = dataloader_info["dataloader"]
        normalizer = dataloader_info["normalizer"]

        use_controlnet = args.use_controlnet or sample_params.get("use_controlnet", False)
        predictor = None
        t_min, t_max, tau = None, None, 0.5
        if use_controlnet:
            from Models.extreme.extreme_prediction import ExtremeSemanticPredictor

            base_folder = config["solver"]["results_folder"]
            seq_len = config["model"]["params"]["seq_length"]
            results_folder = checkpoint_run_dir(base_folder, args.name, seq_len)

            # Load predictor
            predictor_ckpt = results_folder / "checkpoint_predictor.pt"
            if not predictor_ckpt.exists():
                raise FileNotFoundError(
                    f"ControlNet sampling requires checkpoint_predictor.pt at {predictor_ckpt}. "
                    "Run --train_extreme first."
                )
            extreme_cfg = config.get("extreme_model", {})
            pred_cfg = extreme_cfg.get("prediction", {})
            extreme_tail = extreme_cfg.get("extreme_tail", "upper")
            # Predictor architecture follows the prediction config
            predictor = ExtremeSemanticPredictor(
                seq_length=int(config["model"]["params"]["seq_length"]),
                feature_size=int(config["model"]["params"]["feature_size"]),
                K_max=int(pred_cfg.get("K_max", 5)),
                num_timesteps=int(config["model"]["params"]["timesteps"]),
                d_model=int(pred_cfg.get("d_model", 128)),
                n_heads=int(pred_cfg.get("n_heads", 4)),
                n_layers=int(pred_cfg.get("n_layers", 2)),
                dropout=float(pred_cfg.get("dropout", 0.1)),
                context_left=int(pred_cfg.get("context_left", 0)),
                context_right=int(pred_cfg.get("context_right", 0)),
                threshold=pred_cfg.get("threshold"),
                extreme_tail=str(extreme_tail),
            ).to(device)
            predictor.load_state_dict(torch.load(str(predictor_ckpt), map_location=device)["model"], strict=True)
            predictor.eval()

            # t_min / t_max come from prediction section (override via CLI or sample.params)
            t_min = sample_params.get("t_min")
            if t_min is None:
                t_min = pred_cfg.get("t_min", 200)
            t_max = sample_params.get("t_max")
            if t_max is None:
                t_max = pred_cfg.get("t_max", 200)

            tau = sample_params.get("tau", pred_cfg.get("decoding_tau", 0.5))
            # Prefer validation-tuned tau_best if available (saved by validate_extreme)
            hyperparams_path = results_folder / "validation_hyperparams.json"
            if hyperparams_path.exists():
                import json
                with open(hyperparams_path) as f:
                    val_hyper = json.load(f)
                if "tau_best" in val_hyper:
                    tau = float(val_hyper["tau_best"])
                    if logger is not None:
                        logger.log_info("Using validation tau_best={:.4f} from {}.".format(tau, hyperparams_path))

        trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader, logger=logger)
        trainer.load("backbone", load_controlnet=use_controlnet)
        if config['sample']['params']['whe_trace']:
            trace = DecompTrace()
            trainer.ema.ema_model.set_trace(trace)

        samples = trainer.sample(
            num_samples=num_samples,
            batch_size=sample_params["batch_size"],
            use_controlnet=use_controlnet,
            predictor=predictor,
            t_min=t_min,
            t_max=t_max,
            tau=tau,
        )
        np.save(os.path.join(args.output_dir, f'generate_data_norm_{args.name}.npy'), samples)
        unnorm_samples =  unnormalize_sample(samples,normalizer)
        np.save(os.path.join(args.output_dir, f'generate_data_unnorm_{args.name}.npy'), unnorm_samples)

        if config['sample']['params']['whe_trace']:
            save_trace_npz(trace, os.path.join(args.output_dir,f"trace_{args.name}.npz"), normalizer)



if __name__ == '__main__':
    main()
