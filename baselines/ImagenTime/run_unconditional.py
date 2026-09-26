import os, sys
import torch
import numpy as np
import torch.multiprocessing
import logging
from tqdm import tqdm
from metrics import evaluate_model_uncond
from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from models.model import ImagenTime
from models.sampler import DiffusionProcess
from utils.utils import save_checkpoint, restore_state, create_model_name_and_dir, print_model_params, \
    log_config_and_tags
from utils.utils_data import gen_dataloader, denormalize_npy, get_output_paths
from utils.utils_args import parse_args_uncond

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


def generate_samples(model, args, num_samples, batch_size):
    """Sample in image space, convert back to time series."""
    generated = []
    model.eval()
    with torch.no_grad():
        with model.ema_scope():
            process = DiffusionProcess(
                args, model.net,
                (args.input_channels, args.img_resolution, args.img_resolution),
            )
            remaining = num_samples
            while remaining > 0:
                current_batch = min(batch_size, remaining)
                x_img_sampled = process.sampling(sampling_number=current_batch)
                x_ts = model.img_to_ts(x_img_sampled)
                generated.append(x_ts.detach().cpu().numpy())
                remaining -= current_batch
            del process
    return np.vstack(generated)


def save_dataset_outputs(args, generated_norm):
    origin_data = args.raw_data.astype(np.float32)
    generated_data = denormalize_npy(
        generated_norm, args.data_min, args.data_max,
    ).astype(np.float32)

    if origin_data.shape != generated_data.shape:
        raise ValueError(
            f'Generated shape {generated_data.shape} does not match origin shape {origin_data.shape}'
        )

    if getattr(args, 'generated_path', None):
        # benchmark layout (baselines/common.py): only the generated set is written
        generated_path = args.generated_path
        np.save(generated_path, generated_data)
    else:
        origin_path, generated_path = get_output_paths(args)
        np.save(origin_path, origin_data)
        np.save(generated_path, generated_data)
        logging.info(f'Saved origin data to {origin_path} with shape {origin_data.shape}')
    logging.info(f'Saved generated data to {generated_path} with shape {generated_data.shape}')


def resolve_benchmark_paths(args):
    """Take data_path and the output file from baselines/common.py when
    `benchmark_dataset` is set (config key or --benchmark_dataset)."""
    ds = getattr(args, 'benchmark_dataset', None)
    if not ds:
        return
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # baselines/
    import common
    args.data_path = str(common.data_path(ds))
    args.generated_path = str(common.generated_path('ImagenTime', ds, args.seed))


def run_eval_loop(model, args, test_loader, logger, epoch, state):
    gen_sig = []
    real_sig = []
    model.eval()
    with torch.no_grad():
        with model.ema_scope():
            process = DiffusionProcess(
                args, model.net,
                (args.input_channels, args.img_resolution, args.img_resolution),
            )
            for data in tqdm(test_loader):
                x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                x_ts = model.img_to_ts(x_img_sampled)

                if args.dataset in ['temperature_rain']:
                    x_ts = torch.clamp(x_ts, 0, 1)

                gen_sig.append(x_ts.detach().cpu().numpy())
                real_sig.append(data[0].detach().cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)
    scores = evaluate_model_uncond(real_sig, gen_sig, args)
    for key, value in scores.items():
        logger.log(f'test/{key}', value, epoch)

    curr_score = scores['marginal_score_mean'] if 'marginal_score_mean' in scores else scores['disc_mean']
    return curr_score


def main(args):
    # model name and directory
    name = create_model_name_and_dir(args)

    # log args
    logging.info(args)

    # set-up neptune logger. switch to your desired logger
    with CompositeLogger([NeptuneLogger()]) if args.neptune \
            else PrintLogger() as logger:

        # log config and tags
        log_config_and_tags(args, logger, name)

        # set-up data and device
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader = gen_dataloader(args)
        logging.info(args.dataset + ' dataset is ready.')

        model = ImagenTime(args=args, device=args.device).to(args.device)
        if args.use_stft:
            model.init_stft_embedder(train_loader)

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        state = dict(model=model, epoch=0)
        init_epoch = 0

        # restore checkpoint
        if args.resume:
            ema_model = model.model_ema if args.ema else None # load ema model if available
            init_epoch = restore_state(args, state, ema_model=ema_model)

        # print model parameters
        print_model_params(logger, model)

        eval_during_training = getattr(args, 'eval_during_training', True)
        best_score = float('inf')

        if eval_during_training:
            logging.info(f"Continuing training loop from epoch {init_epoch} (with mid-training eval).")
        else:
            logging.info(f"Training for {args.epochs} epochs (no mid-training evaluation).")

        # --- train model ---
        for epoch in range(init_epoch, args.epochs):
            model.train()
            model.epoch = epoch
            logger.log_name_params('train/epoch', epoch)

            # --- train loop ---
            for i, data in enumerate(train_loader, 1):
                x_ts = data[0].to(args.device)
                x_img = model.ts_to_img(x_ts)
                optimizer.zero_grad()
                loss = model.loss_fn(x_img)
                if len(loss) == 2:
                    loss, to_log = loss
                    for key, value in to_log.items():
                        logger.log(f'train/{key}', value, epoch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                model.on_train_batch_end()

            state['epoch'] = epoch + 1

            if eval_during_training:
                is_last_epoch = epoch == args.epochs - 1
                should_eval = (epoch > 0 and epoch % args.logging_iter == 0) or is_last_epoch
                if should_eval:
                    curr_score = run_eval_loop(model, args, test_loader, logger, epoch, state)
                    if curr_score < best_score:
                        best_score = curr_score
                        ema_model = model.model_ema if args.ema else None
                        save_checkpoint(args.log_dir, state, epoch, ema_model)
            else:
                ema_model = model.model_ema if args.ema else None
                save_checkpoint(args.log_dir, state, epoch + 1, ema_model)

        logging.info("Training is complete")

        if hasattr(args, 'raw_data') and getattr(args, 'num_samples', None):
            if eval_during_training and best_score < float('inf'):
                ema_model = model.model_ema if args.ema else None
                restore_state(args, state, ema_model=ema_model)
                logging.info('Restored best checkpoint before post-training generation.')
            gen_batch_size = getattr(args, 'gen_batch_size', None) or args.batch_size
            logging.info(
                f'Generating {args.num_samples} samples from final epoch model '
                f'(batch_size={gen_batch_size})...'
            )
            generated_norm = generate_samples(model, args, args.num_samples, gen_batch_size)
            save_dataset_outputs(args, generated_norm)


if __name__ == '__main__':
    args = parse_args_uncond()  # parse unconditional generation specific args
    resolve_benchmark_paths(args)
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
