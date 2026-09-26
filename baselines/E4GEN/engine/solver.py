import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F

from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import checkpoint_run_dir, instantiate_from_config, get_model_parameters_info


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader)
        self.dataloader = dataloader
        self.step = 0
        self.milestone = 0
        self.args, self.config = args, config
        self.logger = logger

        base = config['solver']['results_folder']
        name = args.name
        seq_len = config['model']['params']['seq_length']

        self.results_folder = checkpoint_run_dir(base, name, seq_len)
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = float(config['solver'].get('base_lr', 1.0e-4))
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    BACKBONE_CKPT = "checkpoint_backbone.pt"

    def save(self, milestone, verbose=False):
        ckpt_name = self.BACKBONE_CKPT if milestone == "backbone" else f"checkpoint-{milestone}.pt"
        ckpt_path = self.results_folder / ckpt_name
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(ckpt_path)))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(ckpt_path))

    def load(self, milestone, verbose=False, load_controlnet=False):
        ckpt_name = self.BACKBONE_CKPT if milestone == "backbone" else f"checkpoint-{milestone}.pt"
        ckpt_path = self.results_folder / ckpt_name
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(ckpt_path)))
        device = self.device
        data = torch.load(str(ckpt_path), map_location=device)
        model_sd = data.get('model', data)
        self.model.load_state_dict(model_sd, strict=not load_controlnet)
        self.step = data.get('step', 0)
        if not load_controlnet:
            self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'], strict=not load_controlnet)
        self.milestone = milestone

        if load_controlnet and getattr(self.model, 'control_net', None) is not None:
            ctrl_path = self.results_folder / 'checkpoint_controlnet.pt'
            if ctrl_path.exists():
                ctrl_data = torch.load(str(ctrl_path), map_location=device)
                ctrl_sd = ctrl_data['control_net']
                self.model.control_net.load_state_dict(ctrl_sd, strict=True)
                if hasattr(self.ema, 'ema_model') and getattr(self.ema.ema_model, 'control_net', None) is not None:
                    self.ema.ema_model.control_net.load_state_dict(ctrl_sd, strict=True)
                if self.logger is not None and verbose:
                    self.logger.log_info('Loaded ControlNet from {}'.format(str(ctrl_path)))


    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name))

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    loss = self.model(data, target=data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        self.save("backbone", verbose=True)
        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, num_samples, batch_size,
               use_controlnet=False, predictor=None, t_min=None, t_max=None, tau=0.5):
        """
        Generate samples using EMA model.
        use_controlnet: if True, use ControlNet with predictor for extreme-conditioned generation.
        predictor: ExtremeSemanticPredictor; required when use_controlnet=True.
        t_min, t_max: timestep window for t_control; required when use_controlnet=True.
        tau: decoding threshold for predictor; used when use_controlnet=True.
        """
        if self.logger is not None:
            tic = time.time()
            ctrl_msg = " use_controlnet=True" if use_controlnet else ""
            self.logger.log_info(
                f"Begin sampling {num_samples} samples (batch_size={batch_size})"
                + ctrl_msg
            )

        pred = predictor if use_controlnet else None

        all_samples = []
        for b in range(0, num_samples, batch_size):
            actual_batch_size = min(batch_size, num_samples - b)
            with torch.no_grad():
                sample = self.ema.ema_model.generate_mts(
                    batch_size=actual_batch_size,
                    predictor=pred,
                    t_min=t_min,
                    t_max=t_max,
                    tau=tau,
                )
            all_samples.append(sample.detach().cpu().numpy())

        samples = np.concatenate(all_samples, axis=0)

        if self.logger is not None:
            self.logger.log_info(
                f"Sampling done, generated {samples.shape[0]} samples, "
                f"time: {time.time() - tic:.2f}s"
            )

        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples

    def n_sample(self, n, shape, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')

        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize

        # Target sample tensor shape: (n, shape[0], shape[1])
        sample_shape = (n, shape[0], shape[1])

        # Sample from pure noise; no in-fill, mask, or target
        if sampling_steps == self.model.num_timesteps:
            samples = self.ema.ema_model.sample(
                shape=sample_shape,
                model_kwargs=model_kwargs
            )
        else:
            samples = self.ema.ema_model.fast_sample(
                shape=sample_shape,
                model_kwargs=model_kwargs,
                sampling_timesteps=sampling_steps
            )

        samples = samples.detach().cpu().numpy()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))

        return samples

    def forward_sample(self, x_start):
       b, c, h = x_start.shape
       noise = torch.randn_like(x_start, device=self.device)
       t = torch.randint(0, self.model.num_timesteps, (b,), device=self.device).long()
       x_t = self.model.q_sample(x_start=x_start, t=t, noise=noise).detach()
       return x_t, t

    def train_classfier(self, classifier):
        device = self.device
        step = 0
        self.milestone_classifier = 0
        self.step_classifier = 0
        dataloader = self.dataloader
        dataloader.dataset.shift_period('test')
        dataloader = cycle(dataloader)

        self.classifier = classifier
        self.opt_classifier = Adam(filter(lambda p: p.requires_grad, self.classifier.parameters()), lr=5.0e-4)
        
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training classifier...'.format(self.args.name), check_primary=False)
        
        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    x, y = next(dataloader)
                    x, y = x.to(device), y.to(device)
                    x_t, t = self.forward_sample(x)
                    logits = classifier(x_t, t)
                    loss = F.cross_entropy(logits, y)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                self.opt_classifier.step()
                self.opt_classifier.zero_grad()
                self.step_classifier += 1
                step += 1

                with torch.no_grad():
                    if self.step_classifier != 0 and self.step_classifier % self.save_cycle == 0:
                        self.milestone_classifier += 1
                        self.save(self.milestone_classifier)
                                            
                    if self.logger is not None and self.step_classifier % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

        # return classifier

