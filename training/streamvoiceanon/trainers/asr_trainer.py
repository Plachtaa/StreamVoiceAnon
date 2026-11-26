import shutil
import warnings
import argparse
import torch
import os
import os.path as osp
import yaml
from tqdm import tqdm
import time
import glob
import logging
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import hydra
from omegaconf import DictConfig, OmegaConf
import random
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from dataloaders.hf_stream_dataloader import build_dataloader

warnings.simplefilter('ignore')

logger = get_logger(__name__, log_level="INFO")


class ASRTrainer:
    def __init__(self, config_path, start_epoch=0, mixed_precision=None):
        """
        Initialize the Dual AR Trainer

        Args:
            config_path: Path to the configuration file
            start_epoch: Starting epoch for training
            mixed_precision: Mixed precision training mode
        """
        self.config_path = config_path
        self.start_epoch = start_epoch
        self.mixed_precision = mixed_precision

        # Load configuration
        self.config = yaml.safe_load(open(config_path))

        # Setup logging directory
        self.log_dir = self.config['log_dir']
        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        shutil.copy(config_path, osp.join(self.log_dir, osp.basename(config_path)))

        # Setup accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
        self.accelerator = Accelerator(
            project_dir=self.log_dir,
            split_batches=True,
            kwargs_handlers=[ddp_kwargs],
            mixed_precision=mixed_precision
        )

        # Setup logging
        self._setup_logging()

        # Initialize training parameters
        self._init_training_params()

        # Initialize models and optimizers
        self._init_models()

        # Load checkpoint if available
        self._load_checkpoint()

    def _setup_logging(self):
        """Setup logging and tensorboard"""
        # Setup file logging
        file_handler = logging.FileHandler(osp.join(self.log_dir, 'train.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
        logger.logger.addHandler(file_handler)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _init_training_params(self):
        """Initialize training parameters from config"""
        self.batch_size = self.config.get('batch_size', 10)
        self.device = self.accelerator.device
        self.log_interval = self.config.get('log_interval', 10)
        self.log_prediction_interval = self.config.get('log_prediction_interval', 1000)
        self.save_interval = self.config.get('save_interval', 1000)

        self.data_params = self.config.get('data_params', None)
        self.sr = self.config['preprocess_params'].get('sr', 24000)
        self.loss_params = self.config.get('loss_params', {})
        self.hop_length = self.config['preprocess_params']['spect_params'].get('hop_length', 300)
        self.win_length = self.config['preprocess_params']['spect_params'].get('win_length', 1200)
        self.n_fft = self.config['preprocess_params']['spect_params'].get('n_fft', 2048)
        self.frame_rate = self.sr // self.hop_length

        self.preprocess_params = self.config['preprocess_params']

        # Initialize dataloader
        self.train_dataloader = build_dataloader(
            batch_size=self.batch_size,
            num_workers=0, # self.config['num_workers'],
            prefetch_factor=None, #12,
            preprocess_params=self.preprocess_params,
            epoch=self.start_epoch,
        )

        self.gradiant_accumulation_steps = self.config.get('gradiant_accumulation_steps', 1)

    def _init_models(self):
        """Initialize models and optimizers"""

        # Initialize main model
        self._init_main_model()

        # Initialize helper models
        self._init_helper_models()

        # Initialize optimizers
        self._init_optimizers()

    def _init_main_model(self):
        """Initialize the main model"""
        with self.accelerator.main_process_first():
            encoder_cfg = DictConfig(yaml.safe_load(open(self.config['encoder']['config_path'])))
            self.encoder_model = hydra.utils.instantiate(encoder_cfg).to(self.device)
            asr_head_cfg = DictConfig(yaml.safe_load(open(self.config['asr_head']['config_path'])))
            self.asr_head_model = hydra.utils.instantiate(asr_head_cfg).to(self.device)
        self.encoder_model = self.accelerator.prepare(self.encoder_model)
        self.asr_head_model = self.accelerator.prepare(self.asr_head_model)

    def _init_helper_models(self):
        with self.accelerator.main_process_first():
            self.wav2vec_model = hydra.utils.instantiate(
                OmegaConf.load(self.config["wav2vec_model"]["config_path"])
            ).to(self.device)
            self.wav2vec_model.eval()
            for param in self.wav2vec_model.parameters():
                param.requires_grad = False

            self.style_encoder = hydra.utils.instantiate(
                OmegaConf.load(self.config["style_encoder"]["config_path"])
            ).to(self.device).eval()
            self.style_encoder.load_state_dict(
                torch.load(
                    self.config["style_encoder"]["checkpoint_path"], map_location="cpu"
                ),
                strict=False,
            )
            for param in self.style_encoder.parameters():
                param.requires_grad = False
            self.style_encoder.eval()

    def _init_optimizers(self):
        """Initialize optimizers and schedulers"""
        from optimizers.default import build_optimizer
        optimizer_params = self.config['optimizer_params']
        self.optimizer, self.scheduler = build_optimizer(
            torch.nn.ModuleList([self.encoder_model, self.asr_head_model]),
            optimizer_params,
            type=optimizer_params['type'],
            lr=optimizer_params['lr']
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.scheduler = self.accelerator.prepare(self.scheduler)

    def _load_checkpoint(self):
        """Load checkpoint if available"""
        available_checkpoints = glob.glob(osp.join(self.log_dir, "ASR_epoch_*_step_*.pth"))
        if len(available_checkpoints) > 0:
            # find the checkpoint that has the highest step number
            latest_checkpoint = max(
                available_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            earliest_checkpoint = min(
                available_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            # delete the earliest checkpoint
            if (
                    earliest_checkpoint != latest_checkpoint
                    and self.accelerator.is_main_process
                    and len(available_checkpoints) > 1
            ):
                os.remove(earliest_checkpoint)
                print(f"Removed {earliest_checkpoint}")
        else:
            latest_checkpoint = self.config.get("pretrained_model", "")



        with self.accelerator.main_process_first():
            if latest_checkpoint != '':
                checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                filtered_sd, skipped_keys = self.filter_state_dict_shapes(
                    checkpoint['net'], self.encoder_model
                )
                missing_keys, unexpected_keys = self.encoder_model.load_state_dict(filtered_sd, strict=False)
                print(f"Missing keys: {missing_keys}")
                print(f"Unexpected keys: {unexpected_keys}")

                filtered_sd, skipped_keys = self.filter_state_dict_shapes(
                    checkpoint['net_asr_head'], self.asr_head_model
                )
                missing_keys, unexpected_keys = self.asr_head_model.load_state_dict(filtered_sd, strict=False)
                print(f"Missing keys: {missing_keys}")
                print(f"Unexpected keys: {unexpected_keys}")
                if not self.config['load_only_params'] and len(skipped_keys) == 0:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.iters = checkpoint['iters']
                self.start_epoch = checkpoint['epoch']
            else:
                self.iters = 0
                self.start_epoch = 0

    def filter_state_dict_shapes(self, params, model):
        model_state_dict = model.state_dict()
        filtered_state_dict = {
            k: v
            for k, v in params.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        skipped_keys = set(params.keys()) - set(filtered_state_dict.keys())
        if skipped_keys:
            print(
                f"Warning: Skipped loading some keys due to shape mismatch: {skipped_keys}"
            )
        return filtered_state_dict, skipped_keys

    def train(self):
        """Main training loop"""
        for epoch in range(self.start_epoch, self.start_epoch + 1000):
            epoch_start_time = time.time()

            try:
                self.train_dataloader.sampler.set_epoch(epoch)
            except AttributeError:
                pass

            self.encoder_model.train()
            self.asr_head_model.train()

            for i, batch in enumerate(tqdm(self.train_dataloader)):
                # Process batch
                self._process_batch(epoch, i, batch)

            # Log epoch completion
            if self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch} completed in {time.time() - epoch_start_time:.2f} seconds")

    @torch.no_grad()
    def calculate_style_vec(
        self,
        audio_16k_tensor: torch.Tensor,
        wave_lens: torch.Tensor,
    ):
        feat_list = []
        for bib in range(audio_16k_tensor.size(0)):
            feat = kaldi.fbank(
                audio_16k_tensor[bib : bib + 1, : wave_lens[bib]],
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000,
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            feat_list.append(feat)
        max_feat_len = max([feat.size(0) for feat in feat_list])
        feat_lens = (
            torch.tensor([feat.size(0) for feat in feat_list], dtype=torch.int32).to(
                audio_16k_tensor.device
            )
            // 2
        )
        feat_list = [
            F.pad(
                feat,
                (0, 0, 0, max_feat_len - feat.size(0)),
                value=float(feat.min().item()),
            )
            for feat in feat_list
        ]
        feat = torch.stack(feat_list, dim=0)
        style_vectors = self.style_encoder(feat, feat_lens)
        return style_vectors
    def _process_batch(self, epoch, i, batch):
        """Process a single batch"""
        grad_norm_g = 0.0
        # Move batch to device
        text_list = batch[0]
        batch = [b.to(self.device, non_blocking=True) if b is not None else b for b in batch[1:]]
        texts, text_lens, mels, mel_lens, langs, waves, wave_lens = batch
        B = waves.size(0)

        # Resample to 16kHz for ASR models
        waves_16k = torchaudio.functional.resample(waves, self.sr, 16000)
        wave_lengths_16k = (wave_lens.float() * 16000 / self.sr).long()

        # audio features
        wav2vec_feature = self.wav2vec_model(waves_16k, wave_lengths_16k)
        wav2vec_feature_lens = (wave_lengths_16k.float() / 320).long()
        wav2vec_mask = torch.arange(wav2vec_feature.size(2)).unsqueeze(0).to(waves.device) < wav2vec_feature_lens.unsqueeze(1)

        style_vectors = self.calculate_style_vec(waves_16k, wave_lengths_16k)

        # Forward pass and loss calculation
        with self.accelerator.autocast():
            wav2vec_feature_pred, vq_results = self.encoder_model(
                x=waves,
                x_lens=wave_lens,
                target_len=wav2vec_feature.size(-1),
                g=style_vectors.unsqueeze(2),
            )
            s2s_loss = self.asr_head_model(
                x=vq_results.latents.mT,
                x_lens=wave_lens // self.hop_length // 4,
                text=texts,
                text_lens=text_lens,
            )
            l1_loss = F.l1_loss(
                wav2vec_feature_pred.mT[wav2vec_mask],
                wav2vec_feature.mT[wav2vec_mask],
            )


            loss_gen = (
                self.loss_params['s2s_loss_weight'] * s2s_loss
                + self.loss_params['l1_loss_weight'] * l1_loss
            )

            # Backward pass
            self.accelerator.backward(loss_gen / self.gradiant_accumulation_steps)

            # Update weights if gradient accumulation steps reached
            if self.iters % self.gradiant_accumulation_steps == 0:
                grad_norm_g1 = torch.nn.utils.clip_grad_norm_(
                    self.encoder_model.parameters(), 10.0
                )
                grad_norm_g2 = torch.nn.utils.clip_grad_norm_(
                    self.asr_head_model.parameters(), 10.0
                )
                self.optimizer.step()
                self.scheduler.step(self.iters)
                self.optimizer.zero_grad()

        # log display args
        display_kwargs = {
        }

        # Log training progress
        self._log_training_progress(epoch, i, grad_norm_g, loss_gen, s2s_loss, l1_loss, **display_kwargs)

        # Save checkpoint
        self._save_checkpoint(epoch)

        # Increment iteration counter
        self.iters += 1

    def _log_training_progress(self, epoch, i, grad_norm_g, loss_gen, s2s_loss, l1_loss, **display_kwargs):
        """Log training progress to tensorboard"""
        if self.iters % self.log_interval == 0 and self.accelerator.is_main_process:
            with torch.no_grad():
                cur_lr = self.scheduler.get_last_lr()[0] if i != 0 else 0

                # Log to console
                print(
                    "Epoch %d, Iteration %d, Loss Gen: %.4f, Loss S2S: %.4f, Loss L1: %.4f, "
                    "Grad Norm G: %.4f"
                    % (
                        epoch,
                        self.iters,
                        loss_gen.item(),
                        s2s_loss.item(),
                        l1_loss.item(),
                        grad_norm_g,
                    )
                )

                # Log to tensorboard
                self.writer.add_scalar('train/lr', cur_lr, self.iters)
                self.writer.add_scalar('grad_norm/ar', grad_norm_g, self.iters)
                self.writer.add_scalar('train/s2s_loss', s2s_loss.item(), self.iters)
                self.writer.add_scalar('train/l1_loss', l1_loss.item(), self.iters)
                self.writer.add_scalar('train/loss_gen', loss_gen.item(), self.iters)

    def _save_checkpoint(self, epoch):
        """Save model checkpoint"""
        if self.iters % self.save_interval == 0 and self.accelerator.is_main_process:
            print('Saving checkpoint...')
            state = {
                'net': self.encoder_model.state_dict(),
                'net_asr_head': self.asr_head_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'iters': self.iters,
                'epoch': epoch,
            }
            save_path = osp.join(self.log_dir, 'ASR_epoch_%05d_step_%05d.pth' % (epoch, self.iters))
            torch.save(state, save_path)

            # Find all checkpoints and remove old ones
            checkpoints = glob.glob(osp.join(self.log_dir, 'ASR_epoch_*.pth'))
            if len(checkpoints) > 1:
                # Sort by step
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                # Remove all except last 1
                for cp in checkpoints[:-1]:
                    os.remove(cp)


def main(args):
    """Main entry point for training"""
    trainer = ASRTrainer(
        config_path=args.config_path,
        start_epoch=args.epoch,
        mixed_precision=args.mixed_precision
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_firefly_arvcasr_8192_delay4.yaml')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--mixed-precision', type=str, default=None)
    args = parser.parse_args()
    main(args)
