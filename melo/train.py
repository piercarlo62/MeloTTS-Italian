﻿# flake8: noqa: E402

import os
if 'MPLBACKEND' in os.environ:
    print(f"Clearing problematic MPLBACKEND={os.environ['MPLBACKEND']}")
    del os.environ['MPLBACKEND']
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
)
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols
from melo.download_utils import load_pretrain_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = (
    True  # If encontered training problem,please try to disable TF32.
)
torch.set_float32_matmul_precision("medium")


torch.backends.cudnn.benchmark = True
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(
#     True
# )  # Not available if torch version is lower than 2.0
torch.backends.cuda.enable_math_sdp(True)
global_step = 0


def run():
    import traceback
    hps = utils.get_hparams()
    if hps.log_dir is not None:
        logger = utils.get_logger(hps.log_dir)
    else:
        logger = utils.get_logger(hps.model_dir)
    logger.info("=== TRAINING STARTED ===")
    logger.info(f"Model directory: {hps.model_dir}")
    print(f"Model directory: {hps.model_dir}")
    logger.info(f"Training files: {hps.data.training_files}")
    print(f"Training files: {hps.data.training_files}")

    logger.info("Setting up distributed training...")
    print("Setting up distributed training...")
    local_rank = int(os.environ["LOCAL_RANK"])
    logger.info(f"Local rank: {local_rank}")
    print(f"Local rank: {local_rank}")

    dist.init_process_group(
        backend="gloo",
        init_method="env://",  # Due to some training problem,we proposed to use gloo instead of nccl.
        rank=local_rank,
    )  # Use torchrun instead of mp.spawn
    rank = dist.get_rank()
    n_gpus = dist.get_world_size()
    
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    global global_step
    if rank == 0:
        logger.info(f"=== TRAINING SETUP ===")
        print(f"=== TRAINING SETUP ===")
        logger.info(f"Process ID: {os.getpid()}")
        print(f"Process ID: {os.getpid()}")
        logger.info(f"Rank: {rank}, World size: {n_gpus}")
        print(f"Rank: {rank}, World size: {n_gpus}")
        logger.info(f"Initial global_step: {global_step}")
        print(f"Initial global_step: {global_step}")
        logger.info(hps)
        print(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    try:
        train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.train.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
        collate_fn = TextAudioSpeakerCollate()
        train_loader = DataLoader(
            train_dataset,
            num_workers=16,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            persistent_workers=True,
            prefetch_factor=4,
        )  # DataLoader config could be adjusted.
        if rank == 0:
            eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
            eval_loader = DataLoader(
                eval_dataset,
                num_workers=0,
                shuffle=False,
                batch_size=1,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
        if (
            "use_noise_scaled_mas" in hps.model.keys()
            and hps.model.use_noise_scaled_mas is True
        ):
            logger.info("Using noise scaled MAS for VITS2")
            print("Using noise scaled MAS for VITS2")
            mas_noise_scale_initial = 0.01
            noise_scale_delta = 2e-6
        else:
            logger.info("Using normal MAS for VITS1")
            print("Using normal MAS for VITS1")
            mas_noise_scale_initial = 0.0
            noise_scale_delta = 0.0
        if (
            "use_duration_discriminator" in hps.model.keys()
            and hps.model.use_duration_discriminator is True
        ):
            logger.info("Using duration discriminator for VITS2")
            print("Using duration discriminator for VITS2")
            net_dur_disc = DurationDiscriminator(
                hps.model.hidden_channels,
                hps.model.hidden_channels,
                3,
                0.1,
                gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
            ).cuda(rank)
        if (
            "use_spk_conditioned_encoder" in hps.model.keys()
            and hps.model.use_spk_conditioned_encoder is True
        ):
            if hps.data.n_speakers == 0:
                raise ValueError(
                    "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
                )
        else:
            logger.info("Using normal encoder for VITS1")
            print("Using normal encoder for VITS1")

        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            mas_noise_scale_initial=mas_noise_scale_initial,
            noise_scale_delta=noise_scale_delta,
            **hps.model,
        ).cuda(rank)

        net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
        optim_g = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, net_g.parameters()),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
        optim_d = torch.optim.AdamW(
            net_d.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
        if net_dur_disc is not None:
            optim_dur_disc = torch.optim.AdamW(
                net_dur_disc.parameters(),
                hps.train.learning_rate,
                betas=hps.train.betas,
                eps=hps.train.eps,
            )
        else:
            optim_dur_disc = None
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    
        pretrain_G, pretrain_D, pretrain_dur = load_pretrain_model()
        hps.pretrain_G = hps.pretrain_G or pretrain_G
        hps.pretrain_D = hps.pretrain_D or pretrain_D
        hps.pretrain_dur = hps.pretrain_dur or pretrain_dur

        if hps.pretrain_G:
            utils.load_checkpoint(
                    hps.pretrain_G,
                    net_g,
                    None,
                    skip_optimizer=True
                )
        if hps.pretrain_D:
            utils.load_checkpoint(
                    hps.pretrain_D,
                    net_d,
                    None,
                    skip_optimizer=True
                )


        if net_dur_disc is not None:
            net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)
            if hps.pretrain_dur:
                utils.load_checkpoint(
                        hps.pretrain_dur,
                        net_dur_disc,
                        None,
                        skip_optimizer=True
                    )
    except Exception as e: 
        logger.error(f"Error in init_model: {e}")
        print(f"Error in init_model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"Traceback: {traceback.format_exc()}")

    try:
        if net_dur_disc is not None:
            _, _, dur_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                net_dur_disc,
                optim_dur_disc,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
                net_g,
                optim_g,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
                net_d,
                optim_d,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            if not optim_g.param_groups[0].get("initial_lr"):
                optim_g.param_groups[0]["initial_lr"] = g_resume_lr
            if not optim_d.param_groups[0].get("initial_lr"):
                optim_d.param_groups[0]["initial_lr"] = d_resume_lr
            if not optim_dur_disc.param_groups[0].get("initial_lr"):
                optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr

        epoch_str = max(epoch_str, 1)
        global_step = (epoch_str - 1) * len(train_loader)
        if rank == 0:
            logger.info(f"=== CHECKPOINT RESUME ===")
            logger.info(f"Resumed from epoch: {epoch_str}")
            logger.info(f"Train loader length: {len(train_loader)}")
            logger.info(f"Calculated global_step: {global_step}")

    except Exception as e:
        if rank == 0:
            logger.info(f"=== NO CHECKPOINTS FOUND ===")
            print(f"=== NO CHECKPOINTS FOUND ===")
            logger.info(f"Exception: {e}")
            print(f"Exception: {e}")
        print(e)
        epoch_str = 1
        global_step = 0
        if rank == 0:
            logger.info(f"Starting fresh - epoch: {epoch_str}, global_step: {global_step}")
            print(f"Starting fresh - epoch: {epoch_str}, global_step: {global_step}")

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None
    scaler = GradScaler(enabled=hps.train.fp16_run)


    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            logger.info(f"=== STARTING EPOCH {epoch} ===")
            print(f"=== STARTING EPOCH {epoch} ===")
            logger.info(f"global_step at epoch start: {global_step}")
            print(f"global_step at epoch start: {global_step}")
        try:
            if rank == 0:
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_d, net_dur_disc],
                    [optim_g, optim_d, optim_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler,
                    [train_loader, eval_loader],
                    logger,
                    [writer, writer_eval],
                )
            else:
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_d, net_dur_disc],
                    [optim_g, optim_d, optim_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler,
                    [train_loader, None],
                    None,
                    None,
                )
        except Exception as e:
            if rank == 0:
                logger.error(f"Error in epoch {epoch}: {e}")
                print(f"Error in epoch {epoch}: {e}")
            print(e)
            torch.cuda.empty_cache()
        if rank == 0:
            logger.info(f"=== COMPLETED EPOCH {epoch} ===")
            print(f"=== COMPLETED EPOCH {epoch} ===")
            logger.info(f"global_step after epoch: {global_step}")
            print(f"global_step after epoch: {global_step}")

        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    if rank == 0 and logger:
        logger.info(f"=== TRAIN_AND_EVALUATE START ===")
        print(f"=== TRAIN_AND_EVALUATE START ===")
        logger.info(f"Epoch: {epoch}, global_step at start: {global_step}")
        print(f"Epoch: {epoch}, global_step at start: {global_step}")
        logger.info(f"log_interval: {hps.train.log_interval}")
        print(f"log_interval: {hps.train.log_interval}")
        logger.info(f"eval_interval: {hps.train.eval_interval}")
        print(f"eval_interval: {hps.train.eval_interval}")

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    for batch_idx, (
        x,
        x_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        speakers,
        tone,
        language,
        bert,
        ja_bert,
    ) in enumerate(tqdm(train_loader)):

        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
            rank, non_blocking=True
        )
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        speakers = speakers.cuda(rank, non_blocking=True)
        tone = tone.cuda(rank, non_blocking=True)
        language = language.cuda(rank, non_blocking=True)
        bert = bert.cuda(rank, non_blocking=True)
        ja_bert = ja_bert.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
            ) = net_g(
                x,
                x_lengths,
                spec,
                spec_lengths,
                speakers,
                tone,
                language,
                bert,
                ja_bert,
            )
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
                )
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                # Add debug logging before the existing log
                if logger:
                    logger.info(f"DEBUG: Logging condition met - global_step: {global_step}, batch_idx: {batch_idx}")
                    print(f"DEBUG: Logging condition met - global_step: {global_step}, batch_idx: {batch_idx}")
                
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                print(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])
                print([x.item() for x in losses] + [global_step, lr])
                import traceback
                try:
                    scalar_dict = {
                        "loss/g/total": loss_gen_all,
                        "loss/d/total": loss_disc_all,
                        "learning_rate": lr,
                        "grad_norm_d": grad_norm_d,
                        "grad_norm_g": grad_norm_g,
                    }
                    scalar_dict.update(
                        {
                            "loss/g/fm": loss_fm,
                            "loss/g/mel": loss_mel,
                            "loss/g/dur": loss_dur,
                            "loss/g/kl": loss_kl,
                        }
                    )
                    scalar_dict.update(
                        {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                    )
                    scalar_dict.update(
                        {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                    )
                    scalar_dict.update(
                        {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                    )

                    image_dict = {
                        "slice/mel_org": utils.plot_spectrogram_to_numpy(
                            y_mel[0].data.cpu().numpy(), logger=logger
                        ),
                        "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].data.cpu().numpy(), logger=logger
                        ),
                        "all/mel": utils.plot_spectrogram_to_numpy(
                            mel[0].data.cpu().numpy(), logger=logger
                        ),
                        "all/attn": utils.plot_alignment_to_numpy(
                            attn[0, 0].data.cpu().numpy(), logger=logger
                        ),
                    }
                    utils.summarize(
                        writer=writer,
                        global_step=global_step,
                        images=image_dict,
                        scalars=scalar_dict,
                    )
                except Exception as e:
                    if logger:
                        logger.error(f"Error on Utils-Summarize: {e}")
                        print(f"Error on Utils-Summarize: {e}")
                        logger.error(f"Full traceback:\n{traceback.format_exc()}")
                        print(f"Full traceback:\n{traceback.format_exc()}")
                    


            if global_step % hps.train.eval_interval == 0:
                if logger:
                    logger.info(f"DEBUG: Checkpoint condition met - global_step: {global_step}")
                    print(f"DEBUG: Checkpoint condition met - global_step: {global_step}")
                    logger.info(f"About to save checkpoints...")
                    print(f"About to save checkpoints...")

                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )
                keep_ckpts = getattr(hps.train, "keep_ckpts", 5)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )

        global_step += 1

        # Add logging after increment
        if rank == 0 and logger and batch_idx % 50 == 0:  # Log every 50 batches
            logger.info(f"DEBUG: After increment - batch_idx: {batch_idx}, global_step: {global_step}")
            print(f"DEBUG: After increment - batch_idx: {batch_idx}, global_step: {global_step}")


    if rank == 0:
        logger.info(f"=== TRAIN_AND_EVALUATE END ===")
        print(f"=== TRAIN_AND_EVALUATE END ===")
        logger.info(f"Epoch: {epoch}, global_step at end: {global_step}")
        print(f"Epoch: {epoch}, global_step at end: {global_step}")
        logger.info("====> Epoch: {}".format(epoch))
        print("====> Epoch: {}".format(epoch))
    torch.cuda.empty_cache()


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    print("Evaluating ...")
    with torch.no_grad():
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
            tone,
            language,
            bert,
            ja_bert,
        ) in enumerate(eval_loader):
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            ja_bert = ja_bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            for use_sdp in [True, False]:
                y_hat, attn, mask, *_ = generator.module.infer(
                    x,
                    x_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    ja_bert,
                    y=spec,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                image_dict.update(
                    {
                        f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].cpu().numpy()
                        )
                    }
                )
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                            0, :, : y_hat_lengths[0]
                        ]
                    }
                )
                image_dict.update(
                    {
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            mel[0].cpu().numpy()
                        )
                    }
                )
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()
    print('Evauate done')
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
