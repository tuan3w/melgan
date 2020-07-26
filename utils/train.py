import os
import math
import tqdm
import torch
import numpy as np
import itertools
import traceback
from utils.pqmf import PQMF
from model.generator import Generator
from .utils import get_commit_hash
from .validation import validate
from utils.stft_loss import MultiResolutionSTFTLoss


def train(args, pt_dir, chkpt_path, trainloader, valloader, writer, logger, hp, hp_str):
    model = Generator(hp.audio.n_mel_channels, hp.model.n_residual_layers,
                        ratios=hp.model.generator_ratio, mult = hp.model.mult,
                        out_band = hp.model.out_channels).cuda()
    print("Generator : \n",model)

    optim = torch.optim.Adam(model.parameters(),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))

    githash = get_commit_hash()


    init_epoch = -1
    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint. Will use new.")

        if githash != checkpoint['githash']:
            logger.warning("Code might be different: git hash is different.")
            logger.warning("%s -> %s" % (checkpoint['githash'], githash))

    else:
        logger.info("Starting new training run.")

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    try:
        model.train()
        stft_loss = MultiResolutionSTFTLoss()
        for epoch in itertools.count(init_epoch+1):

            trainloader.dataset.shuffle_mapping()
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
            for (mel, audio) in loader:
                mel = mel.cuda()      # torch.Size([16, 80, 64])
                audio = audio.cuda()  # torch.Size([16, 1, 16000])

                # generator
                optim.zero_grad()
                predict = model(mel)[:, :, :hp.audio.segment_length]  # torch.Size([16, 1, 12800])
                loss = 0.0
                sc_loss, mag_loss, phase_loss = stft_loss(predict[:, :, :audio.size(2)].squeeze(1), audio.squeeze(1))
                loss = sc_loss + mag_loss + phase_loss
                loss.backward()
                optim.step()


                step += 1
                # logging
                if step % hp.log.summary_interval == 0:
                    writer.log_training(loss, sc_loss, mag_loss, phase_loss, step)

                if step % hp.log.validation_interval == 0:
                    # save sample
                    idx = np.random.randint(audio.shape[0])
                    t1 = audio[idx].data.cpu()
                    t2 = predict[idx].data.cpu()
                    writer.log_audio(t2, t1, step)
            if epoch % hp.log.save_interval == 0:
                save_path = os.path.join(pt_dir, '%s_%s_%04d.pt'
                    % (args.name, githash, epoch))
                torch.save({
                    'model': model.state_dict(),
                    'optim_g': optim.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'hp_str': hp_str,
                    'githash': githash,
                }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
