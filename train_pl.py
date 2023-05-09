import os
from dataset.data_loader import AudioVisualDataModule
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchaudio.transforms as Ta
print("CUDA is Available: ", torch.cuda.is_available())
import argparse
import av
import numpy as np
import torch
import speechmetrics

def str2list(v: str):
    return eval(v)

def str2bool(v: str):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def dicttodevice(d, device):
    for key in d.keys():
        d[key] = d[key].to( device )
    return d

# from nnet.losses import get_si_snr_with_pitwrapper
# from model.visual import Encoder3D
# from model.inception_resnet_3D import InceptionResnet3D


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

from mel2wav.modules import Generator, Discriminator, Audio2Mel
import torch.nn.functional as F
import torchaudio
import torchmetrics

class SpeechMetrics:
    def __init__(self, sampling_rate):
        self.metrics = {

            "snr": torchmetrics.SignalNoiseRatio(),
            "stoi": torchmetrics.audio.stoi.ShortTimeObjectiveIntelligibility(fs = sampling_rate),
            "pesq": torchmetrics.audio.pesq.PerceptualEvaluationSpeechQuality(fs = sampling_rate, mode = "nb"),
            "si_sdr": torchmetrics.ScaleInvariantSignalDistortionRatio(),
            "sdr": torchmetrics.SignalDistortionRatio(),

        }

    def __call__(self, preds, target):
        
        preds = preds/torch.max(torch.abs(preds), dim = -1)[0].unsqueeze(-1)
        target = target/torch.max(torch.abs(target), dim = -1)[0].unsqueeze(-1)

        results = {}
        for name, metric in self.metrics.items():
            results[name] = round(metric(preds.cpu(), target.cpu()).detach().cpu().item(), 3)
        return results

class AudioVisualModel(pl.LightningModule):
    def __init__(
        self,
        args
        ):
        super(AudioVisualModel, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False


        self.lr = args.lr

        self.datamodule = AudioVisualDataModule(args)
        self.train_dataset = self.datamodule.train_dataset
        self.valid_dataset = self.datamodule.valid_dataset
        self.test_dataset  = self.datamodule.test_dataset

        netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers)
        self.netD = Discriminator(args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor)
        self.fft = nn.Sequential(
            torchaudio.transforms.Resample(orig_freq=args.sampling_rate, new_freq=22050,),
            Audio2Mel(n_mel_channels=args.n_mel_channels, mel = args.mel)
        )

        # netG.load_state_dict(torch.load("./models/multi_speaker.pt", map_location = "cpu"))

        self.netG = nn.Sequential(
            netG,
            torchaudio.transforms.Resample(orig_freq=22050, new_freq=args.sampling_rate,),

        )

        print(self.netG)

        self.metrics = SpeechMetrics(sampling_rate=args.sampling_rate)
        
        self.logger_step = 0



        self.args = args

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")


        parser.add_argument("--lr", type=float, default=1E-4)


        parser.add_argument("--exp_name", type=str, default="melgan")
        parser.add_argument("--audio_only", type=str2bool, default="True")
        parser.add_argument("--noise_db_bounds", type=str2list, default='(-2.5, 2.5)')
        parser.add_argument("--noise_db_bounds_test", type=str2list, default='(-2.5, 2.5)')
        parser.add_argument("--denoise_audio", type=str2bool, default='False')
        
        parser.add_argument("--mel", type=str2bool, default='True')

        parser.add_argument("--n_mel_channels", type=int, default=80)
        parser.add_argument("--ngf", type=int, default=32)
        parser.add_argument("--n_residual_layers", type=int, default=3)

        parser.add_argument("--ndf", type=int, default=16)
        parser.add_argument("--num_D", type=int, default=3)
        parser.add_argument("--n_layers_D", type=int, default=4)
        parser.add_argument("--downsamp_factor", type=int, default=4)
        parser.add_argument("--lambda_feat", type=float, default=10)
        parser.add_argument("--cond_disc", action="store_true")

        parser.add_argument("--seq_len", type=int, default=8192)



        return parent_parser

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def forward(self, batch_inp):

        x_t = batch_inp["audio"]



        s_t = self.fft(x_t).detach()

        # Ideal Binary Mask
        if self.args.mel:
            threshold = 1 
        else:
            threshold = 0
        ibm = torch.sign(s_t+threshold) # anything larget than -1 is +1 else 0
        x_pred_t = self.netG(ibm)

        with torch.no_grad():
            s_pred_t = self.fft(x_pred_t.detach())
            s_error = F.l1_loss(s_t, s_pred_t).item()
        return s_error



    def training_step(self, batch_inp, batch_idx, ):

        x_t = batch_inp["audio"][:, 0, ]


        optimizer_g, optimizer_d = self.optimizers()
        # lrscheduler_g, lrscheduler_d = self.lr_schedulers()

        s_t = self.fft(x_t).detach()

        # Ideal Binary Mask
        if self.args.mel:
            threshold = 1 
        else:
            threshold = 0
        ibm = torch.sign(s_t+threshold) # anything larget than -1 is +1 else 0
        x_pred_t = self.netG(ibm)

        with torch.no_grad():
            s_pred_t = self.fft(x_pred_t.detach())
            s_error = F.l1_loss(s_t, s_pred_t)

        N = min(x_pred_t.shape[-1], x_t.shape[-1])
        x_t = x_t[..., :N]
        x_pred_t = x_pred_t[..., :N]
        
        # x_t = 0.9 * x_t / x_t.max(dim = -1, keepdim = True)[0]
        # x_pred_t = 0.9 * x_pred_t / x_pred_t.max(dim = -1, keepdim = True)[0]
        #######################
        # Train Discriminator #
        #######################
        self.toggle_optimizer(optimizer = optimizer_d, optimizer_idx = 1)

        D_fake_det = self.netD(x_pred_t.detach())
        D_real = self.netD(x_t)

        loss_D = 0
        # L_{D} = -\mathbb{E}_{\left(x, y\right)\sim{p}_{data}}\left[\min\left(0, -1 + D\left(x, y\right)\right)\right] -\mathbb{E}_{z\sim{p_{z}}, y\sim{p_{data}}}\left[\min\left(0, -1 - D\left(G\left(z\right), y\right)\right)\right]
        # L_{G} = -\mathbb{E}_{z\sim{p_{z}}, y\sim{p_{data}}}D\left(G\left(z\right), y\right)
        for scale_fake in D_fake_det:
            loss_D += F.relu(1 + scale_fake[-1]).mean()

        for scale_real in D_real:
            loss_D += F.relu(1 - scale_real[-1]).mean()

        optimizer_d.zero_grad()
        self.manual_backward( loss_D )
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        ###################
        # Train Generator #
        ###################
        self.toggle_optimizer( optimizer_g, optimizer_idx = 0)

        D_fake = self.netD(x_pred_t)

        loss_G = 0
        for scale in D_fake:
            loss_G += -scale[-1].mean()

        loss_feat = 0
        feat_weights = 4.0 / (self.args.n_layers_D + 1)
        D_weights = 1.0 / self.args.num_D
        wt = D_weights * feat_weights
        for i in range(self.args.num_D):
            for j in range(len(D_fake[i]) - 1):
                loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())


        optimizer_g.zero_grad()
        self.manual_backward(loss_G + self.args.lambda_feat * loss_feat)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)




        ######################
        # Update tensorboard #
        ######################
        loss = loss_G + self.args.lambda_feat * loss_feat + loss_D

        losses = {
            "loss": loss,
            "discriminator": loss_D,
            "generator": loss_G,
            "feature_matching": loss_feat,
            "mel_reconstruction": s_error,

        }
        self.post_process(x_pred_t, x_t, s_t, s_pred_t, ibm, losses, trainval = "train")
        
    def validation_step(self, batch_inp, batch_idx):
        x_t = batch_inp["audio"][:, 0, ]

        s_t = self.fft(x_t).detach()

        # Ideal Binary Mask
        if self.args.mel:
            threshold = 1 
        else:
            threshold = 0
        ibm = torch.sign(s_t+threshold) # anything larget than -1 is +1 else 0
        x_pred_t = self.netG(ibm)

        with torch.no_grad():
            s_pred_t = self.fft(x_pred_t.detach())
            s_error = F.l1_loss(s_t, s_pred_t)

        N = min(x_pred_t.shape[-1], x_t.shape[-1])
        x_t = x_t[..., :N]
        x_pred_t = x_pred_t[..., :N]

        losses = {
            "mel_reconstruction": s_error,
        }
        self.post_process(x_pred_t, x_t, s_t, s_pred_t, ibm, losses, trainval = "val")

    def post_process(self, x_pred_t, x_t, s_t, s_pred_t, ibm, losses, trainval = "train"):
        

        if self.logger_step % 100 == 0:
            scores = self.metrics(x_pred_t, x_t)
            
            
            self.log(f"{trainval}/stoi", scores["stoi"], on_step=False, on_epoch=True, logger=True, sync_dist= False, rank_zero_only = False)
        
            step = self.global_step
            writer = self.logger.experiment
            writer.add_scalar('Epoch', self.current_epoch, step,)


            for k in scores.keys():
                writer.add_scalar(f"{trainval}/{k}", scores[k], step)

            for k in losses.keys():
                writer.add_scalar( f"{trainval}/{k}" , losses[k].cpu().item(), step)

            writer.add_audio(f"{trainval}_aud/pred", x_pred_t[0].reshape(-1, 1), global_step=step, sample_rate=self.args.sampling_rate)
            writer.add_audio(f"{trainval}_aud/target", x_t[0].reshape(-1, 1), global_step=step, sample_rate=self.args.sampling_rate)

            fig, AX = plt.subplots(3, min(s_t.shape[0]+1, 2), figsize = (12, 8), sharex=True, sharey=True)

            for i in range(min(s_t.shape[0], 2)):
                AX[0, i].imshow(s_t[i].detach().cpu(), aspect="auto", origin='lower')
                AX[1, i].imshow(ibm[i].detach().cpu(), aspect="auto", origin='lower', cmap = "gray")
                AX[2, i].imshow(s_pred_t[i].detach().cpu(), aspect="auto", origin='lower')

            ylabels = ["Target", "Ideal Binary Mask", "Predicted"]
            for i in range(3):
                AX[i, 0].set_ylabel( ylabels[i] )

            writer.add_figure(f"{trainval}/MelSpec", figure = fig, global_step = step )
            
        self.logger_step += 1


    



    def configure_optimizers(self):
        from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
        from torch.optim import Adam

        optG = torch.optim.Adam(self.netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
        # schG = ReduceLROnPlateau(optG, factor  = 0.85, patience = 3, verbose = True)

        optD = torch.optim.Adam(self.netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        # schD = ReduceLROnPlateau(optD, factor  = 0.85, patience = 3, verbose = True)

        return [optG, optD], []

        # return ({
        #     "optimizer": optG,
        #     "lr_scheduler": {
        #         "scheduler": schG,
        #         "interval": "epoch",
        #         "monitor": 'val_loss',
        #         "strict": False,
        #     },
        # },
        # {
        #     "optimizer": optD,
        #     "lr_scheduler": {
        #         "scheduler": schD,
        #         "interval": "epoch",
        #         "monitor": 'val_loss',
        #         "strict": False,
        #     },
        # })



def main(args):
    import os
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.strategies import DDPStrategy
    from pytorch_lightning.callbacks import TQDMProgressBar
    from pytorch_lightning.callbacks import StochasticWeightAveraging
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks import ModelCheckpoint

    # from model.callbacks import SNRSheduler

    # seed everything
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)
    pl.utilities.seed.seed_everything(args.seed)
    torch.cuda.empty_cache()





    checkpoint_callback = ModelCheckpoint(
        monitor='val/stoi',
        save_on_train_epoch_end = False, # does checkpointing at teh end of validation step (False) or training epoch (True)
        save_last= True,
        every_n_epochs=args.check_val_every_n_epoch,
        mode= 'max',
        save_top_k = 1,
        filename= "{epoch:02d}-{val/stoi:.2f}",
        dirpath='./checkpoints/',
        )
    checkpoint_callback.CHECKPOINT_NAME_LAST = args.exp_name + "-last"
    logger = TensorBoardLogger("lightning_logs", name= args.exp_name)
    # early_stopping_callback = EarlyStopping(monitor="loss",  check_finite = False, patience=5, verbose=True, mode="min", check_on_train_epoch_end = False)

    call_backs = [
        checkpoint_callback,
        # early_stopping_callback,
        TQDMProgressBar(refresh_rate=1),
        # StochasticWeightAveraging(swa_lrs=1e-2),

    ]

    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

    strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=call_backs,
            logger = logger,
            strategy = strategy,
            accelerator = "gpu",
            devices = -1,
            # profiler="advanced"
            )

    model = AudioVisualModel(args = args)

    if args.start_from_beginning:
        if args.ckpt_path.lower() == 'none':



            trainer.fit( model )

        elif args.ckpt_path.lower() != 'last':
            print("------------  Loading from checkpoint   ----------")
            def load_pretrained(model, args):
                ckpt_path = args.ckpt_path
                pretrained = torch.load( ckpt_path, map_location=lambda storage, loc: storage )['state_dict']


                pretrained_dict = pretrained # speech brain pretrained model
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
                missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]

                print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
                if len(missed_params) > 0:
                    print('First 10 missed_params: \n', missed_params[:10])
                # print('miss matched params:',missed_params)
                model.load_state_dict(model_dict)

                return model

            model = load_pretrained(model, args)


            trainer.fit( model )

    else:

        trainer.fit( model, ckpt_path = args.ckpt_path )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speech Separation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, default='last')
    parser.add_argument("--start_from_beginning", type=str2bool, default= 0)

    parser = pl.Trainer.add_argparse_args(parser)

    parser = AudioVisualModel.add_model_specific_args(parser)
    parser = AudioVisualDataModule.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)