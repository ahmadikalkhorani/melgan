import sys
sys.path.append('../')

from torch.utils.data import Dataset
import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
from torch.utils.data import Dataset
import torchaudio
import pdb
import decord as de 
from pathlib import Path
import torchvision.transforms as Tv
import torchaudio.transforms as Ta
de.bridge.set_bridge('torch')
import pytorch_lightning as pl
import random
import re
import argparse
import av
import numpy as np
import torch

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
                
de.bridge.set_bridge('torch')

EPS = torch.tensor(1E-5)
# torch.tensor(torch.finfo(x.dtype).eps
eps_function = lambda x: torch.tensor(torch.finfo(x.dtype).tiny)

RMS = lambda x: torch.max( torch.sqrt( torch.mean(x ** 2) ),  eps_function(x))

def audio_normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = torch.max( torch.sqrt( torch.mean(x ** 2) ),  eps)
  samples = samples * (desired_rms / rms)
  return samples


MAX = lambda x: torch.max( torch.max(torch.abs(x)),  eps_function(x))

def first_non_zero_idx(x):
    # x should be 2D: for example : 1x16000 
    x = x.clone()
    x = x.sort(dim=1)[0]
    first_positive = (x <= 0).sum(dim=1)
    return int(first_positive.item())


def normalize_resample(s_path, fs, duration, rng: random.Random, normalize = 'enegry', remove_starting_zero = False,):

    assert normalize.lower() in ['energy', 'max', 'none']
    
    
    N = int(fs * duration)
    s, _fs = torchaudio.load( s_path )

    if remove_starting_zero:
        s = s[:, first_non_zero_idx(s):]
        
    s = s[0:1, :] # left channel only
    s = torchaudio.transforms.Resample(_fs, fs)(s) 

    S = torch.zeros((1, N), dtype = s.dtype)

    n_s = torch.numel(s)
    
    if n_s>N:
        idx_start = rng.randint(0, n_s - N)
        idx_end = idx_start + N
        S[0, :] = s[0, idx_start:idx_end]
    elif n_s<=N:
        S[0, :n_s] = s

    if normalize.lower() == 'energy':
        S = S/RMS(S)
    elif normalize.lower() == 'max':
        
        S = S/MAX(S)
    
    return S



def mix_s_n(s, n, db, ref_mic):
    # signal will have `db` SNR in the result mixture
    # pdb.set_trace()
    if torch.sum(n**2) < 1E-5:
        alpha = 0.0
    else:
        alpha = torch.sqrt(  torch.sum(s[ref_mic]**2)/(eps_function(s) + torch.sum(n[ref_mic]**2)*10**(db/10)) )
    
    mix = s + alpha*n
    # mix = s + 0.3 * n # Google's approach

    return mix




def videoAugment(d, MaxShiftFrame = 10, rng = None):

    rng  = random.Random() if rng is None else rng

    n_shift = max(-MaxShiftFrame, min(int(rng.normalvariate(0, MaxShiftFrame/4)), MaxShiftFrame))

    if n_shift >= 0:
        augmented_video = torch.cat([d['video'][n_shift:], d['video'][:n_shift] * 0.0 ])
        augmented_audio = d['audio']
    else:
        n_shift = abs( n_shift )
        augmented_video = d['video']
        augmented_audio = d['audio'][n_shift:] + [d['audio'][i] * 0.0 for i in range(n_shift)]

    return {'audio': augmented_audio, 'video': augmented_video}


class WSJWrapper():
    def __init__(self, data_folder_path, sr = 8000, mode = 'min', n_speaker = 2, task = 'tr', **kwargs):

        assert sr in [8000, 16000]
        assert task in ['tr', 'cv', 'tt']

        self.n_spk = n_speaker

        self.sr = sr

        self.folder = os.path.join(data_folder_path, f"{int(n_speaker)}speakers", f"wav{int(sr/1000)}k", mode)

        self.utt_list = {
            'tr': os.listdir( os.path.join(os.path.join(self.folder, 'tr'), 'mix') ),
            'cv': os.listdir( os.path.join(os.path.join(self.folder, 'cv'), 'mix') ),
            'tt': os.listdir( os.path.join(os.path.join(self.folder, 'tt'), 'mix') ),
        }

        self.len = {
                'tr': len(self.utt_list['tr']),
                'cv': len(self.utt_list['cv']),
                'tt': len(self.utt_list['tt']),
            }

    def __len__(self):
        return len( self.utt_list )           

    def get(self, idx, task):

        S = []
        for i in range(self.n_spk):
            s, fs = torchaudio.load( os.path.join(self.folder, task, f"s{i+1}", self.utt_list[task][idx]) )
            S.append(s)
        
        s, fs = torchaudio.load( os.path.join(self.folder, task, "mix", self.utt_list[task][idx]) )
        S.append(s)

        S = torch.stack(S)

        return {'audio': S}





def RGB_last(img):
    if img.shape[-1] > 3:
        img = torch.transpose(img, -1, -3)
        img = torch.transpose(img, -2, -3)
    return img

def RGB_first(img):
    if img.shape[-3] > 3:
        img = torch.transpose(img, -1, -3)
        img = torch.transpose(img, -2, -1)
    return img

class LRS3Wrapper():
    def __init__(
        self, 
        data_folder_path , 
        duration = 3.0, 
        sr = 8000, 
        n_speaker = 2, 
        task = 'tr', 
        SIR_lb = -2.5, 
        SIR_ub = +2.5,
        audio_only = False, 
        dynamic_mixing = True,
        **kwargs
        ):

        assert task in ['tr', 'cv', 'tt']

        self.n_spk = n_speaker
        self.duration = duration
        self.dynamic_mixing = dynamic_mixing

        self.sr = sr

        self.folder = Path(data_folder_path)

        self.SIR_lb = SIR_lb
        self.SIR_ub = SIR_ub
        
        self.spk_list = {
            'tr': [f for f in self.folder.joinpath('pretrain').glob('*') if len(os.listdir(f)) > 0], 
            'cv': [f for f in self.folder.joinpath('trainval').glob('*') if len(os.listdir( f)) > 0], 
            'tt': [f for f in self.folder.joinpath('test').glob('*') if len(os.listdir( f)) > 0], 
            
        }

        
        self.len = {
            'tr': 200000,
            'cv': 40000,
            'tt': 3000,
        }

        self.audio_only = audio_only

        self.idx = 0


    
    def load_AV(self, data_folder,  audio_only, rng, sample_rate = 16000, duration = 3.0, video_fps = 25, size = (160, 160), task = 'cv'):
        
        # rng = random.Random(seed)
            
        try:
            
            N_frames = int(duration * video_fps)
            
            # d = np.load(data_folder, allow_pickle = 1)['data'].item()
            d = torch.load(data_folder)
            if task == 'tr':
                d = videoAugment(d, 10, rng=rng)
            # a = torch.from_numpy( np.concatenate(d['audio'], -1) ).reshape(-1)
            a = torch.cat( d['audio'] , -1).reshape(-1)
            
            a = Ta.Resample(16000, new_freq=sample_rate)(a)
            
            
            audio = torch.zeros(int(N_frames * sample_rate / video_fps))
            n = min(audio.shape[0], a.shape[0])
            audio[:n] = a[:n]
            

            
            if audio_only:
                return audio, None
            # faces = v
            v = d['video']
            video = torch.zeros(N_frames, *v.shape[1:], dtype = torch.uint8)
            n = min(N_frames, v.shape[0])
            video[:n] = v[:n]
            
                
        
            video = RGB_first(video) # N_frame x W x H x C  ->   N_frame x C x W x H
            video = Tv.Resize(size)(video)
            
            video = (video-127.5)/128.0

            return audio, video
        except:
            print("\n")
            print(data_folder)
            print("\n")
            raise 




    def get(self, idx, task):
        self.idx = idx
        # try:
        
        rng = random.Random() if (self.dynamic_mixing & (random.random() < 1.0)  &  (task  == 'tr' )) else random.Random(idx)
        speakers = rng.sample(self.spk_list[task], self.n_spk)
        
        files_path = [rng.sample( [f for f in Path(speakers[i]).glob("*.pt")] , 1)[0] for i in range(self.n_spk)]
        
        snrs = torch.zeros(self.n_spk)
        snrs[:self.n_spk//2] = torch.tensor([rng.uniform(self.SIR_lb, self.SIR_ub) for _ in range(self.n_spk//2)])
        # np.random.uniform(self.SIR_lb, self.SIR_ub, self.n_spk//2) # the maximum snr would be 2 * mix_snr -> divided by 2
        snrs[self.n_spk//2:2*(self.n_spk//2)] = -torch.flip(snrs[:self.n_spk//2], dims = (0, ))
        weights = 10.0**(snrs/20.0)


        S = torch.zeros(self.n_spk + 1, int(self.sr * self.duration))
        Video = []
        for i in range(self.n_spk):
            # data = get_AV(video_path = s_paths[i], catalog = self.catalog[task], detector = self.detector, sample_rate = self.sr, duration = self.duration, video_fps = 25, seed = idx, audio_only = self.audio_only)
            a, v = self.load_AV(
                data_folder = files_path[i], 
                audio_only = self.audio_only, 
                seed = idx, 
                sample_rate = self.sr, 
                duration = self.duration, 
                video_fps = 25, 
                rng = rng,
            )
                
            audio = a[:int(self.sr * self.duration)]
            S[i, :audio.shape[-1]] =  weights[i] * audio / RMS(audio)
            # S[i, :audio.shape[-1]] =  audio # Google's approach \Sum(a_i) + 0.3 * noise
            if not self.audio_only:
                Video.append( v )

        S[-1] = torch.sum(S, dim = 0)

        gain = torch.max(torch.tensor([1., torch.max(torch.abs(S[-1]))]) ) / 0.9

        S = S / (eps_function(S) + gain)

        if self.audio_only:
            return {'audio': S} 
        

        return {'audio': S, 'video': torch.stack(Video)}



class AudioSet():
    def __init__(self, data_folder_path, replace = False, sr = 16000, duration = 3.0, normalize = 'energy', dynamic_mixing = True, **kwargs):
        train_folder = os.path.join(data_folder_path, "unbalanced_train_segments")
        eval_folder = os.path.join(data_folder_path, "eval_segments")
        
        
        train_files = pd.read_csv(os.path.join(data_folder_path, "unbalanced_train_segments.csv"))['file'].to_list()
        eval_files = pd.read_csv(os.path.join(data_folder_path, "eval_segments.csv"))['file'].to_list()
        
        n_eval = len(eval_files)
        
        self.audio_files = {
            'tr': train_files, 
            'cv': eval_files[:n_eval//2],
            'tt': eval_files[n_eval//2:],
        }

        self.sr = sr
        self.duration = duration
        self.normalize = normalize
        
        self.dynamic_mixing = dynamic_mixing

        self.data_folder_path = data_folder_path


    def get(self, idx, task):
        
        rng = random.Random() if (self.dynamic_mixing & (random.random() < 1.0)  &  (task  == 'tr' )) else random.Random(idx)
        
        # files_path = rng.sample(self.audio_files[task], 1)[0]

        path = rng.sample(self.audio_files[task], 1)[0]
        try:
            n = normalize_resample(s_path = path,  fs = self.sr, rng = rng, duration = self.duration, normalize = self.normalize)
        except:
            try:
                os.remove(path)
            except:
                pass
            return self.get(idx+1, task)
        
        # if n.abs().max() == 0:
        #     pd.DataFrame([str(Path(path).absolute())]).to_csv(f'{self.data_folder_path}/rm_these.csv', mode='a+', header=None, index=None)
        #     return self.get(idx+1, task)
        
        return {
            'audio': n
        }


class WHAM():
    def __init__(self, data_folder_path, replace = False, sr = 16000, duration = 3.0, normalize = 'energy', **kwargs):
        tr_folder = os.path.join(data_folder_path, "tr")
        cv_folder = os.path.join(data_folder_path, "cv")
        tt_folder = os.path.join(data_folder_path, "tt")

        
        self.audio_files = {
            'tr': [os.path.join(tr_folder, f) for f in os.listdir(tr_folder) ], 
            'cv': [os.path.join(cv_folder, f) for f in os.listdir(cv_folder) ],
            'tt': [os.path.join(tt_folder, f) for f in os.listdir(tt_folder) ],
        }

        self.sr = sr
        self.duration = duration
        self.normalize = normalize


    def get(self, idx, task, rng):
        try:

            path = rng.sample(self.audio_files[task], 1)[0]

            n = normalize_resample(s_path = path, rng = rng, fs = self.sr, duration = self.duration, normalize = self.normalize)
            
            return {
                'audio': n.reshape(1, -1)
            }
        except:
            print("Error in reading noise file:  ", path)
            return {}




















import glob


class MultiChannelEasyCom():
    def __init__(
        self,
        data_folder_path="/scratch/vahid/SPEAR_Dataset_2/",
        duration=3.0,
        sr=16000,
        n_speaker=2,
        task='tr',
        SIR_lb=0.0,
        SIR_ub=0.0,
        audio_only=False,
        dynamic_mixing=True,
        num_mic=4,
        ref_mic=1,
        **kwargs
    ):

        assert task in ['tr', 'cv', 'tt']

        self.n_spk = n_speaker
        self.duration = duration

        self.sr = sr
        self.fps = 20
        self.num_mic = num_mic
        self.ref_mic = ref_mic

        self.dynamic_mixing = dynamic_mixing

        self.folder = Path(data_folder_path)

        self.SIR_lb = SIR_lb
        self.SIR_ub = SIR_ub

        test_files = glob.glob(f"{self.folder}/preprocessed/Session_1*/*.pt")
        train_files = [f for f in glob.glob(
            f"{self.folder}/preprocessed/Session_*/*.pt") if f not in test_files]

        rng = random.Random(2023)
        rng.shuffle(train_files)

        N = int(0.9 * len(train_files))

        self.clip_list = {
            'tr': train_files[:N],
            'cv': train_files[N:],
            'tt': test_files,

        }

        self.len = {
            'tr': int(len(self.clip_list["tr"]) * 60 / duration),
            'cv': int(len(self.clip_list["cv"]) * 60 / duration),
            'tt': int(len(self.clip_list["tt"]) * 60 / duration),
        }

        print(self.len)

        self.audio_only = audio_only

        self.idx = 0

    def normalze_resample(self, s, old_freq):
        s = Ta.Resample(old_freq, new_freq=self.sr)(s)
        s = s/RMS(s[self.ref_mic])
        return s

    def get(self, idx, task):

        file_path = self.clip_list[task][idx//int(60/self.duration)]


        d = torch.load(file_path,  map_location=lambda storage, loc: storage)

        t1 = (idx % (60/self.duration)) * self.duration
        t2 = t1 + self.duration

        S = d["audio"]

        S = torchaudio.transforms.Resample(16000, new_freq=self.sr)(S)

        S = S[..., int(t1 * self.sr): int(t2 * self.sr)]

        gain = torch.max(torch.tensor([1., torch.max(torch.abs(S))])) / 0.9

        a = S.clone() / gain

        v = d["video"][:, int(t1 * self.fps): int(t2 * self.fps)]
        v = (v - 127.5)/128.0

        return {"audio": a, "video": v}

        # if task != "tt":
        #     raise "Not implemented"

        # return


class SingleChannelAVSpeech():
    def __init__(
        self,
        data_folder_path,
        duration=3.0,
        sr=16000,
        n_speaker=2,
        task='tr',
        SIR_lb=-1,
        SIR_ub=+1,
        audio_only=False,
        dynamic_mixing=True,
        google_method_mix=False,
        denoise_audio=True,
        **kwargs
    ):

        assert task in ['tr', 'cv', 'tt']

        self.n_spk = n_speaker
        self.duration = duration

        self.sr = sr

        self.dynamic_mixing = dynamic_mixing

        self.folder = Path(data_folder_path)

        self.SIR_lb = SIR_lb
        self.SIR_ub = SIR_ub

        self.denoise_audio = denoise_audio

        train_files = pd.read_csv(self.folder.joinpath('preprocessed').joinpath('train_snr15.csv'))
        train_files = train_files.drop_duplicates()['file'].to_list()

        val_files = pd.read_csv(self.folder.joinpath('preprocessed').joinpath('val_snr15.csv'))
        val_files = val_files.drop_duplicates()['file'].to_list()
        
        
        
        # test_files = pd.read_csv("/scratch/vahid/AVSpeech/test_all.csv")
        # test_files = test_files[test_files[" SNR"] > 20].iloc[:3000].drop_duplicates()['file'].to_list()
        

        # if self.denoise_audio:
        #     train_files = [self.folder.joinpath('preprocessed', 'train', f) for f in os.listdir(os.path.join(self.folder, 'preprocessed', 'train_enhanced'))]

        train_files = [self.folder.joinpath('preprocessed', 'train', f) for f in train_files]
        val_files = [self.folder.joinpath('preprocessed', 'train', f) for f in val_files]
        # test_files = [self.folder.joinpath('preprocessed', 'test', f) for f in test_files]
        
        rng = random.Random(2023)

        rng.shuffle(train_files)
        

        self.clip_list = {
            'tr': train_files,
            'cv': val_files,
            # 'tt': test_files,

        }

        self.len = {
            # 'tr': len(self.clip_list["tr"]),
            # 'cv': min(3000, len(self.clip_list["cv"])),
             'tr': 100,
            'cv': 100,
            # 'tt': min(3000, len(self.clip_list["tt"])),
        }

        self.audio_only = audio_only

        self.idx = 0

        self.google_method_mix = google_method_mix

        if self.google_method_mix:
            print('Google Method of Mixing')
            
    def get(self, idx, task):
        self.idx = idx
        # try:
        rng = random.Random( idx ) if task in ["cv", "tt"] else random.Random()

        files_path = rng.sample(self.clip_list[task], self.n_spk)
        
        


        snrs = torch.zeros(self.n_spk)

        snrs[:self.n_spk//2] = torch.tensor([rng.uniform(
            self.SIR_lb, self.SIR_ub) for _ in range(self.n_spk//2)])
        # np.random.uniform(self.SIR_lb, self.SIR_ub, self.n_spk//2) # the maximum snr would be 2 * mix_snr -> divided by 2
        snrs[self.n_spk//2:2*(self.n_spk//2)] = - \
            torch.flip(snrs[:self.n_spk//2], dims=(0, ))
        weights = 10.0**(snrs/20.0)
        

        S = torch.zeros(self.n_spk + 2, int(self.sr * self.duration))
        Video = []
        for i in range(self.n_spk):
            # data = get_AV(video_path = s_paths[i], catalog = self.catalog[task], detector = self.detector, sample_rate = self.sr, duration = self.duration, video_fps = 25, seed = idx, audio_only = self.audio_only)
            a, v = self.load_AV(
                data_folder=files_path[i],
                audio_only=self.audio_only,
                sample_rate=self.sr,
                duration=self.duration,
                video_fps=25,
                rng=rng,
            )

            audio = a[:int(self.sr * self.duration)]
            # S[i, :audio.shape[-1]] =  weights[i] * audio / RMS(audio)
            if self.google_method_mix:
                # Google's approach \Sum(a_i) + 0.3 * noise
                S[i, :audio.shape[-1]] = audio
            else:
                S[i, :audio.shape[-1]] = weights[i] * audio / RMS(audio)
            if not self.audio_only:
                Video.append(v)

        S[-2] = torch.sum(S, dim=0)

        gain = torch.max(torch.tensor([1., torch.max(torch.abs(S[-1]))])) / 0.9

        S = S / (eps_function(S) + gain)
        
        S = S.unsqueeze(-2)

        if self.audio_only:
            return {'audio': S}

        return {'audio': S, 'video': torch.stack(Video)}            


    def load_AV(self, data_folder,  audio_only, rng, sample_rate=16000, duration=3.0, video_fps=25, size=(160, 160), task='cv'):

        # rng = random.Random(seed)

        try:

            N_frames = int(duration * video_fps)

            # d = np.load(data_folder, allow_pickle = 1)['data'].item()

            # print(data_folder)

            d = torch.load(data_folder)
            # if self.denoise_audio:
            #     p = Path(data_folder)
            #     L = list(p.parts)
            #     L[-2] = 'train_enhanced'
            #     p = str(Path(*L))
            #     d['audio'] = torch.load(p)['audio']

            # if task == 'tr':
            #     d = videoAugment(d, 1, rng=rng)
            # a = torch.from_numpy( np.concatenate(d['audio'], -1) ).reshape(-1)
            a = torch.cat(d['audio'], -1).reshape(-1)

            a = Ta.Resample(16000, new_freq=sample_rate)(a)

            audio = torch.zeros(int(N_frames * sample_rate / video_fps))

            n = min(audio.shape[0], a.shape[0])
            audio[:n] = a[:n]

            if torch.isnan(audio).any():
                raise
            
            audio = audio

            if audio_only:
                return audio, None
            # faces = v
            v = d['video']
            video = torch.zeros(N_frames, *v.shape[1:], dtype=torch.uint8)
            n = min(N_frames, v.shape[0])
            video[:n] = v[:n]

            # N_frame x W x H x C  ->   N_frame x C x W x H
            video = RGB_first(video)
            video = Tv.Resize(size)(video)
            if video.max() > 127.5:
                video = (video-127.5)/128.0

            if torch.isnan(video).any():
                print(
                    f"datasamle : {data_folder} video contains nan value. Skipping ...")
                raise

            if torch.isnan(audio).any():
                print(
                    f"datasamle : {data_folder} audio contains nan value. Skipping ...")
                raise

            return audio, video
        except:
            print("\n")
            print(data_folder)
            print("\n")
            raise




class MultiChannelAVSpeech():
    def __init__(
        self,
        data_folder_path="/fs/scratch/PAA0005/vahid/Datasets/AVSpeech",
        duration=3.0,
        sr=16000,
        n_speaker=2,
        task='tr',
        SIR_lb=-5.0,  # Signal-to-interference ratio
        SIR_ub=+5.0,
        audio_only=False,
        dynamic_mixing=True,
        num_mic=4,
        ref_mic=1,
        moving_source=False,
        **kwargs
    ):

        assert task in ['tr', 'cv', 'tt']

        self.n_spk = n_speaker
        self.duration = duration

        self.sr = sr
        self.num_mic = num_mic
        self.ref_mic = ref_mic

        self.dynamic_mixing = dynamic_mixing

        self.folder = Path(data_folder_path)
        self.sub_folder = "preprocessed_moving" if moving_source else "preprocessed_stationary"

        self.SIR_lb = SIR_lb
        self.SIR_ub = SIR_ub

        train_files = os.listdir(
            f"{self.folder}/MultiChannel/{self.sub_folder}/tr/mix/")

        rng = random.Random(2023)
        rng.shuffle(train_files)

        N = int(0.9 * len(train_files))

        self.clip_list = {
            'tr': train_files[:N],
            'cv': train_files[N:],
            # 'tt': test_files,

        }

        self.len = {
            'tr': 20000,
            'cv': 3000,
            'tt': 1000,
        }

        self.audio_only = audio_only

        self.idx = 0

    def normalze_resample(self, s, old_freq):
        s = Ta.Resample(old_freq, new_freq=self.sr)(s)
        s = s/RMS(s[self.ref_mic])
        return s

    def get(self, idx, task):

        rng = random.Random(idx) if task in ["cv", "tt"] else random.Random()

        idx = rng.randint(0, len(self.clip_list[task]))

        f = self.clip_list[task][idx]

        task = "tr" if task in ["tr", "cv"] else "tt"

        spk1_clean, fs = torchaudio.load(
            f"{self.folder}/MultiChannel/{self.sub_folder}/{task}/spk1_clean/{f}")
        spk2_clean, fs = torchaudio.load(
            f"{self.folder}/MultiChannel/{self.sub_folder}/{task}/spk2_clean/{f}")

        spk1, fs = torchaudio.load(
            f"{self.folder}/MultiChannel/{self.sub_folder}/{task}/spk1/{f}")
        spk2, fs = torchaudio.load(
            f"{self.folder}/MultiChannel/{self.sub_folder}/{task}/spk2/{f}")

        noise, fs = torchaudio.load(
            f"{self.folder}/MultiChannel/{self.sub_folder}/{task}/noise/{f}")

        spk1_clean, spk2_clean = self.normalze_resample(
            spk1_clean, fs), self.normalze_resample(spk2_clean, fs)
        spk1, spk2 = self.normalze_resample(
            spk1, fs), self.normalze_resample(spk2, fs)
        noise = self.normalze_resample(noise, fs)

        self.nspk = 2

        S = torch.zeros(
            self.nspk + 2, spk1_clean.shape[0], int(self.duration * self.sr))

        S[0] = spk1_clean
        S[1] = spk2_clean

        sirs = torch.zeros(self.n_spk)

        sirs[:self.n_spk//2] = torch.tensor([rng.uniform(self.SIR_lb, self.SIR_ub) for _ in range(self.n_spk//2)])
        # np.random.uniform(self.SIR_lb, self.SIR_ub, self.n_spk//2) # the maximum snr would be 2 * mix_snr -> divided by 2
        sirs[self.n_spk//2:2*(self.n_spk//2)] = - torch.flip(sirs[:self.n_spk//2], dims=(0, ))
        weights = 10.0**(sirs/20.0) / self.nspk

        S[-2] = noise
        S[-1] = weights[0] * spk1 + weights[1] * spk2

        if self.audio_only:
            return {'audio': S}

        idx = re.search(r'(.*ENDTIME\d+\.\d+)-', f)
        video_paths = [
            f[:idx.end()-1].replace(".wav", "") + ".pt",
            f[idx.end():].replace(".wav", "") + ".pt",
        ]

        v = torch.stack([torch.load(os.path.join(self.folder, "preprocessed/train/", video_paths[i]),
                        map_location=lambda storage, loc: storage)["video"] for i in range(self.n_spk)])

        return {'audio': S, 'video': v.to(torch.float32)}




import decord as de 

de.bridge.set_bridge("torch")



class SingleChannelVoxCeleb2():
    def __init__(
        self,
        data_folder_path = "/scratch/vahid/VoxCeleb2/preprocessed_video",
        duration=3.0,
        sr=8000,
        n_speaker=2,
        task='tr',
        SIR_lb=0.0,
        SIR_ub=0.0,
        audio_only=False,
        dynamic_mixing=True,
        google_method_mix=False,
        denoise_audio=True,
        **kwargs
    ):

        assert task in ['tr', 'cv', 'tt']

        self.n_spk = n_speaker
        self.duration = duration

        self.sr = sr

        self.dynamic_mixing = dynamic_mixing

        self.folder = Path(data_folder_path)

        self.SIR_lb = SIR_lb
        self.SIR_ub = SIR_ub

        self.denoise_audio = denoise_audio

    
        train_files = glob.glob("/scratch/vahid/VoxCeleb2/preprocessed_video/train/*.mp4")
        test_files = glob.glob("/scratch/vahid/VoxCeleb2/preprocessed_video/test/*.mp4")
        
        rng = random.Random(2023)

        rng.shuffle(train_files)
        
        N = int(0.8 * len(train_files))

        self.clip_list = {
            'tr': train_files[:N],
            'cv': train_files[N:],
            'tt': test_files,

        }

        self.len = {
            'tr': 20000,
            'cv': min(3000, len(self.clip_list["cv"])),
            'tt': 3000,
        }

        self.audio_only = audio_only

        self.idx = 0

        self.google_method_mix = google_method_mix

        if self.google_method_mix:
            print('Google Method of Mixing')
            
    def get(self, idx, task):
        self.idx = idx
        # try:
        rng = random.Random( idx ) if task in ["cv", "tt"] else random.Random()

        files_path = rng.sample(self.clip_list[task], self.n_spk)

        snrs = torch.zeros(self.n_spk)

        snrs[:self.n_spk//2] = torch.tensor([rng.uniform(
            self.SIR_lb, self.SIR_ub) for _ in range(self.n_spk//2)])
        # np.random.uniform(self.SIR_lb, self.SIR_ub, self.n_spk//2) # the maximum snr would be 2 * mix_snr -> divided by 2
        snrs[self.n_spk//2:2*(self.n_spk//2)] = - \
            torch.flip(snrs[:self.n_spk//2], dims=(0, ))
        weights = 10.0**(snrs/20.0)

        S = torch.zeros(self.n_spk + 2, int(self.sr * self.duration))
        Video = []
        for i in range(self.n_spk):
            # data = get_AV(video_path = s_paths[i], catalog = self.catalog[task], detector = self.detector, sample_rate = self.sr, duration = self.duration, video_fps = 25, seed = idx, audio_only = self.audio_only)
            a, v = self.load_AV(
                data_folder=files_path[i],
                audio_only=self.audio_only,
                sample_rate=self.sr,
                duration=self.duration,
                video_fps=25,
                rng=rng,
            )

            audio = a[:int(self.sr * self.duration)]
            # S[i, :audio.shape[-1]] =  weights[i] * audio / RMS(audio)
            if self.google_method_mix:
                # Google's approach \Sum(a_i) + 0.3 * noise
                S[i, :audio.shape[-1]] = audio
            else:
                S[i, :audio.shape[-1]] = weights[i] * audio / RMS(audio)
            if not self.audio_only:
                Video.append(v)

        S[-2] = torch.sum(S, dim=0)

        gain = torch.max(torch.tensor([1., torch.max(torch.abs(S[-1]))])) / 0.9

        S = S / (eps_function(S) + gain)
        
        S = S.unsqueeze(-2)

        if self.audio_only:
            return {'audio': S}

        return {'audio': S, 'video': torch.stack(Video)}            


    def load_AV(self, data_folder,  audio_only, rng, sample_rate=16000, duration=3.0, video_fps=25, size=(160, 160), task='cv'):

        # rng = random.Random(seed)

        try:

            N_frames = int(duration * video_fps)

            a, v = de.AVReader(data_folder, sample_rate=sample_rate)[:N_frames]
            
           
            
            
            a = torch.cat(a, -1).reshape(-1)


            audio = torch.zeros(int(N_frames * sample_rate / video_fps))

            n = min(audio.shape[0], a.shape[0])
            audio[:n] = a[:n]

            if torch.isnan(audio).any():
                raise
            
            audio = audio

            if audio_only:
                return audio, None
            # faces = v
            # v = d['video']
            video = torch.zeros(N_frames, *v.shape[1:], dtype=torch.uint8)
            n = min(N_frames, v.shape[0])
            video[:n] = v[:n]

            # N_frame x W x H x C  ->   N_frame x C x W x H
            video = RGB_first(video)
            video = Tv.Resize(size)(video)
            if video.max() > 127.5:
                video = (video-127.5)/128.0

            if torch.isnan(video).any():
                print(
                    f"datasamle : {data_folder} video contains nan value. Skipping ...")
                raise

            if torch.isnan(audio).any():
                print(
                    f"datasamle : {data_folder} audio contains nan value. Skipping ...")
                raise

            return audio, video
        except:
            print("\n")
            print(data_folder)
            print("\n")
            raise





DatasetsWrappers = {
    'wsj': WSJWrapper,
    'lrs3': LRS3Wrapper,    
    'audioset': AudioSet,
    'wham': WHAM,
    
    'singlechannelvoxceleb2': SingleChannelVoxCeleb2,
    'singlechannelavspeech': SingleChannelAVSpeech,
    'multichannelavspeech': MultiChannelAVSpeech,
    'easycom': MultiChannelEasyCom,
}


def check_not_silent(audio, eps = 1e-5):
    # some youtube videos are blocked by youtube
    assert(torch.any(torch.abs(audio).max(-1)[0] > eps).item())

class AudioVisualDataset(Dataset):
    def __init__(
        self,
        audio_dataset_path,
        noise_dataset_path=None,
        audio_dataset: str = 'WSJ',
        noise_dataset: str = 'Audioset',
        n_speakers=2,
        sampling_rate=16000,
        duration=3.0,
        task='tr',
        audio_only=True,
        add_noise=False,
        noise_db_bounds=(-2.5, 2.5),
        noise_db_bounds_test=(-2.5, 2.5),
        dynamic_mixing=False,
        ref_mic=1,
        mic_idx=[-2, -1],
        moving_source=False,
        denoise_audio = False,
    ):
        # duration = int(np.ceil((sampling_rate * duration)/(512)) * 512)/sampling_rate # TCNN

        self.duration = duration
        self.ref_mic = ref_mic
        self.mic_idx = mic_idx

        self.dynamic_mixing = dynamic_mixing

        self.dataset_path_wrapper = DatasetsWrappers[audio_dataset.lower()](data_folder_path=audio_dataset_path, sr=sampling_rate, n_speaker=n_speakers, duration=duration, audio_only=audio_only, dynamic_mixing=dynamic_mixing, moving_source=moving_source, denoise_audio = denoise_audio)
        
        if add_noise:
            self.noise_path_wrapper = DatasetsWrappers[ noise_dataset.lower() ](data_folder_path = noise_dataset_path, replace = False, sr = sampling_rate, duration = duration, normalize = 'none', dynamic_mixing = dynamic_mixing) 
        
        self.task = task
        self.audio_only = audio_only
        self.add_noise = add_noise
        self.sampling_rate = sampling_rate
        self.n_speaker = n_speakers

        if task.lower() in ['cv', 'tt']:
            self.noise_db_bounds = noise_db_bounds_test
        else:
            self.noise_db_bounds = noise_db_bounds

        self.task = task
        
    def change_noise_db_bounds(self, noise_db_bounds):
        self.noise_db_bounds = noise_db_bounds
        
        print("noise_db_bounds is changed to : ", noise_db_bounds)
        

    def __len__(self):
        return self.dataset_path_wrapper.len[self.task]
    

    def __getitem__(self, idx):
        task = self.task

        rng = random.Random(idx) if (not self.dynamic_mixing or (task in ["tt", "cv"])) else random.Random()

        try:

        # Audio

            # (n_speaker+1) x 1 x n_samples
            d = self.dataset_path_wrapper.get(idx, task=self.task)
            audio = d['audio'][:, self.mic_idx, :]
            
            torch.any(torch.abs(audio).max(-1)[0] < 1E-5)
            
            check_not_silent(audio, eps = 1e-5)
            
            # audio: spk1_clean, spk2_clean, mixed_clean_speech, noise,
            S = torch.zeros(audio.shape[0], len(self.mic_idx), int(self.duration * self.sampling_rate))

            if audio.shape[-1] >= S.shape[-1]:

                N1 = rng.randint(0, audio.shape[-1] - S.shape[-1])

                N2 = N1 + S.shape[-1]

                S = audio[..., N1:N2].reshape(S.shape)
            else:
                S[..., :audio.shape[-1]] = audio.reshape(S[..., :audio.shape[-1]].shape)

            if self.add_noise:
                # spk1, spk2, ..., spkn, noisy_mix_speech'

                rng = random.Random() if (self.dynamic_mixing & (random.random() < 0.7) & (task == 'tr')) else random.Random(idx)
                
                dB = rng.uniform(self.noise_db_bounds[0], self.noise_db_bounds[1])
                
                n = S[-1] if torch.sum(torch.abs(S[-1])) > 0.001 else self.noise_path_wrapper.get(idx, task = self.task, rng = rng)['audio']
                # alpha = torch.sqrt(torch.sum(S[-2][self.ref_mic]**2)/(torch.sum(n[self.ref_mic]**2)*10**(dB/10)))
                
                S[-1] = mix_s_n(s = S[-2], n = n, db = dB, ref_mic = self.ref_mic)
                # mix = S[-2] + alpha * n
            else:
                S[-1] = S[-2]
                
                
            


        

            gain = torch.max(torch.tensor([1., torch.max(torch.abs(S))])) / 0.9

            a = S.clone() / gain
            
        
       

            if self.audio_only:
                return {'audio': a}
            else:
                v = d['video'].clone()
                # if(v.max() > 10):
            
                v = 2*(v - v.min()) / ( v.max() - v.min() ) - 1 # convert to [-1, 1]
                # mix_mean_ = torch.mean(v)
                # mix_std_ = torch.std(v, dim=(-3, -2, -1), keepdim=True)  # [75, 1, 1, 1]  
                # v = (v - mix_mean_)/mix_std_.
                            

                return {'audio': a, 'video': v}

        except Exception as err:
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    
            print("Error in Dataset : ", idx, ' -  ', err)
            # TODO: for avspeech some videos have not downloaded properly
            idx = np.random.randint(
                0, self.dataset_path_wrapper.len[self.task])
            return self.__getitem__(idx)


class AudioVisualDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
    ):

        super(AudioVisualDataModule, self).__init__()

        # print("  ***  Validation and Test are moving source.  ***  ")

        self.train_dataset = AudioVisualDataset(
            audio_dataset=args.audio_dataset,
            noise_dataset=args.noise_dataset,
            audio_dataset_path=args.audio_dataset_path,
            noise_dataset_path=args.noise_dataset_path,
            n_speakers=args.n_speakers,
            sampling_rate=args.sampling_rate,
            duration=args.duration,
            task='tr',
            audio_only=args.audio_only,
            add_noise=args.add_noise,
            noise_db_bounds=args.noise_db_bounds,
            noise_db_bounds_test=args.noise_db_bounds_test,
            dynamic_mixing=args.dynamic_mixing,
            ref_mic=args.ref_mic,
            mic_idx=args.mic_idx,
            moving_source=args.moving_source,
            denoise_audio = args.denoise_audio,
        )
        self.valid_dataset = AudioVisualDataset(
            audio_dataset=args.audio_dataset,
            noise_dataset=args.noise_dataset,
            audio_dataset_path=args.audio_dataset_path,
            noise_dataset_path=args.noise_dataset_path,
            n_speakers=args.n_speakers,
            sampling_rate=args.sampling_rate,
            duration=args.duration,
            task='cv',
            audio_only=args.audio_only,
            add_noise=args.add_noise,
            noise_db_bounds=args.noise_db_bounds,
            noise_db_bounds_test=args.noise_db_bounds_test,
            dynamic_mixing=False,
            ref_mic=args.ref_mic,
            mic_idx=args.mic_idx,
            moving_source=args.moving_source,
            denoise_audio = args.denoise_audio,

        )
        self.test_dataset = AudioVisualDataset(
            audio_dataset=args.audio_dataset,
            noise_dataset=args.noise_dataset,
            audio_dataset_path=args.audio_dataset_path,
            noise_dataset_path=args.noise_dataset_path,
            n_speakers=args.n_speakers,
            sampling_rate=args.sampling_rate,
            duration=args.duration,
            task='tt',
            audio_only=args.audio_only,
            add_noise=args.add_noise,
            noise_db_bounds=args.noise_db_bounds,
            noise_db_bounds_test=args.noise_db_bounds_test,
            dynamic_mixing=False,
            ref_mic=args.ref_mic,
            mic_idx=args.mic_idx,
            moving_source=args.moving_source,
            denoise_audio = args.denoise_audio,

        )

        self.batch_size = args.batch_size
        self.num_dataset_workers = args.num_dataset_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule")
        parser.add_argument("--audio_dataset", type=str, default='SingleChannelAVSpeech')
        parser.add_argument("--noise_dataset", type=str, default='WHAM')
        parser.add_argument("--audio_dataset_path", type=str, default='/scratch/vahid/AVSpeech/')
        parser.add_argument("--noise_dataset_path", type=str, default='./')
        parser.add_argument("--n_speakers", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--duration", type=float, default=3.0)
        parser.add_argument("--sampling_rate", type=int, default=16000)
        parser.add_argument("--num_dataset_workers",
                            type=int, default=os.cpu_count())
        parser.add_argument("--dynamic_mixing", type=str2bool, default='True')
        parser.add_argument("--add_noise", type=str2bool, default='False')
        parser.add_argument("--moving_source", type=str2bool, default='False')
        parser.add_argument("--ref_mic", type=int, default=0)
        parser.add_argument("--mic_idx", type=str2list, default='[0]')

        return parent_parser

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_dataset_workers,
            drop_last=True,
            pin_memory=True,  # TODO
            persistent_workers=True if self.num_dataset_workers > 0 else False,  # TODO
            shuffle=False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_dataset_workers,
            pin_memory=False,
            drop_last=True,
            shuffle=False,
            persistent_workers=False if self.num_dataset_workers > 0 else False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_dataset_workers,
            shuffle=False,
        )


if __name__ == "__main__":

    # check
    pass
