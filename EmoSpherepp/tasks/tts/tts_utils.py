import importlib
import glob
import os
import torch

from data_gen.tts.base_binarizer import BaseBinarizer
from data_gen.tts.base_preprocess import BasePreprocessor
from data_gen.tts.txt_processors.base_text_processor import get_txt_processor_cls
from utils.commons.hparams import hparams, set_hparams


class VocoderInfer:
    def __init__(self, hparams):
        vocoder_name = hparams.get("vocoder", "BigVGAN")
        config_path = hparams.get("vocoder_config")
        if config_path is None:
            if vocoder_name.lower() == "bigvgan":
                config_path = "configs/models/vocoder/bigvgan_16k.yaml"
            elif vocoder_name.lower() == "hifigan":
                config_path = "configs/models/vocoder/hifigan.yaml"
            else:
                raise KeyError("vocoder_config")
        self.config = config = set_hparams(config_path, global_hparams=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vocoder_cls = hparams.get("vocoder_cls")
        if vocoder_cls is None:
            if vocoder_name.lower() == "bigvgan":
                vocoder_cls = "models.vocoder.bigvgan.BigVGAN"
            elif vocoder_name.lower() == "hifigan":
                vocoder_cls = "models.vocoder.hifigan.HiFiGAN"
            else:
                raise KeyError("vocoder_cls")
        pkg = ".".join(vocoder_cls.split(".")[:-1])
        cls_name = vocoder_cls.split(".")[-1]
        vocoder = getattr(importlib.import_module(pkg), cls_name)
        self.model = vocoder(config)

        ckpt_path = hparams["vocoder_ckpt"]
        if os.path.isdir(ckpt_path):
            ckpt_candidates = sorted(
                glob.glob(os.path.join(ckpt_path, "g_*")),
                key=lambda x: os.path.basename(x),
            )
            if len(ckpt_candidates) == 0:
                raise FileNotFoundError(
                    f"No vocoder checkpoint found under directory: {ckpt_path}"
                )
            ckpt_path = ckpt_candidates[-1]
        checkpoint_dict = torch.load(ckpt_path, map_location=self.device)

        if isinstance(checkpoint_dict, dict) and "generator" in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict["generator"])
        else:
            self.model.load_state_dict(checkpoint_dict)
        self.model.to(self.device)
        self.model.eval()

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).to(device)
            c = c.transpose(2, 1)
            y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out


def parse_dataset_configs():
    max_tokens = hparams["max_tokens"]
    max_sentences = hparams["max_sentences"]
    max_valid_tokens = hparams["max_valid_tokens"]
    if max_valid_tokens == -1:
        hparams["max_valid_tokens"] = max_valid_tokens = max_tokens
    max_valid_sentences = hparams["max_valid_sentences"]
    if max_valid_sentences == -1:
        hparams["max_valid_sentences"] = max_valid_sentences = max_sentences
    return max_tokens, max_sentences, max_valid_tokens, max_valid_sentences


def parse_mel_losses():
    mel_losses = hparams.get("mel_losses", "l1").split("|")
    loss_and_lambda = {}
    for i, l in enumerate(mel_losses):
        if l == "":
            continue
        if ":" in l:
            l, lbd = l.split(":")
            lbd = float(lbd)
        else:
            lbd = 1.0
        loss_and_lambda[l] = lbd
    print("| Mel losses:", loss_and_lambda)
    return loss_and_lambda


def load_data_preprocessor():
    preprocess_cls = hparams["preprocess_cls"]
    pkg = ".".join(preprocess_cls.split(".")[:-1])
    cls_name = preprocess_cls.split(".")[-1]
    preprocessor: BasePreprocessor = getattr(importlib.import_module(pkg), cls_name)()
    preprocess_args = {}
    preprocess_args.update(hparams["preprocess_args"])
    return preprocessor, preprocess_args


def load_data_binarizer():
    binarizer_cls = hparams["binarizer_cls"]
    pkg = ".".join(binarizer_cls.split(".")[:-1])
    cls_name = binarizer_cls.split(".")[-1]
    binarizer: BaseBinarizer = getattr(importlib.import_module(pkg), cls_name)()
    binarization_args = {}
    binarization_args.update(hparams["binarization_args"])
    return binarizer, binarization_args
