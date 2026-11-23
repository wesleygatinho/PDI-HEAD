"""As tres arquiteturas de segmentacao com interface unica.

  unet        U-Net com encoder ResNet-34 (ImageNet)           -> CNN (baseline)
  segformer   SegFormer MiT-B2 (ImageNet-1k)                   -> Transformer hierarquico
  mask2former Mask2Former Swin-T (ADE20K semantic)             -> Transformer com mascaras

Todas recebem 1 canal (o campo fisico) e devolvem a probabilidade da classe
"estrutura" em 128x128. A primeira convolucao pre-treinada (3 canais) e adaptada
para 1 canal somando os pesos sobre os canais de cor (mesma estrategia do
segmentation_models_pytorch). Por padrao (input_scale = 1) todas recebem o campo
na resolucao nativa 128x128; SegFormer e Mask2Former predizem em 1/4 da resolucao
(32x32) e a saida e interpolada bilinearmente para 128x128.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# revisao com model.safetensors (conversao oficial do Hugging Face); evita torch.load de pickle
SAFE_REV = "refs/pr/2"

ARCH_NAMES = {"unet": "U-Net", "segformer": "SegFormer", "mask2former": "Mask2Former"}
ARCH_BACKBONES = {
    "unet": "ResNet-34 encoder (ImageNet)",
    "segformer": "MiT-B2 (ImageNet-1k)",
    "mask2former": "Swin-T (ADE20K)",
}


def _adapt_conv_to_1ch(conv: nn.Conv2d) -> nn.Conv2d:
    new = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def dice_bce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum(dim=(1, 2))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2)) + y.sum(dim=(1, 2)) + 1)
    return bce + dice.mean()


class UNetSeg(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None,
                            in_channels=1, classes=1)

    def logits(self, x):  # x [B,1,128,128]
        return self.net(x)[:, 0]

    def loss(self, x, y):
        return dice_bce_loss(self.logits(x), y)

    @torch.no_grad()
    def proba(self, x):
        return torch.sigmoid(self.logits(x).float())


class SegFormerSeg(nn.Module):
    def __init__(self, pretrained: bool = True, scale: int = 1):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", num_labels=1, revision=SAFE_REV, use_safetensors=True)
        else:
            self.net = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained("nvidia/mit-b2", num_labels=1))
        pe = self.net.segformer.encoder.patch_embeddings[0]
        pe.proj = _adapt_conv_to_1ch(pe.proj)
        self.net.config.num_channels = 1
        self.scale = scale

    def logits(self, x):
        h, w = x.shape[-2:]
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        lg = self.net(pixel_values=x).logits  # [B,1,H*s/4,W*s/4]
        return F.interpolate(lg, size=(h, w), mode="bilinear", align_corners=False)[:, 0]

    def loss(self, x, y):
        return dice_bce_loss(self.logits(x), y)

    @torch.no_grad()
    def proba(self, x):
        return torch.sigmoid(self.logits(x).float())


class Mask2FormerSeg(nn.Module):
    CKPT = "facebook/mask2former-swin-tiny-ade-semantic"

    def __init__(self, pretrained: bool = True, scale: int = 1, aux_loss: bool = False):
        super().__init__()
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

        kw = dict(num_labels=2, id2label={0: "background", 1: "structure"},
                  label2id={"background": 0, "structure": 1})
        if pretrained:
            self.net = Mask2FormerForUniversalSegmentation.from_pretrained(
                self.CKPT, ignore_mismatched_sizes=True, revision=SAFE_REV, use_safetensors=True, **kw)
        else:
            self.net = Mask2FormerForUniversalSegmentation(Mask2FormerConfig.from_pretrained(self.CKPT, **kw))
        pe = self.net.model.pixel_level_module.encoder.embeddings.patch_embeddings
        pe.projection = _adapt_conv_to_1ch(pe.projection)
        pe.num_channels = 1
        self.scale = scale
        if not aux_loss:
            # Sem supervisao profunda: a loss (casamento hungaro + CE/BCE/Dice) e calculada so na saida
            # final do decodificador, nao nas 9 camadas intermediarias. A arquitetura e a inferencia
            # nao mudam; o custo de treino cai ~3,4x (o casamento por amostra e limitado por CPU).
            self.net.get_auxiliary_logits = lambda *a, **k: []
        self.aux_loss = aux_loss

    def _up(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return x

    def loss(self, x, y):
        xs = self._up(x)
        ys = F.interpolate(y[:, None], size=xs.shape[-2:], mode="nearest")[:, 0]
        mask_labels, class_labels = [], []
        for m in ys:
            ms, cs = [], []
            for c in (0, 1):
                mc = (m == c).float() if c == 1 else (m == 0).float()
                if mc.sum() > 0:
                    ms.append(mc); cs.append(c)
            mask_labels.append(torch.stack(ms))
            class_labels.append(torch.tensor(cs, device=x.device, dtype=torch.long))
        out = self.net(pixel_values=xs, mask_labels=mask_labels, class_labels=class_labels)
        return out.loss

    @torch.no_grad()
    def proba(self, x):
        h, w = x.shape[-2:]
        out = self.net(pixel_values=self._up(x))
        cls = out.class_queries_logits.float().softmax(-1)[..., :-1]      # [B,Q,2] (sem "no-object")
        msk = out.masks_queries_logits.float().sigmoid()                   # [B,Q,h',w']
        sem = torch.einsum("bqc,bqhw->bchw", cls, msk)                     # [B,2,h',w']
        p1 = sem[:, 1] / (sem.sum(1) + 1e-6)
        return F.interpolate(p1[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0].clamp(0, 1)


def build_model(arch: str, pretrained: bool = True, scale: int = 1, m2f_aux_loss: bool = False) -> nn.Module:
    if arch == "unet":
        return UNetSeg(pretrained)
    if arch == "segformer":
        return SegFormerSeg(pretrained, scale)
    if arch == "mask2former":
        return Mask2FormerSeg(pretrained, scale, m2f_aux_loss)
    raise ValueError(arch)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
