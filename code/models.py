import open_clip
import torch
from torch import nn
from torchvision import transforms as T

from config import LORA_ALPHA, LORA_RANK

BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
FEATURE_DIM = {"dinov2": 768, "biomedclip": 512}
MEAN_STD = {
    "dinov2": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "biomedclip": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
}


def load_backbone(name, device):
    if name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    else:
        clip, _ = open_clip.create_model_from_pretrained(BIOMEDCLIP)
        model = clip.visual  # only need the image encoder
    return model.to(device).eval()


def get_transform(name):
    mean, std = MEAN_STD[name]
    return T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])


# q, k, v are one fused layer in both models, so peft would also train k.
# same idea as _LoRA_qkv_timm in github.com/JamesQFreeman/LoRA-ViT
class LoRAqkv(nn.Module):
    def __init__(self, qkv, rank=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.scale = alpha / rank
        self.a_q = nn.Linear(self.dim, rank, bias=False)
        self.b_q = nn.Linear(rank, self.dim, bias=False)
        self.a_v = nn.Linear(self.dim, rank, bias=False)
        self.b_v = nn.Linear(rank, self.dim, bias=False)
        # b starts at 0 so training starts from the frozen model
        nn.init.zeros_(self.b_q.weight)
        nn.init.zeros_(self.b_v.weight)

    def forward(self, x):
        out = self.qkv(x)
        out[..., :self.dim] += self.scale * self.b_q(self.a_q(x))
        out[..., -self.dim:] += self.scale * self.b_v(self.a_v(x))
        return out


class LoRAModel(nn.Module):
    def __init__(self, name, backbone):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        blocks = backbone.blocks if name == "dinov2" else backbone.trunk.blocks
        for block in blocks:
            block.attn.qkv = LoRAqkv(block.attn.qkv)
        self.head = nn.Linear(FEATURE_DIM[name], 1)

    def forward(self, x):
        feats = self.backbone(x).float()
        with torch.autocast(x.device.type, enabled=False):  # keep logits in fp32
            logit = self.head(feats).squeeze(1)
        return logit, feats

    def trainable(self):
        return {k: v.detach().cpu().clone() for k, v in self.named_parameters() if v.requires_grad}
