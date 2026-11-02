import torch
from torchvision import transforms as T

BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# normalization each model was pretrained with
MEAN_STD = {
    "dinov2": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
    "biomedclip": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),  # clip
}


def load_backbone(name, device):
    """Returns (encode, transform). encode(batch) gives one feature vector per image."""
    if name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        encode = model  # forward() returns the cls token, 768-d
    elif name == "biomedclip":
        import open_clip
        model, _ = open_clip.create_model_from_pretrained(BIOMEDCLIP)
        encode = model.encode_image  # image embedding, 512-d
    else:
        raise ValueError(name)
    model.to(device).eval()

    # shorter side to 224 then center crop, same as the paper (and what MedMNIST-C expects)
    mean, std = MEAN_STD[name]
    transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])
    return encode, transform
