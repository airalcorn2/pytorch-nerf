import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(True)

    def forward(self, x):
        # Extract feature pyramid from image. See Section 4.1., Section B.1 in the
        # Supplementary Materials, and: https://github.com/sxyu/pixel-nerf/blob/master/src/model/encoder.py.
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        feats1 = self.resnet.relu(x)

        feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))
        feats3 = self.resnet.layer2(feats2)
        feats4 = self.resnet.layer3(feats3)

        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i], latent_sz, mode="bilinear", align_corners=True
            )

        latents = torch.cat(latents, dim=1)
        return latents
