import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, device='cuda'):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output

class CustomArcFaceModel(nn.Module):
    def __init__(self, num_classes, device='cuda'):
        super(CustomArcFaceModel, self).__init__()
        self.device = device
        self.backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        self.arc_margin_product = ArcMarginProduct(2048, num_classes, device=self.device)
        nn.init.kaiming_normal_(self.arc_margin_product.weight)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        features = F.normalize(features)
        features = features.view(features.size(0), -1)
        if labels is not None:
            logits = self.arc_margin_product(features, labels)
            return logits

        return features

    def cosine_similarity(self, x1, x2):
        return torch.sum(x1 * x2) / (torch.norm(x1) * torch.norm(x2))

    def find_most_similar_celebrity(self, user_face_embedding, celebrity_face_embeddings):
        max_similarity = -1
        most_similar_celebrity_index = -1

        for i, celebrity_embedding in enumerate(celebrity_face_embeddings):
            similarity = self.cosine_similarity(user_face_embedding, celebrity_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_celebrity_index = i

        return most_similar_celebrity_index, max_similarity

