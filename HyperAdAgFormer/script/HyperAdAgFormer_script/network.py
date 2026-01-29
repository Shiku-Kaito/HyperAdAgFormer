import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
from torch.nn.utils.rnn import pad_sequence
from HyperAdAgFormer_script.hypernet_token import HyperLinearLayer_token
from HyperAdAgFormer_script.hypernet_classifier import HyperLinearLayer as HyperLinearLayer_classifier

class TableMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_extractor = Mlp(in_features=20, hidden_features=1024, out_features=512)

        self.head1 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, bags, table, len_list):
        tab_feat = self.feature_extractor(table)

        if tab_feat.shape[0]!=1:
            x1 = self.head1(tab_feat).squeeze()
        else:
            x1 = self.head1(tab_feat)[0,:]
        x1 = self.sigmoid(x1)

        return {"y_bag": x1}

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, num_classes=20):
        super().__init__()
        # self.num_classes = num_classes
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x num_heads x N x N

        attn = attn.softmax(dim=-1)
        weights = attn

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, weights


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_classes=20,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_classes=num_classes,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        o, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights


class HyperAdAgFormer(nn.Module):
    def __init__(
        self,
        args=None, 
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.2,
        attn_drop_rate=0.2,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        train_loader=None
    ):
        super().__init__()
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential()
        
        self.table_feature_extractor = TableMLP(args=args)
        self.table_feature_extractor = self.table_feature_extractor.feature_extractor.to(args.device)
        
        self.fc = nn.Linear(512, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_classes=num_classes,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head1 = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.hypernet = HyperLinearLayer_token(args=args, in_features=embed_dim, out_features=embed_dim, embedding_model=self.table_feature_extractor, embedding_output_size=embed_dim, train_loader=train_loader)
        self.hypernet_classifier = HyperLinearLayer_classifier(args=args, in_features=embed_dim, out_features=1, embedding_model=self.table_feature_extractor, embedding_output_size=embed_dim, train_loader=train_loader)
        
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim))

    def forward(self, bags, table, len_list):
        # ----- Feature extract from instances -----
        ins_feat = self.feature_extractor(bags)
        ini_idx = 0
        ins_feats = []
        for length in len_list:
            ins_feats.append(ins_feat[ini_idx : ini_idx + length])
            ini_idx += length
        x = pad_sequence(ins_feats, batch_first=True, padding_value=0)

        # ----- Generate "Tabular-conditioned Transformation Parameter (TCTP)" and "Bag-level Classifier" -----
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        additive_cls_token = self.hypernet(table)[0].unsqueeze(1)
        cls_token = cls_token + additive_cls_token
        x = torch.cat((cls_token, x), dim=1)

        # ----- Adaptive aggregation -----
        attn_weights = []
        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights = (weights_i.sum(dim=1)/ 8)

        x = self.norm(x)

        # ----- Adaptive classification -----
        if x.shape[0]!=1:
            x1 = self.hypernet_classifier([x[:, 0, :], table]).squeeze()
        else:
            x1 = self.hypernet_classifier([x[:, 0, :], table])[0,0,:]
        x1 = self.sigmoid(x1)


        attn_weights_list = []
        for idx, length in enumerate(len_list):
            attn_weights_list.append(attn_weights[idx][0][:length].cpu().detach().numpy())
            
        return {"y_bag": x1, "atten_weight":attn_weights_list, "bag_feat": x[:, 0, :]}
    