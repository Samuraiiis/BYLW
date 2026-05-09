import torch
import torch.nn as nn

# ===============================================================
# 🧩 CBAM 模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x * sa


# ===============================================================
# 🧩 边界细化模块
class BoundaryRefineHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.refine(x)


# ===============================================================
# 🧠 模块注入函数
def inject_custom_modules(model):
    """
    遍历 YOLO 模型结构，将 CBAM 与 BRH 注入到指定层。
    """
    for name, layer in model.model.named_children():
        if "neck" in name.lower():
            print(f"✅ 在 {name} 中添加 CBAM 模块 ...")
            channels = getattr(layer, "cv3", None)
            channels = channels.conv.out_channels if channels else 256
            layer.cbam = CBAM(channels)
        if "seg" in name.lower() or "head" in name.lower():
            print(f"✅ 在 {name} 中添加 BoundaryRefineHead 模块 ...")
            layer.refine_head = BoundaryRefineHead(256)
    print("✅ 模块注入完成！")
    return model
