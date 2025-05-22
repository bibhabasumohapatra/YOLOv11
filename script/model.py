import torch.nn as nn
import torch
from script.layers import *

class DetectionModel(nn.Module):
    def __init__(self, depth=0.5, width=0.25, max_channels=1024, nc=80):
        """
        Initialize the DetectionModel.

        Args:
            depth (float): Depth multiplier for scaling the model.
            width (float): Width multiplier for scaling the model.
            max_channels (int): Maximum number of channels.
            nc (int): Number of classes for detection.
        """
        super().__init__()

        # Backbone
        self.model = nn.Sequential(
            Conv(3, int(64 * width), 3, 2),  # 0-P1/2
            Conv(int(64 * width), int(128 * width), 3, 2),  # 1-P2/4
            C3k2(int(128 * width), int(256 * width), n=int(2 * depth), c3k=False, e=0.25),
            Conv(int(256 * width), int(256 * width), 3, 2),  # 3-P3/8
            C3k2(int(256 * width), int(512 * width), n=int(2 * depth), c3k=False, e=0.25),
            ##
            Conv(int(512 * width), int(512 * width), 3, 2),  # 5-P4/16
            C3k2(int(512 * width), int(512 * width), n=int(2 * depth), c3k=True),
            ##
            Conv(int(512 * width), int(1024 * width), 3, 2),  # 7-P5/32
            C3k2(int(1024 * width), int(1024 * width), n=int(2 * depth), c3k=True),
            SPPF(int(1024 * width), int(1024 * width), k=5),  # 9
            C2PSA(int(1024 * width), int(1024 * width), n=int(2 * depth)),
        #     )  # 10
        #     # Head
        # self.neck = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # -1 
            Concat(dimension=1),  # cat backbone P4
            C3k2(int(1536 * width), int(512 * width), n=int(2 * depth), c3k=False),  # 13

            nn.Upsample(scale_factor=2, mode="nearest"),  # -1
            Concat(dimension=1),  # cat backbone P3
            C3k2(int(1024 * width), int(256 * width), n=int(2 * depth), c3k=False),  # 16 (P3/8-small)

            Conv(int(256 * width), int(256 * width), 3, 2),  # -1
            Concat(dimension=1),  # cat head P4
            C3k2(int(768 * width), int(512 * width), n=int(2 * depth), c3k=False),  # 19 (P4/16-medium)

            Conv(int(512 * width), int(512 * width), 3, 2),  # -1
            Concat(dimension=1),  # cat head P5
            C3k2(int(1536 * width), int(1024 * width), n=int(2 * depth), c3k=True),  # 22 (P5/32-large)
            Detect(nc=nc, ch=[int(256 * width), int(512 * width), int(1024 * width)])
        )

        # self.head = Detect(nc=nc, ch=[int(256 * width), int(512 * width), int(1024 * width)])  # Detect(P3, P4, P5)

    def forward(self, x):
        """
        Forward pass through the DetectionModel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        for n, layer in enumerate(self.model):
            if n==4:
                x = layer(x)
                p4 = x.clone() 
            elif n==6:
                x = layer(x)
                p6 = x.clone()
            elif n==10:
                x = layer(x)
                p10 = x.clone()
                
            elif n==12 and isinstance(layer, Concat):
                x = layer((x, p6))

            elif n==15 and isinstance(layer, Concat):
                x = layer((x, p4))

            elif n==18 and isinstance(layer, Concat):
                x = layer((x, p13))
                
            elif n==21 and isinstance(layer, Concat):
                x = layer((x, p10))

            elif n==13 and isinstance(layer, C3k2):
                x = layer(x)
                p13 = x.clone()

            elif n==16 and isinstance(layer, C3k2):
                x = layer(x)
                p16 = x.clone()

            elif n==19 and isinstance(layer, C3k2):
                x = layer(x)
                p19 = x.clone()

            ## head
            elif n==23 and isinstance(layer, Detect):
                # print(x.shape, p19.shape, p16.shape)
                img_dummy = torch.zeros(1, 3, 256, 256)
                layer.stride = torch.tensor([ 8., 16., 32.])
                out = layer([p16, p19, x])
                out = layer.postprocess(out[0].permute(0, 2, 1))
            else:
                x = layer(x)
        
        return out

if __name__ == "__main__":
    # Example usage
    model = DetectionModel()
    # print(model)
    weights = torch.load('/home/bibhabasum/projects/IIIT/ultralytics/model_state_dict.pth') # Load weights

    model.load_state_dict(weights, strict=True)  # Load model weights

      # Example input


# self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
# x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
# box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

# a, b = self.dfl(box).chunk(2, 1)
# a = self.anchors.unsqueeze(0) - a
# b = self.anchors.unsqueeze(0) + b
# box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

# return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)