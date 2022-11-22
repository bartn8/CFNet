import numpy as np
import cv2
import torch

import torchvision.transforms as transforms

import importlib
cfnet = importlib.import_module("thirdparty.CFNet.models.cfnet")

class CFNetBlock:
    def __init__(self, max_disparity = 192, device = "cpu", verbose=False):
        self.logName = "CFNet Block"
        self.verbose = verbose
        self.max_disparity = max_disparity
        self.device = device
        self.disposed = False
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.processed =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def log(self, x):
        if self.verbose:
            print(f"{self.logName}: {x}")

    def build_model(self):
        if self.disposed:
            self.log("Session disposed!")
            return

        self.log(f"Building Model...")
        self.model = cfnet.CFNet(self.max_disparity)
        self.model = torch.nn.DataParallel(self.model)  


    def load(self, model_path):
        # load the checkpoint file specified by model_path.loadckpt
        print("loading model {}".format(model_path))
        state_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict['model'])

    def dispose(self):
        if not self.disposed:
            del self.model
            self.disposed = True

    def _conv_image(self, img):
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        h,w = img.shape[:2]
        img = self.processed(img).numpy()

        top_pad = 32 - (h % 32)
        right_pad = 32 - (w % 32)
        assert top_pad > 0 and right_pad > 0
        # pad images
        img = np.lib.pad(img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

        self.log(f"Original shape: {(h,w)}, padding: {(top_pad, right_pad)}, new shape: {img.shape}")

        return torch.from_numpy(np.expand_dims(img, axis=0)).to(self.device), top_pad, right_pad

    def test(self, left_vpp, right_vpp):
        #Input conversion
        left_vpp, toppad, rightpad = self._conv_image(left_vpp)
        right_vpp, _, _ = self._conv_image(right_vpp)

        self.model.eval()
        with torch.no_grad():
            disp_ests, _, _  = self.model(left_vpp, right_vpp)
            return disp_ests[0][:, toppad:, :-rightpad].numpy().squeeze()
