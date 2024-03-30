import torch
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class GradientCAM:
    def __init__(self, cam):
        self.cam = cam

    def run(self, input_tensor, target):
        targets = [ClassifierOutputTarget(target)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)

        input_tensor = input_tensor.squeeze(0).permute(1, 2, 0)
        min_value = torch.min(input_tensor)
        max_value = torch.max(input_tensor)
        rescaled_image = (input_tensor - min_value) / (max_value - min_value)
        rescaled_image = torch.clamp(rescaled_image, 0, 1)

        visualization = show_cam_on_image(rescaled_image.cpu().numpy(), grayscale_cam[0, :], use_rgb=True)

        return grayscale_cam[0, :], visualization
