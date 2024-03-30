import numpy as np

class SaliencyMap:
    def __init__(self, saliency):
        self.saliency = saliency

    def run(self, input_tensor, target):
        input_tensor.requires_grad_(True)
        saliency_map = self.saliency.attribute(input_tensor, target)
        normalized_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        visualization = normalized_saliency_map.squeeze(0).permute(1, 2, 0).cpu().numpy() 
        saliency_map = saliency_map.squeeze(0).permute(1, 2, 0).cpu().numpy()
        saliency_map = np.sum(saliency_map, axis=2) / saliency_map.shape[2]
        return saliency_map, visualization 
