import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from lime import lime_image

class Lime:
    def __init__(self, explainer, model, device): 
        self.explainer = explainer
        self.model = model
        self.device = device

    def get_preprocess_transform(self):
        transf = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transf
    
    def batch_predict(self, images):
        transform = self.get_preprocess_transform()
        self.model.eval()
        batch = torch.stack(tuple(transform(i) for i in images), dim=0)

        self.model.to(self.device)
        batch = batch.to(self.device)

        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def generate_prediction_sample(self, exp, exp_class, weight=0.1, show_positive=True, hide_background=True):
        image, mask = exp.get_image_and_mask(exp_class,
                                            positive_only=show_positive,
                                            num_features=3,
                                            hide_rest=hide_background,
                                            min_weight=weight)
        return image, mask

    def explanation_heatmap(self, exp, exp_class):
        dict_heatmap = dict(exp.local_exp[exp_class])
        heatmap = np.vectorize(dict_heatmap.get)(exp.segments)
        return heatmap
        
    def run(self, img): 
        normalized_img = img.permute(1, 2, 0)
        exp = self.explainer.explain_instance(np.array(normalized_img),
                                               self.batch_predict,
                                               top_labels=5,
                                               hide_color=0,
                                               num_samples=1000)

        segments = exp.segments

        for i, exp_class in enumerate(exp.top_labels):
            print(f"\n\n\033[1;31mExplanation for predicted class {exp_class}: rank {i}\033[0m")
            image, mask = self.generate_prediction_sample(exp, exp_class, weight=0.0001, show_positive=True, hide_background=True)
            lime_heatmap = self.explanation_heatmap(exp, exp_class)
            break

        return lime_heatmap, segments, image, mask
