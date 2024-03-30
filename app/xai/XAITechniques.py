class XAITechniques:
    def __init__(self, gradcam, saliency_map, integrated_gradients_method, lime):
        self.gradcam = gradcam
        self.saliency_map = saliency_map
        self.integrated_gradients_method = integrated_gradients_method
        self.lime = lime

    def run_gradcam(self, input_tensor, target):
        return self.gradcam.run(input_tensor, target)

    def run_saliency_map(self, input_tensor, target):
        return self.saliency_map.run(input_tensor, target)

    def run_integrated_gradients(self, inputs, predicted_class):
        return self.integrated_gradients_method.run(inputs, predicted_class)

    def run_lime(self, img):
        return self.lime.run(img)
