import torch

class IntegratedGradients: 
    def __init__(self, model):
        self.model = model

    def run(self, inputs, predicted_class):
        integrated_gradients = torch.zeros_like(inputs)
        baseline = torch.zeros_like(inputs)
        m_steps = 200  # Number of steps for approximation
        alphas = torch.linspace(0.0, 1.0, m_steps)

        for alpha in alphas:
            x_interpolated = baseline + alpha * (inputs - baseline)
            x_interpolated.requires_grad_(True)

            logits = self.model(x_interpolated)
            predicted_class_logits = logits[0][predicted_class]

            gradient = torch.autograd.grad(predicted_class_logits, x_interpolated)[0]
            integrated_gradients += gradient / m_steps

        integrated_gradients = torch.sum(torch.abs(integrated_gradients), dim=1)
        integrated_gradients = integrated_gradients.permute(1, 2, 0).cpu().squeeze(2).numpy()

        return integrated_gradients
