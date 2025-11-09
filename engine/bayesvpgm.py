import torch
import torch.nn as nn
import torch.nn.functional as F


class nn_model(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(nn_model, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


class pi_k_model(nn.Module):
    def __init__(self, nks, max_num_category):
        super(pi_k_model, self).__init__()
        self.lamb = nn.Parameter(torch.ones(1) * 0.000007)  # This is \tau in the paper
        self.nks = nks.clone().detach().requires_grad_(True)

    def forward(self, pis, tags):
        # Expand temperature to match the size of logits
        temperature = self.lamb.unsqueeze(1).expand(pis.size(0), pis.size(1))
        alpha_k = pis / temperature

        # Function to manipulate nks according to the tag
        def f(tag, nks):
            # Create a mask that zeroes out elements after the 'tag' index
            mask = torch.arange(nks.size(0)) < tag
            return nks * mask

        # Create _nks with shape (b, c)
        _b = pis.shape[0]  # Batch size
        _c = pis.shape[1]  # Number of categories (same as nks.shape)
        _nks = torch.zeros((_b, _c), dtype=torch.float32)

        # Apply function f for each row
        for i in range(_b):
            _nks[i] = f(tags[i], self.nks)

        alpha_k_prime = _nks + alpha_k
        # Sum over the second dimension (sum of each row)
        sum_alpha = alpha_k_prime.sum(dim=1, keepdim=True)
        return alpha_k_prime / sum_alpha


class MDCA(torch.nn.Module):
    def __init__(self):
        super(MDCA, self).__init__()

    def forward(self, output, target):
        output = torch.softmax(output, dim=1)
        # [batch, classes]
        loss = torch.tensor(0.0)
        batch, classes = output.shape
        for c in range(classes):
            avg_count = (target == c).float().mean()
            avg_conf = torch.mean(output[:, c])
            loss += torch.abs(avg_conf - avg_count)
        denom = classes
        loss /= denom
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, **kwargs):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        # logging.info("using gamma={}".format(gamma))

    def forward(self, input, target):
        target = target.view(-1, 1)

        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()


class ClassficationAndMDCA(nn.Module):
    def __init__(self, loss="FL+MDCA", alpha=0.1, beta=1.0, gamma=1.0, **kwargs):
        super(ClassficationAndMDCA, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if "NLL" in loss:
            self.classification_loss = nn.CrossEntropyLoss()
        elif "FL" in loss:
            self.classification_loss = FocalLoss(gamma=self.gamma)
        self.MDCA = MDCA()

    def forward(self, logits, targets):
        loss_cls = self.classification_loss(logits, targets)
        loss_cal = self.MDCA(logits, targets)
        return loss_cls + self.beta * loss_cal
