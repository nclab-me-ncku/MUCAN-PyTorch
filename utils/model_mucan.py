from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from tools import timer, get_system_info
from logger import Logger


class Down(nn.Module):
    """
    net1: dimension reduction block
    net2: normal FFN block
    """
    def __init__(self, input_dim, output_dim, dropout=0.4):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(dropout),
            nn.ELU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout),
            nn.ELU(),
        )

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.net2(out1)
        return out2
    

class Up(nn.Module):
    """
    net1: dimension extension block
    net2: normal FFN block (with skip-connection)
    """
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Dropout(dropout), 
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout), 
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x, z):
        out1 = self.net1(x)
        out1_cat = torch.cat((out1, z), dim=1)
        out2 = self.net2(out1_cat)
        return out2


class Generator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.Dropout(0.4),
            nn.ELU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Dropout(0.4),
            nn.ELU(),
        )
        self.down1 = Down(input_dim=input_dim, output_dim=64)
        self.down2 = Down(input_dim=64, output_dim=32)
        self.up1 = Up(input_dim=32, output_dim=64)
        self.up2 = Up(input_dim=64, output_dim=input_dim)

    def forward(self, x, m):
        xm = torch.cat((x, m), dim=1)
        x0 = self.input_layer(xm)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        out = self.up1(x2, x1)
        out = self.up2(out, x0)
        return out

    
class Discriminator(nn.Module):
    def __init__(self, input_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        label = self.net(x)
        return label

    
class Aligner(object):
    def __init__(self, input_dim: int):
        self.gen_d0k = Generator(input_dim)
        self.gen_dk0 = Generator(input_dim)
        self.disc_d0 = Discriminator(input_dim)
        self.disc_dk = Discriminator(input_dim)
        
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        self.gen_d0k_optim = optim.Adam(self.gen_d0k.parameters(), lr=0.001)
        self.gen_dk0_optim = optim.Adam(self.gen_dk0.parameters(), lr=0.001)
        self.disc_d0_optim = optim.Adam(self.disc_d0.parameters(), lr=0.01)
        self.disc_dk_optim = optim.Adam(self.disc_dk.parameters(), lr=0.01)
        
    def save_weights(self, pth_path: str):
        states = {
            'gen_dk0': self.gen_dk0.state_dict(),
            'gen_d0k': self.gen_d0k.state_dict(),
            'disc_d0': self.disc_d0.state_dict(),
            'disc_dk': self.disc_dk.state_dict(),
        }
        torch.save(states, pth_path)
        print(f'Save model weights to \033[34m{pth_path}\033[0m')
    
    def load_weights(self, pth_path: str):
        states = torch.load(pth_path)
        self.gen_dk0.load_state_dict(states['gen_dk0'])
        self.gen_d0k.load_state_dict(states['gen_d0k'])
        self.disc_d0.load_state_dict(states['disc_d0'])
        self.disc_dk.load_state_dict(states['disc_dk'])
        print(f'Load model weights from \033[34m{pth_path}\033[0m')
    
    @timer
    def fit(self, day0_x: list[np.ndarray], dayk_x: list[np.ndarray],
            device=torch.device('cpu'), param=dict(), enable_logging=False):
        """
        Training process
        """
        batch_size = param.get('batch_size', 256)
        n_epochs = param.get('n_epochs', 400)
        n_masks = param.get('n_masks', 10)
        
        # torch Dataset
        d0_dataset = NeuralDataset(spikes=day0_x, labels=day0_x)  # trivial label
        dk_dataset = NeuralDataset(spikes=dayk_x, labels=dayk_x)  # trivial label
        
        # torch DataLoader
        d0_loader = DataLoader(d0_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        dk_loader = DataLoader(dk_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        self.models = dict()  # for saving validation value and model
        if enable_logging:
            self.logger = Logger()
            self.logger.update(desc='Start training')
        
        self.gen_d0k.train().to(device)
        self.gen_dk0.train().to(device)
        self.disc_d0.train().to(device)
        self.disc_dk.train().to(device)
        
        with tqdm(total=n_epochs, desc='Training') as pbar:
            for epoch in range(1, n_epochs + 1):
                
                # main train process
                self._train_step(d0_loader, dk_loader, n_masks, device)
                
                # update info
                if enable_logging:
                    self._add_logs()
                    self.logger.update(desc=f'Epoch [{epoch}/{n_epochs}]')
                pbar.set_postfix(get_system_info())
                pbar.update()
                
                # validation
                if epoch % 10 == 0 and epoch / n_epochs >= 0.5:
                    self._validate(d0_loader, dk_loader, device)
        try:
            self.gen_dk0 = self.models[min(self.models.keys())]
        except:
            pass
        
        self.gen_d0k.eval().cpu()
        self.gen_dk0.eval().cpu()
        self.disc_d0.eval().cpu()
        self.disc_dk.eval().cpu()
        
    @torch.no_grad()
    def transform(self, x_test: list[np.ndarray], device=torch.device('cpu')) -> list[np.ndarray]:
        """
        Test aligner
        """
        self.gen_dk0.eval().to(device)
        x_aligned = list()
        for x in x_test:
            tensor = torch.from_numpy(x).float().to(device)
            x_aligned.append(self.gen_dk0(*self._random_mask(tensor)).cpu().detach().numpy())
            
        return x_aligned
        
    @torch.no_grad()
    def _validate(self, d0_loader, dk_loader, device):
        self.gen_dk0.eval().to(device)
        x_autual = torch.tensor([]).to(device)  # actual day-0
        x_aligned = torch.tensor([]).to(device)  # fake day-0 (aligned day-k)
        
        for (d0_x, _), (dk_x, _) in zip(d0_loader, dk_loader):
            d0_x, dk_x = d0_x.to(device), dk_x.to(device)
            dk_x_aligned = self.gen_dk0(*self._random_mask(dk_x))                            
            x_autual = torch.cat((x_autual, d0_x), dim=0)
            x_aligned = torch.cat((x_aligned, dk_x_aligned), dim=0)
        
        mmd = self.mmd_rbf(x_autual.detach(), x_aligned.detach()).cpu()
        try:
            # self.logger.add(mmd, 'MMD')
            self.logger.update(desc=f'The MMD value is {mmd}')
        except:
            pass
        self.models.update({mmd: self.gen_dk0})
        self.gen_dk0.train()
    
    def _add_logs(self):
        """
        Add logs to logger container
        """
        logs = [
            (self.loss_idt_dk.item(), 'loss_idt_dk'),
            (self.loss_idt_d0.item(), 'loss_idt_d0'),
            (self.loss_GAN_d0k.item(), 'loss_GAN_d0k'),
            (self.loss_GAN_dk0.item(), 'loss_GAN_dk0'),
            (self.loss_cycle_d0.item(), 'loss_cycle_d0'),
            (self.loss_cycle_dk.item(), 'loss_cycle_dk'),
            (self.loss_disc_dk.item(), 'loss_disc_dk'),
            (self.loss_disc_d0.item(), 'loss_disc_d0'),
        ]
        for value, name in logs:
            try:
                self.logger.add(value, name)
            except Exception as err:
                pass
            
    def _train_step(self, d0_loader, dk_loader, n_masks, device):
        for (d0_x, _), (dk_x, _) in zip(d0_loader, dk_loader):
            
            d0_x = d0_x.to(device)
            dk_x = dk_x.to(device)
            target_real = torch.ones((d0_x.size(0), 1), requires_grad=False, dtype=torch.float, device=device)
            target_fake = torch.zeros((d0_x.size(0), 1), requires_grad=False, dtype=torch.float, device=device)

            # train generators
            self.gen_d0k_optim.zero_grad()
            self.gen_dk0_optim.zero_grad()
            self._backward_gen(d0_x, dk_x, target_real, n_masks) 
            self.gen_d0k_optim.step()
            self.gen_dk0_optim.step()

            # train day-0 discriminator
            self.disc_d0_optim.zero_grad()
            self._backward_disc_d0(d0_x, dk_x, target_real, target_fake)
            self.disc_d0_optim.step()

            # train day-k discriminator
            self.disc_dk_optim.zero_grad()
            self._backward_disc_dk(d0_x, dk_x, target_real, target_fake)
            self.disc_dk_optim.step()
        
    def _backward_gen(self, d0, dk, target_real, n_masks):
        """
        Calculate the loss for generators gen_d0k and gen_dk0
        """
        lambda_idt, lambda_GAN, lambda_cycle = 5.0, 1.0, 5.0
        
        # identity loss
        idt_dk = self.gen_d0k(*self._random_mask(dk))
        self.loss_idt_dk = self.criterion_identity(idt_dk, dk) * lambda_idt
        
        idt_d0 = self.gen_dk0(*self._random_mask(d0))
        self.loss_idt_d0 = self.criterion_identity(idt_d0, d0) * lambda_idt
        
        # GAN loss
        fake_dk = self.gen_d0k(*self._random_mask(d0, n_masks))
        pred_fake = self.disc_dk(fake_dk)
        self.loss_GAN_d0k = self.criterion_GAN(pred_fake, target_real) * lambda_GAN
        
        fake_d0 = self.gen_dk0(*self._random_mask(dk, n_masks))
        pred_fake = self.disc_d0(fake_d0)
        self.loss_GAN_dk0 = self.criterion_GAN(pred_fake, target_real) * lambda_GAN
        
        # cycle loss
        rec_d0 = self.gen_dk0(*self._random_mask(fake_dk))
        self.loss_cycle_d0 = self.criterion_cycle(rec_d0, d0) * lambda_cycle
        
        rec_dk = self.gen_d0k(*self._random_mask(fake_d0))
        self.loss_cycle_dk = self.criterion_cycle(rec_dk, dk) * lambda_cycle
        
        # total loss
        self.loss_gen = (
            self.loss_idt_d0 + self.loss_idt_dk
            + self.loss_GAN_d0k + self.loss_GAN_dk0
            + self.loss_cycle_d0 + self.loss_cycle_dk
        )
        self.loss_gen.backward()
    
    def _backward_disc_d0(self, d0, dk, target_real, target_fake):
        """
        Calculate the loss for discriminator disc_d0
        """
        pred_real = self.disc_d0(d0)
        loss_disc_real = self.criterion_GAN(pred_real, target_real)
        
        pred_fake = self.disc_d0(self.gen_dk0(*self._random_mask(dk)).detach())
        loss_disc_fake = self.criterion_GAN(pred_fake, target_fake)
        
        self.loss_disc_d0 = (loss_disc_real + loss_disc_fake) * 0.5
        self.loss_disc_d0.backward()
    
    def _backward_disc_dk(self, d0, dk, target_real, target_fake):
        """
        Calculate the loss for discriminator disc_dk
        """
        pred_real = self.disc_dk(dk)
        loss_disc_real = self.criterion_GAN(pred_real, target_real)
        
        pred_fake = self.disc_dk(self.gen_d0k(*self._random_mask(d0)).detach())
        loss_disc_fake = self.criterion_GAN(pred_fake, target_fake)
        
        self.loss_disc_dk = (loss_disc_real + loss_disc_fake) * 0.5
        self.loss_disc_dk.backward()
    
    def _random_mask(self, x: torch.Tensor, n_mask: int = 0):
        mask = torch.ones_like(x).to(x.device)
        mask_indices = torch.rand(x.size()).to(x.device).argsort(dim=-1)[..., :n_mask]
        mask.scatter_(-1, mask_indices, 0)
        return torch.mul(x, mask), mask
    
    def rbf_kernel(self, X, Y, gamma=None):
        if gamma is None:
            gamma = 1.0 / X.size(1)
        K = torch.exp(-gamma * torch.cdist(X, Y) ** 2)
        return K
    
    def mmd_rbf(self, X, Y, **kernel_args):
        XX = self.rbf_kernel(X, X, **kernel_args)
        YY = self.rbf_kernel(Y, Y, **kernel_args)
        XY = self.rbf_kernel(X, Y, **kernel_args)
        return XX.mean() + YY.mean() - 2 * XY.mean()
    
    
class NeuralDataset(Dataset):
    """
    A custom torch Dataset class for neural data.
    """
    def __init__(self, spikes: list[np.ndarray], labels: list[np.ndarray]):
        super().__init__()
        self.lengths = [len(item) for item in spikes]
        self.sections = [sum(self.lengths[:i]) for i, _ in enumerate(self.lengths, 1)][:-1]
        
        self.spike, self.label = self.format_data(spikes, labels)
    
    def __len__(self):
        return len(self.spike)
    
    def __getitem__(self, idx):
        return self.spike[idx], self.label[idx]
    
    def format_data(self, spikes: list[np.ndarray], labels: list[np.ndarray], n_taps: Optional[int] = None):
        """
        Format trial-based data into Tensor (float)
        Add time lags to firing rates
        """
        if n_taps is not None:
            spike = np.concatenate([self.add_lags(arr, n_taps) for arr in spikes])
        else:
            spike = np.concatenate(spikes)
        label = np.concatenate(labels)
        return torch.from_numpy(spike).float(), torch.from_numpy(label).float()
    
    def add_lags(self, arr: np.ndarray, n_lags: int, zero_padding=True):
        """
        Add time lags to a 2D array. (n, c) -> (n, t, c)
        """
        pad = np.pad(arr, [(n_lags - 1, 0), (0, 0)])
        pack = [np.roll(pad[np.newaxis], i, axis=1) for i in range(n_lags - 1, -1, -1)]
        arrs = np.vstack(pack).transpose(1, 0, 2)[n_lags-1:]
        return arrs if zero_padding else arrs[n_lags-1:]

