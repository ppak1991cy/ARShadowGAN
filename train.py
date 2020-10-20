"""
    ARShadowGAN
    func  : 完整训练
    Author: Chen Yu
    Date  : 2020.10.20
"""
import os

import torchvision
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from vgg import vgg16
from dataset import ShadowARDataset
from net import Generator, Discriminator
from attention import AttentionNet


BETA_1 = 10
BETA_2 = 1
BETA_3 = 0.01


class Trainer(object):

    def __init__(self, batch_size=2, num_workers=4, device='cuda'):
        self.batch_size = batch_size
        self.device = device
        self.e = 0

        self.attention_net = AttentionNet(4).to(self.device)
        self.generator = Generator(6).to(self.device)
        self.discriminator = Discriminator(7).to(self.device)
        self.vgg = vgg16().to(self.device)
        self.vgg.eval()
        for p in self.vgg.parameters():  # reset requires_grad
            p.requires_grad_(False)

        self.train_ds = DataLoader(ShadowARDataset(), batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True, drop_last=True)

        self.optimizer_attention = torch.optim.Adam(self.attention_net.parameters(), lr=1e-5, betas=(0.9, 0.99))
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=1e-5, betas=(0.9, 0.99))
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=1e-5, betas=(0.9, 0.99))
        self.one = torch.FloatTensor([1]).to(device)

    def eval_attention(self, e, no_vs, vo_mask):
        self.attention_net.eval()
        pred_rs_mask, pred_ro_mask = self.attention_net(torch.cat([no_vs, vo_mask], dim=1))
        pred_rs_mask = (pred_rs_mask * 0.5) + 0.5
        pred_ro_mask = (pred_ro_mask * 0.5) + 0.5
        save_dir = os.path.join("pred_mask")
        os.makedirs(save_dir, exist_ok=True)
        torchvision.utils.save_image(pred_rs_mask, os.path.join(save_dir, "shadow_{}.png".format(e)),
                                     nrow=self.batch_size, padding=2)
        torchvision.utils.save_image(pred_ro_mask, os.path.join(save_dir, "occluder_{}.png".format(e)),
                                     nrow=self.batch_size, padding=2)

    def save_attention(self, e):
        save_dir = os.path.join("param_attention")
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'attention_net': self.attention_net.state_dict(),
            'optimizer': self.optimizer_attention.state_dict(),
            'epoch': e
        }
        path = os.path.join(save_dir, "attention_%d.pkl" % e)
        torch.save(checkpoint, path)

    def load_attention(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.attention_net.load_state_dict(checkpoint["attention_net"])
        self.optimizer_attention.load_state_dict(checkpoint["optimizer"])
        self.e = checkpoint["epoch"]
        # self.e = 1080

    def train_attention(self, epoch=100000):
        for e in range(self.e, epoch):
            self.attention_net.train()
            for step, (vs, no_vs, rs_mask, ro_mask, vo_mask) in enumerate(self.train_ds):
                self.attention_net.zero_grad()
                no_vs = no_vs.to(self.device)
                vo_mask = vo_mask.to(self.device)
                rs_mask = rs_mask.to(self.device)
                ro_mask = ro_mask.to(self.device)
                pred_rs_mask, pred_ro_mask = self.attention_net(torch.cat([no_vs, vo_mask], dim=1))
                shadow_loss = F.mse_loss(rs_mask, pred_rs_mask)
                occluder_loss = F.mse_loss(ro_mask, pred_ro_mask)
                loss = occluder_loss + shadow_loss
                loss.backward()
                self.optimizer_attention.step()
                if step % 5 == 0:
                    print('epoch e: %d  step: %d  shadow loss: %.4f  occluder: %.4f' % (e, step, shadow_loss, occluder_loss))
            self.eval_attention(e, no_vs, vo_mask)
            if e % 10 == 0:
                self.save_attention(e)

    def update_generator(self, no_vs, vo_mask, vs, x):
        self.generator.train()
        self.discriminator.eval()
        self.generator.zero_grad()
        for p in self.discriminator.parameters():  # reset requires_grad
            p.requires_grad_(False)

        y_, y_r = self.generator(x)

        fake_y_ = no_vs + y_
        fake_y_r = no_vs + y_r

        vgg_vs = F.upsample(vs, (228, 228), mode='nearest')
        vgg_fake_y_ = F.upsample(fake_y_, (228, 228), mode='nearest')
        vgg_fake_y_r = F.upsample(fake_y_r, (228, 228), mode='nearest')
        vgg_vs = self.vgg(vgg_vs)
        vgg_fake_y_ = self.vgg(vgg_fake_y_)
        vgg_fake_y_r = self.vgg(vgg_fake_y_r)

        loss_l2 = F.mse_loss(vs - no_vs, y_) + F.mse_loss(vs - no_vs, y_r)
        loss_vgg = F.mse_loss(vgg_vs, vgg_fake_y_) + F.mse_loss(vgg_vs, vgg_fake_y_r)

        loss_gan = self.discriminator(torch.cat([no_vs, vo_mask, fake_y_r], dim=1)).mean()

        loss = BETA_1 * loss_l2 + BETA_2 * loss_vgg - BETA_3 * loss_gan
        loss.backward(self.one)
        self.optimizer_generator.step()
        return loss_l2, loss_vgg, loss_gan

    def update_discriminator(self, no_vs, vo_mask, vs, x):
        self.generator.eval()
        self.discriminator.train()
        for p in self.discriminator.parameters():  # reset requires_grad
            p.requires_grad_(True)

        self.discriminator.zero_grad()
        y_, y_r = self.generator(x)
        fake_y_r = no_vs + y_r

        fake = self.discriminator(torch.cat([no_vs, vo_mask, fake_y_r], dim=1)).mean()
        real = self.discriminator(torch.cat([no_vs, vo_mask, vs], dim=1)).mean()
        loss = - BETA_3 * (real - fake)
        loss.backward(self.one)
        self.optimizer_discriminator.step()
        return loss

    def save(self, e):
        save_dir = os.path.join("param")
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_generator': self.optimizer_generator.state_dict(),
            'optimizer_discriminator': self.optimizer_discriminator.state_dict(),
            'epoch': e
        }
        path = os.path.join(save_dir, "gen_%d.pkl" % e)
        torch.save(checkpoint, path)

    def load(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.optimizer_generator.load_state_dict(checkpoint["optimizer_generator"])
        self.optimizer_discriminator.load_state_dict(checkpoint["optimizer_discriminator"])
        self.e = checkpoint["epoch"]
        # self.e = 1080

    def eval(self, e, no_vs, x, vs):
        self.generator.eval()
        y_, y_r = self.generator(x)
        fake_y_r = no_vs + y_r
        fake_srd = (fake_y_r * 0.5 + 0.5)
        save_dir = os.path.join("examples")
        os.makedirs(save_dir, exist_ok=True)
        torchvision.utils.save_image(fake_srd, os.path.join(save_dir, "fake_{}.png".format(e)),
                                     nrow=self.batch_size, padding=2)
        vs = (vs * 0.5 + 0.5)
        save_dir = os.path.join("examples")
        os.makedirs(save_dir, exist_ok=True)
        torchvision.utils.save_image(vs, os.path.join(save_dir, "real_{}.png".format(e)),
                                     nrow=self.batch_size, padding=2)

    def train(self, epoch=100000):
        self.attention_net.eval()
        for p in self.attention_net.parameters():  # reset requires_grad
            p.requires_grad_(False)

        for e in range(self.e, epoch):
            self.attention_net.train()
            for step, (vs, no_vs, rs_mask, ro_mask, vo_mask) in enumerate(self.train_ds):
                vs = vs.to(self.device)
                no_vs = no_vs.to(self.device)
                vo_mask = vo_mask.to(self.device)

                pred_rs_mask, pred_ro_mask = self.attention_net(torch.cat([no_vs, vo_mask], dim=1))
                pred_rs_mask = pred_rs_mask.detach()
                pred_ro_mask = pred_ro_mask.detach()
                x = torch.cat([no_vs, vo_mask, pred_rs_mask, pred_ro_mask], dim=1)

                loss_l2, loss_vgg, loss_gan = self.update_generator(no_vs, vo_mask, vs, x)
                d_loss = self.update_discriminator(no_vs, vo_mask, vs, x)
                if step % 5 == 0 and step != 0:
                    print("epoch %d  step %d  g_loss l2 %.4f  vgg %.4f gan %.4f d_loss %.4f" %
                          (e, step, loss_l2, loss_vgg, loss_gan, d_loss))
            self.save(0)
            self.eval(e, no_vs, x, vs)
            if e % 10 == 0:
                self.save(e)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.load_attention('param_attention/attention_1520.pkl')
    trainer.load('param/gen_410.pkl')
    # trainer.train_attention()
    trainer.train()

