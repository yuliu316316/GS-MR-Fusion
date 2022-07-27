# coding: utf-8
import os
import torch

import preprocessing
import utils
from tensorboardX import SummaryWriter

from model import G, D
from model import GAFNet
from unet import Unet
#from test import test_all
from metric import dice_coef
import pytorch_ssim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


class CGAN:
    def __init__(self, config: dict):
        self.config = config

        # self.gen = G().to(device)
        self.gen = GAFNet().to(device)
        #self.dis = D().to(device)
        self.dis = Unet().to(device)
        self.dis1 = Unet().to(device)

        self.gen_op = torch.optim.Adam(self.gen.parameters(), lr=config['lr'])
        self.dis_op = torch.optim.Adam(self.dis.parameters(), lr=config['lr'])
        self.dis1_op = torch.optim.Adam(self.dis1.parameters(), lr=config['lr'])

        self.lda = config['lda']
        self.epsilon1 = config['epsilon1']
        self.epsilon2 = config['epsilon2']

    def train_step(self, inf_x, vis_x, input_seg1, input_seg2, k=2):
        self.gen.train()
        d_loss_val = 0
        g_loss_val = 0

        fusion = self.gen(inf_x, vis_x)  # [2,1,240,240]
        with torch.no_grad():
            fusion_detach = fusion

        for _ in range(k):
            self.dis_op.zero_grad()
            inf_output = self.dis(inf_x)
            fus_output = self.dis(fusion_detach)
            #dis_loss = self.dis_loss_func(vis_output, fus_output)
            dis_loss = self.dis_loss_BraTS(inf_output, fus_output, input_seg1)
            d_loss_val += dis_loss.cpu().item()
            dis_loss.backward(retain_graph=True)
            self.dis_op.step()

        for _ in range(k):
            self.dis1_op.zero_grad()
            vis_output = self.dis1(vis_x)
            fus_output = self.dis1(fusion_detach)
            dis_loss = self.dis_loss_BraTS(vis_output, fus_output, input_seg2)
            d_loss_val += dis_loss.cpu().item()
            dis_loss.backward(retain_graph=True)
            self.dis1_op.step()

        self.gen_op.zero_grad()
        fus_output1 = self.dis(fusion)
        fus_output2 = self.dis1(fusion)
        g_loss, v_gan_loss, content_loss, ssimloss, pixelloss = self.gen_loss_BraTS(fus_output1, fus_output2, fusion, inf_x, vis_x, input_seg1, input_seg2)
        g_loss_val += g_loss.cpu().item()
        g_loss.backward(retain_graph=False)
        self.gen_op.step()
        return d_loss_val / k, g_loss_val, v_gan_loss, content_loss, ssimloss, pixelloss

    @staticmethod
    def dis_loss_func(vis_output, fusion_output):
        return torch.mean(torch.square(vis_output - torch.Tensor(vis_output.shape).uniform_(0.7, 1.2).to(device))) + \
               torch.mean(torch.square(fusion_output - torch.Tensor(fusion_output.shape).uniform_(0, 0.3).to(device)))

    @staticmethod
    def dis_loss_BraTS(img_output, fusion_output, target):
        dice1 = dice_coef(img_output, target)
        dice2 = dice_coef(fusion_output, target)
        # dis_loss = torch.square(dice1-1.0) + torch.square(dice2) + torch.square(dice3-1.0)
        dis_loss = torch.square(dice1 - torch.Tensor(dice1.shape).uniform_(0.7, 1.2)) + \
                   torch.square(dice2 - torch.Tensor(dice2.shape).uniform_(0, 0.3))

        return dis_loss

    def gen_loss_func(self, fusion_output, fusion_img, inf_img, vis_img):
        gan_loss = torch.mean(torch.square(fusion_output - torch.Tensor(fusion_output.shape).uniform_(0.7, 1.2).to(device)))
        content_loss = torch.mean(torch.square(fusion_img - inf_img)) + \
                       self.epsilon1 * torch.mean(torch.square(utils.gradient(fusion_img) - utils.gradient(vis_img)))
        return self.lda *gan_loss + content_loss, self.lda *gan_loss, content_loss

    def gen_loss_BraTS(self, fusion_output1, fusion_output2, fusion_img, inf_img, vis_img, input_seg1, input_seg2):
        dice_loss1 = dice_coef(fusion_output1, input_seg1)
        gan_loss1 = torch.square(dice_loss1 - torch.Tensor(dice_loss1.shape).uniform_(0.7, 1.2))
        dice_loss2 = dice_coef(fusion_output2, input_seg2)
        gan_loss2 = torch.square(dice_loss2 - torch.Tensor(dice_loss2.shape).uniform_(0.7, 1.2))
        gan_loss = gan_loss1 + gan_loss2
        #t1_gradient_loss = torch.mean(torch.square(utils.gradient(fusion_img) - utils.gradient(inf_img)))
        ssim_loss = 2*(1-pytorch_ssim.ssim(inf_img, fusion_img)) + (1-pytorch_ssim.ssim(vis_img, fusion_img))
        pixel_loss = (torch.norm((fusion_img - inf_img)) + 2*torch.norm((fusion_img - vis_img)))/(240*240)
        content_loss = ssim_loss + self.epsilon1*pixel_loss
        return gan_loss + self.lda * content_loss, gan_loss, self.lda * content_loss, ssim_loss, self.epsilon1*pixel_loss

    def train(self):
        if self.config['is_train']:
            data_dir_ir = os.path.join(self.config['data'], 'Train_ir')
            data_dir_vi = os.path.join(self.config['data'], 'Train_vi')
        else:
            data_dir_ir = os.path.join(self.config['data'], 'Test_ir')
            data_dir_vi = os.path.join(self.config['data'], 'Test_ir')

        # train_data_ir, train_label_ir = preprocessing.get_images2(data_dir_ir, self.config['image_size'],
        #                                                           self.config['label_size'], self.config['stride'])
        # train_data_vi, train_label_vi = preprocessing.get_images2(data_dir_vi, self.config['image_size'],
        #                                                           self.config['label_size'], self.config['stride'])

        train_data_t1, train_data_t2, train_seg1, train_seg2 = preprocessing.get_BraTS_images2(self.config['image_size'])
        print(train_data_t1.shape)

        random_index = torch.randperm(len(train_data_t1))
        train_data_t1 = train_data_t1[random_index]
        train_data_t2 = train_data_t2[random_index]
        train_seg1 = train_seg1[random_index]
        train_seg2 = train_seg2[random_index]
        batch_size = self.config['batch_size']

        if self.config['is_train']:
            with SummaryWriter(self.config['summary_dir']) as writer:

                batch_steps = len(train_data_t1) // batch_size
                epochs = self.config['epoch']
                for epoch in range(1, 1 + epochs):
                    d_loss_mean = 0
                    g_loss_mean = 0
                    content_loss_mean = 0
                    for step in range(1, 1 + batch_steps):
                        start_idx = (step - 1) * batch_size
                        inf_x = train_data_t1[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        vis_x = train_data_t2[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        input_seg1 = train_seg1[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        input_seg2 = train_seg2[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])

                        inf_x = torch.tensor(inf_x).float().to(device)
                        vis_x = torch.tensor(vis_x).float().to(device)
                        input_seg1 = torch.tensor(input_seg1).float().to(device)
                        input_seg2 = torch.tensor(input_seg2).float().to(device)

                        d_loss, g_loss, v_gan_loss, content_loss, ssim_loss, loss_pixel = self.train_step(inf_x, vis_x, input_seg1, input_seg2, 2)
                        d_loss_mean += d_loss
                        g_loss_mean += g_loss
                        content_loss_mean += content_loss
                        print('Epoch {}/{}, Step {}/{}, gen loss = {:.4f}, v_gan_loss = {:.4f}, '
                              'content_loss {:.4f}, dis loss = {:.4f}, ssim loss = {:.4f}, pixel loss = {:.4f}' \
                              .format(epoch, epochs, step, batch_steps,
                                                        g_loss, v_gan_loss, content_loss, d_loss, ssim_loss, loss_pixel))
                    #test_all(self.gen, os.path.join(self.config['output'], 'test{}'.format(epoch)))

                    d_loss_mean /= batch_steps
                    g_loss_mean /= batch_steps
                    content_loss_mean /= batch_steps
                    writer.add_scalar('scalar/gen_loss', g_loss_mean, epoch)
                    writer.add_scalar('scalar/dis_loss', d_loss_mean, epoch)
                    writer.add_scalar('scalar/content_loss', content_loss_mean, epoch)

                    # for name, param in self.gen.named_parameters():
                    #     if 'bn' not in name:
                    #         writer.add_histogram('gen/'+name, param, epoch)
                    #
                    # for name, param in self.dis.named_parameters():
                    #     if 'bn' not in name:
                    #         writer.add_histogram('dis/'+name, param, epoch)
                    if not os.path.exists('%s/' % (self.config['model'])):
                        os.makedirs('%s/' % (self.config['model']))
                    torch.save(self.gen.state_dict(), '%s/%s_generator.pth' % (self.config['model'], epoch))
            print('Saving model......')
            torch.save(self.gen.state_dict(), '%s/final_generator.pth' % (self.config['model']))
            print("Training Finished, Total EPOCH = %d" % self.config['epoch'])

