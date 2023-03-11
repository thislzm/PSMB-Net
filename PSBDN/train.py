import argparse
import math
import os
import ssl
import time

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from tqdm import tqdm

from Model import Base_Model, Discriminator
from Model_util import padding_image
from make import getTxt
from perceptual import LossNetwork
from pytorch_msssim import msssim
from test_dataset import dehaze_test_dataset
from train_dataset import dehaze_train_dataset
from utils_test import to_psnr, to_ssim_skimage

ssl._create_default_https_context = ssl._create_unverified_context

# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='Siamese Dehaze Network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=2, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=20000, type=int)
parser.add_argument("--type", default=-1, type=int, help="choose a type 012345")

parser.add_argument('--train_dir', type=str, default='/home/lzm/datasets_train/Outdoor/train/')
parser.add_argument('--train_name', type=str, default='hazy,clean')
parser.add_argument('--test_dir', type=str, default='/home/lzm/deHaze/outdoor_Test/')
parser.add_argument('--test_name', type=str, default='hazy,clean')

parser.add_argument('--model_save_dir', type=str, default='./output_result')
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--gpus', default='0,1,2,3', type=str)
# --- Parse hyper-parameters test --- #
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
parser.add_argument('--vgg_model', default='', type=str, help='load trained model or not')
parser.add_argument('--use_bn', action='store_true', help='if bs>8 please use bn')
parser.add_argument('--restart', action='store_true', help='')
parser.add_argument('--num', type=str, default='9999999', help='')
parser.add_argument('--sep', type=int, default='10', help='')
parser.add_argument('--save_psnr', action='store_true', help='')
parser.add_argument('--seps', action='store_true', help='')

print('+++++++++++++++++++++++++++++++ property set ++++++++++++++++++++++++++++++++++++++++')

args = parser.parse_args()
# --- train --- #
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch = args.train_epoch
start_epoch = 0
sep = args.sep

tag = 'else'
if args.type == 0:
    args.train_dir = '../datasets_train/thick_660/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "../datasets_test/Haze1k/Haze1k_thin/dataset/test/"
    args.test_name = 'input,target'
    tag = 'thin'
elif args.type == 1:
    args.train_dir = '../datasets_train/thick_660/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "../datasets_test/Haze1k/Haze1k_moderate/dataset/test/"
    args.test_name = 'input,target'
    tag = 'moderate'
elif args.type == 2:
    args.train_dir = '../datasets_train/thick_660/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "../datasets_test/Haze1k/Haze1k_thick/dataset/test/"
    args.test_name = 'input,target'
    tag = 'thick'
elif args.type == 3:
    args.train_dir = '../datasets_train/Dense_hazy/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "../datasets_test/Dense_hazy/test/"
    args.test_name = 'hazy,clean'
    tag = 'dense'
elif args.type == 4:
    args.train_dir = '../datasets_train/nhhaze/train/'
    args.train_name = 'hazy,clear'
    args.test_dir = "../datasets_test/nhhaze/test/"
    args.test_name = 'hazy,clean'
    tag = 'nhhaze'
elif args.type == 5:
    args.train_dir = '../datasets_train/Outdoor/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "../datasets_test/Outdoor/test/"
    args.test_name = 'hazy,clean'
    tag = 'outdoor'

print('We are training datasets: ', tag)

getTxt(args.train_dir, args.train_name, args.test_dir, args.test_name)

predict_result = args.predict_result
test_batch_size = args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
# output_dir = os.path.join(args.model_save_dir, 'output_result')

# --- Gpu device --- #
device_ids = [int(i) for i in list(filter(str.isdigit, args.gpus))]

print('use gpus ->', args.gpus)
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
if args.use_bn:
    print('we are using BatchNorm')
else:
    print('we are using InstanceNorm')
SDN = Base_Model(bn=args.use_bn)
print('SDN parameters:', sum(param.numel() for param in SDN.parameters()))
DNet = Discriminator(bn=args.use_bn)
print('Discriminator parameters:', sum(param.numel() for param in DNet.parameters()))

# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(SDN.parameters(), lr=0.0001)
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[5000, 7000, 8000], gamma=0.5)
D_optim = torch.optim.Adam(DNet.parameters(), lr=0.0001)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(D_optim, milestones=[5000, 7000, 8000], gamma=0.5)

# --- Load training data --- #
dataset = dehaze_train_dataset(args.train_dir, args.train_name, tag)
print('trainDataset len: ', len(dataset))
train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, drop_last=True,
                          num_workers=4)
# --- Load testing data --- #

test_dataset = dehaze_test_dataset(args.test_dir, args.test_name, tag)
print('testDataset len: ', len(test_dataset))
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0,
                         pin_memory=True)

# val_dataset = dehaze_val_dataset(val_dataset)
# val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Multi-GPU --- #
SDN = SDN.to(device)
SDN = torch.nn.DataParallel(SDN, device_ids=device_ids)
DNet = DNet.to(device)
DNet = torch.nn.DataParallel(DNet, device_ids=device_ids)
writer = SummaryWriter(os.path.join(args.model_save_dir, 'tensorboard'))

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True)
# vgg_model.load_state_dict(torch.load(os.path.join(args.vgg_model , 'vgg16.pth')))
vgg_model = vgg_model.features[:16].to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model)
loss_network.eval()

msssim_loss = msssim

# --- Load the network weight --- #
if args.restart:
    pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
    num = sorted([int(i.split('.')[0].split('_')[1]) for i in pkl_list])[-1]
    name = [i for i in pkl_list if 'epoch_' + str(num) + '_' in i][0]
    SDN.load_state_dict(
        torch.load(os.path.join(args.model_save_dir, name),
                   map_location="cuda:{}".format(device_ids[0])))
    print('--- {} epoch weight loaded ---'.format(num))
    start_epoch = int(num) + 1
elif args.num != '9999999':
    pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
    name = [i for i in pkl_list if 'epoch_' + str(args.num) + '_' in i][0]
    SDN.load_state_dict(
        torch.load(os.path.join(args.model_save_dir, name),
                   map_location="cuda:{}".format(device_ids[0])))
    print('--- {} epoch weight loaded ---'.format(args.num))
    start_epoch = int(args.num) + 1
else:
    print('--- no weight loaded ---')

iteration = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
pl = []
sl = []
best_psnr = 0
best_psnr_ssim = 0
best_ssim = 0
best_ssim_psnr = 0
print()
start_time = time.time()

for epoch in range(start_epoch, train_epoch):
    print('++++++++++++++++++++++++ {} Datasets +++++++ {} epoch ++++++++++++++++++++++++'.format(tag, epoch))
    scheduler_G.step()
    scheduler_D.step()
    SDN.train()
    DNet.train()
    with tqdm(total=len(train_loader)) as t:
        for (hazy, clean) in train_loader:
            # print(batch_idx)
            iteration += 1
            hazy = hazy.to(device)
            clean = clean.to(device)

            img = SDN(hazy)

            output = SDN(hazy, img, False)

            DNet.zero_grad()

            real_out = DNet(clean).mean()
            fake_out = DNet(output).mean()
            img_out = DNet(img).mean()
            D_loss = 1 - real_out - real_out + img_out + fake_out

            # no more forward
            D_loss.backward(retain_graph=True)
            SDN.zero_grad()

            adversarial_loss = torch.mean(1 - fake_out)
            smooth_loss_l1 = F.smooth_l1_loss(output, clean)
            perceptual_loss = loss_network(output, clean)
            msssim_loss_ = -msssim_loss(output, clean, normalize=True)

            # adversarial_loss_img = torch.mean(1 - img_out)
            smooth_loss_l1_img = F.smooth_l1_loss(img, clean)
            # perceptual_loss_img = loss_network(img, clean)
            # msssim_loss_img = -msssim_loss(img, clean, normalize=True)

            adversarial_loss_co = torch.mean(fake_out - img_out)
            smooth_loss_l1_co = F.smooth_l1_loss(img, output)
            perceptual_loss_co = loss_network(img, output)
            msssim_loss_co = -msssim_loss(img, output, normalize=True)

            total_loss = smooth_loss_l1 + 0.01 * perceptual_loss + 0.0005 * adversarial_loss + 0.5 * msssim_loss_ + 1.5 * smooth_loss_l1_img + 1.5 * (
                    smooth_loss_l1_co + 0.01 * perceptual_loss_co + 0.5 * msssim_loss_co)  # 0.01 * perceptual_loss_img + 0.0005 * adversarial_loss_img + 0.5 * msssim_loss_img +

            total_loss.backward()
            D_optim.step()
            G_optimizer.step()

            #         if iteration % 2 == 0:
            #             frame_debug = torch.cat(
            #                 (hazy, output, clean), dim=0)
            #             writer.add_images('train_debug_img', frame_debug, iteration)
            writer.add_scalars('training', {'training total loss': total_loss.item()
                                            }, iteration)
            writer.add_scalars('training_img', {'img loss_l1': smooth_loss_l1.item(),
                                                'perceptual': perceptual_loss.item(),
                                                'msssim': msssim_loss_.item()

                                                }, iteration)
            writer.add_scalars('GAN_training', {
                'd_loss': D_loss.item(),
                'd_score': real_out.item(),
                'g_score': fake_out.item()
            }, iteration
                               )
            step_2_loss = smooth_loss_l1 + 0.01 * perceptual_loss + 0.0005 * adversarial_loss + 0.5 * msssim_loss_
            step_1_loss = 1.5 * smooth_loss_l1_img
            co_loss = 1.5 * (smooth_loss_l1_co + 0.01 * perceptual_loss_co + 0.5 * msssim_loss_co)
            t.set_description(
                "===> Epoch[{}] : step_1_loss: {:.2f} step_2_loss: {:.2f} co_loss: {:.2f} G_loss: {:.2f} D_loss: {:.2f} ".format(
                    epoch, step_1_loss.item(), step_2_loss.item(), co_loss.item(), total_loss.item(), D_loss.item(),
                    time.time() - start_time))
            t.update(1)

    if args.seps:
        torch.save(SDN.state_dict(),
                   os.path.join(args.model_save_dir,
                                'epoch_' + str(epoch) + '_' + '.pkl'))
        continue

    if tag in ['outdoor']:
        if epoch >= 30:
            sep = 1
    elif tag in ['thin', 'thick', 'moderate']:
        if epoch >= 100:
            sep = 1
    else:
        if epoch >= 500:
            sep = 1

    if epoch % sep == 0:

        with torch.no_grad():
            psnr_list = []
            ssim_list = []
            SDN.eval()
            for (hazy, clean, _) in tqdm(test_loader):
                hazy = hazy.to(device)
                clean = clean.to(device)

                h, w = hazy.shape[2], hazy.shape[3]
                max_h = int(math.ceil(h / 4)) * 4
                max_w = int(math.ceil(w / 4)) * 4
                hazy, ori_left, ori_right, ori_top, ori_down = padding_image(hazy, max_h, max_w)

                img = SDN(hazy)
                frame_out = SDN(hazy, img, False)

                frame_out = frame_out.data[:, :, ori_top:ori_down, ori_left:ori_right]
                # frame_out_up = SDN(hazy_up)
                # frame_out_down = SDN(hazy_down)
                # frame_out=(torch.cat([frame_out_up.permute(0,2,3,1), frame_out_down[:,:,80:640,:].permute(0,2,3,1)], 1)).permute(0,3,1,2)

                #                 imwrite(frame_out, output_dir +'/' +str(batch_idx) + '.png', range=(0, 1))

                psnr_list.extend(to_psnr(frame_out, clean))
                ssim_list.extend(to_ssim_skimage(frame_out, clean))

            avr_psnr = sum(psnr_list) / len(psnr_list)
            avr_ssim = sum(ssim_list) / len(ssim_list)
            pl.append(avr_psnr)
            sl.append(avr_ssim)
            if avr_psnr >= max(pl):
                best_epoch_psnr = epoch
                best_psnr = avr_psnr
                best_psnr_ssim = avr_ssim
            if avr_ssim >= max(sl):
                best_epoch_ssim = epoch
                best_ssim = avr_ssim
                best_ssim_psnr = avr_psnr

            print(epoch, 'dehazed', avr_psnr, avr_ssim)
            if best_epoch_psnr == best_epoch_ssim:
                print('best epoch is {}, psnr: {}, ssim: {}'.format(best_epoch_psnr, best_psnr, best_ssim))
            else:
                print('best psnr epoch is {}: PSNR: {}, ssim: {}'.format(best_epoch_psnr, best_psnr, best_psnr_ssim))
                print('best ssim epoch is {}: psnr: {}, SSIM: {}'.format(best_epoch_ssim, best_ssim_psnr, best_ssim))
            print()
            frame_debug = torch.cat((frame_out, clean), dim=0)
            writer.add_images('my_image_batch', frame_debug, epoch)
            writer.add_scalars('testing', {'testing psnr': avr_psnr,
                                           'testing ssim': avr_ssim
                                           }, epoch)
            if best_epoch_psnr == epoch or best_epoch_ssim == epoch:
                torch.save(SDN.state_dict(),
                           os.path.join(args.model_save_dir,
                                        'epoch_' + str(epoch) + '_' + str(round(avr_psnr, 2)) + '_' + str(
                                            round(avr_ssim, 3)) + '_' + str(tag) + '.pkl'))
os.remove(os.path.join(args.train_dir, 'train.txt'))
os.remove(os.path.join(args.test_dir, 'test.txt'))
writer.close()
