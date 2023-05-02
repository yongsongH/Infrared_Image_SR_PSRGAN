import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
import torch

from utilss import utils_logger
from utilss import utils_image as util
from utilss import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from torchvision import models
from models import enhance_model_gan as net




def main(json_path='options/train_kdsrgan.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iterG, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iterD, init_path_D = option.find_last_checkpoint(opt['path']['models'], net_type='D')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netD'] = init_path_D
    current_step = max(init_iterG, init_iterD)

    # opt['path']['pretrained_netG'] = ''
    # current_step = 0
    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''


    
    model = net.ModelGAN(opt)

    #######################################   Step 1   #############################################################
    G_state_dict = torch.load('latest model/20000_G.pth')
    # # D_state_dict = torch.load('latest model/115000_D.pth')
    # #
    from collections import OrderedDict
    #
    G_new_state_dict = OrderedDict()
    # D_new_state_dict = OrderedDict()
    for k, v in G_state_dict.items():
        name = str('module.' + k[0:])
        #######################################   Step 2   #############################################################
        # if k != 'KD.c1_d.0.weight' and k != 'KD.c1_d.0.bias' and k != 'KD.c1_d.1.weight' and k != 'KD.c1_d.1.bias' and k != 'KD.c1_r.0.weight' and k != 'KD.c1_r.0.bias' and k != 'KD.c1_r.1.weight'and k != 'KD.c1_r.1.bias'and k != 'KD.c2_d.0.weight'and k != 'KD.c2_d.0.bias'and k != 'KD.c2_d.1.weight'and k != 'KD.c2_d.1.bias'and k != 'KD.c2_r.0.weight'and k != 'KD.c2_r.0.bias'and k != 'KD.c2_r.1.weight'and k != 'KD.c2_r.1.bias'and k != 'KD.c3_d.0.weight'and k != 'KD.c3_d.0.bias'and k != 'KD.c3_d.1.weight'and k != 'KD.c3_d.1.bias'and k != 'KD.c3_r.0.weight'and k != 'KD.c3_r.0.bias'and k != 'KD.c3_r.1.weight'and k != 'KD.c3_r.1.bias'and k != 'KD.c4.0.weight'and k != 'KD.c4.0.bias'and k != 'KD.c4.1.weight'and k != 'KD.c4.1.bias'and k != 'KD.c5.0.weight'and k != 'KD.c5.0.bias'and k != 'KD.c5.1.weight'and k != 'KD.c5.1.bias'and k != 'KD.esa.conv1.0.weight'and k != ' KD.esa.conv1.0.bias'and k != 'KD.esa.conv1.1.weight'and k != 'KD.esa.conv1.1.bias'and k != 'KD.esa.conv_f.0.weight'and k != 'KD.esa.conv_f.0.bias'and k != ' KD.esa.conv_f.1.weight'and k != 'KD.esa.conv_f.1.bias'and k != 'KD.esa.conv_max.0.weight'and k != 'KD.esa.conv_max.0.bias'and k != 'KD.esa.conv_max.1.weight'and k != 'KD.esa.conv_max.1.bias'and k != 'KD.esa.conv2.0.weight'and k != 'KD.esa.conv2.0.bias'and k != 'KD.esa.conv2.1.weight'and k != 'KD.esa.conv2.1.bias'and k != 'KD.esa.conv3.0.weight'and k != 'KD.esa.conv3.0.bias'and k != 'KD.esa.conv3.1.weight'and k != 'KD.esa.conv3.1.bias'and k != 'KD.esa.conv3_.0.weight'and k != 'KD.esa.conv3_.0.bias'and k != 'KD.esa.conv3_.1.weight'and k != 'KD.esa.conv3_.1.bias'and k != 'KD.esa.conv4.0.weight'and k != 'KD.esa.conv4.0.bias'and k != 'KD.esa.conv4.1.weight'and k != 'KD.esa.conv4.1.bias'and k != 'tail.1.weight'and k != 'tail.1.bias'and k != 'tail.4.weight'and k != 'tail.4.bias'and k != 'tail.6.weight'and k != 'tail.6.bias'and k != 'tail.8.weight'and k != 'copress.weight'and k != 'copress.bias':
        ###############################################   Step 3   ##########################################################
        # if k != 'tail.1.weight'and k != 'tail.1.bias'and k != 'tail.4.weight'and k != 'tail.4.bias'and k != 'tail.6.weight'and k != 'tail.6.bias'and k != 'tail.8.weight'and k != 'copress.weight'and k != 'copress.bias':
        #     v.requires_grad = False
        #     G_new_state_dict[name] = v
        # print(v.requires_grad)

        if k != 'tail.8.weight' and k != 'copress.weight' and k != 'copress.bias':
            v.requires_grad = False
            G_new_state_dict[name] = v
        # print(v.requires_grad)



    model.netG.load_state_dict(G_new_state_dict, strict=False)  

    # for k, v in D_state_dict.items():
    #     name = str('module.' + k[0:]) 
    #     D_new_state_dict[name] = v

    # for k, v in G_state_dict.items():
    #     name = str('module.'+k[0:] )
    #     G_new_state_dict[name] = v
    # #
    # # model.netD.load_state_dict(D_new_state_dict, strict=True)
    # model.netG.load_state_dict(G_new_state_dict, strict=False)

    model.init_train()




    logger.info(model.info_network())
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(3000):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:

                avg_psnr = 0.0
                avg_ssim = 0.0

                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    # -----------------------
                    # calculate PSNR & SSIM
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                    current_ssim = util.calculate_ssim(E_img, H_img, border=border)

                    logger.info('{:->4d}--> {:>10s} | {:<4.4f}dB | {:<4.4f}'.format(idx, image_name_ext, current_psnr, current_ssim))

                    avg_psnr += current_psnr
                    avg_ssim += current_ssim

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx

                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average PSNR : {:<.4f}\n'.format(epoch, current_step, avg_psnr, avg_ssim))

    logger.info('Saving the final model.')
    model.save('DIV2Ksub_1_RRDB+KDB+Sample_8000')
    logger.info('End of training.')


if __name__ == '__main__':
    main()


