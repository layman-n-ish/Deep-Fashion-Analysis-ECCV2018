from src.dataset import DeepFashionCAPDataset
from src.const import base_path
import pandas as pd
import torch
import torch.utils.data
from src import const
from src.utils import parse_args_and_merge_const
from tensorboardX import SummaryWriter
import os
import time
import math

if __name__ == '__main__':
    parse_args_and_merge_const()
    if os.path.exists('models') is False:
        os.makedirs('models')

    df = pd.read_csv(base_path + const.USE_CSV)
    train_df = df[df['evaluation_status'] == 'train']
    train_dataset = DeepFashionCAPDataset(train_df, mode=const.DATASET_PROC_METHOD_TRAIN)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True, num_workers=const.N_WORKERS)
    val_df = df[df['evaluation_status'] == 'test']
    val_dataset = DeepFashionCAPDataset(val_df, mode=const.DATASET_PROC_METHOD_VAL)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=const.VAL_BATCH_SIZE, shuffle=False, num_workers=const.N_WORKERS)
    val_step = len(val_dataloader)

    N_TRAIN_SAMPLES = train_df.shape[0]
    N_TEST_SAMPLES = val_df.shape[0]

    net = const.USE_NET() # the whole network
    net = net.to(const.device)

    if const.CHKPT != '':
        checkpoint = torch.load(const.CHKPT)
        learning_rate = checkpoint['learning_rate']
    else:
        learning_rate = const.LEARNING_RATE
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    writer = SummaryWriter(const.TRAIN_DIR)

    if const.CHKPT != '':
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
    else:
        epoch = 0

    total_step = len(train_dataloader)
    step = 0
    while (epoch < const.NUM_EPOCH):
        start_time = time.time()
        net.train()
        for i, sample in enumerate(train_dataloader):
            step += 1
            for key in sample:
                sample[key] = sample[key].to(const.device)
            output = net(sample)
            loss = net.cal_loss(sample, output)

            optimizer.zero_grad() # clears old gradients from the last step
            loss['all'].backward() # computes the derivative of the loss w.r.t. the parameters (backprop)
            optimizer.step() # causes the optimizer to take a step based on the gradients of the parameters

            if (i + 1) % 10 == 0:
                end_time = time.time() - start_time
                start_time = time.time()

                if 'category_loss' in loss:
                    writer.add_scalar('loss/category_loss', loss['category_loss'], step)
                    writer.add_scalar('loss_weighted/category_loss', loss['weighted_category_loss'], step)
                if 'attr_loss' in loss:
                    writer.add_scalar('loss/attr_loss', loss['attr_loss'], step)
                    writer.add_scalar('loss_weighted/attr_loss', loss['weighted_attr_loss'], step)
                if 'lm_vis_loss' in loss:
                    writer.add_scalar('loss/lm_vis_loss', loss['lm_vis_loss'], step)
                    writer.add_scalar('loss_weighted/lm_vis_loss', loss['weighted_lm_vis_loss'], step)
                if 'lm_pos_loss' in loss:
                    writer.add_scalar('loss/lm_pos_loss', loss['lm_pos_loss'], step)
                    writer.add_scalar('loss_weighted/lm_pos_loss', loss['weighted_lm_pos_loss'], step)
                writer.add_scalar('loss_weighted/all', loss['all'], step)
                writer.add_scalar('global/learning_rate', learning_rate, step)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time taken: {:.4f}'
                      .format(epoch + 1, const.NUM_EPOCH, i + 1, total_step, loss['all'].item(), end_time))
            if (i + 1) % math.ceil((N_TRAIN_SAMPLES/const.BATCH_SIZE)) == 0:
                print('Saving Model....')
                net.set_buffer('step', step)
                model_dict = {'epoch': epoch, \
                              'model_state_dict': net.state_dict(), \
                              'optim_state_dict': optimizer.state_dict(), \
                              'loss': loss, \
                              'learning_rate': learning_rate}

                torch.save(model_dict, '/home/ubuntu/deepfashion/Deep-Fashion-Analysis-ECCV2018/models/whole_' + str(epoch+1) + '.pkl')
                print('OK.')
                if const.VAL_WHILE_TRAIN:
                    print('Now Evaluate..')
                    with torch.no_grad():
                        net.eval()
                        evaluator = const.EVALUATOR()
                        for j, sample in enumerate(val_dataloader):
                            for key in sample:
                                sample[key] = sample[key].to(const.device)
                            output = net(sample)
                            evaluator.add(output, sample)
                            if (j + 1) % 100 == 0:
                                print('Val Step [{}/{}]'.format(j + 1, val_step))
                        ret = evaluator.evaluate()
                        for topk, accuracy in ret['category_accuracy_topk'].items():
                            print('metrics/category_top{}'.format(topk), accuracy)
                            writer.add_scalar('metrics/category_top{}'.format(topk), accuracy, step)

                        for topk, accuracy in ret['attr_group_recall'].items():
                            for attr_type in range(1, 6):
                                print('metrics/attr_top{}_type_{}_{}_recall'.format(
                                    topk, attr_type, const.attrtype2name[attr_type]), accuracy[attr_type - 1]
                                )
                                writer.add_scalar('metrics/attr_top{}_type_{}_{}_recall'.format(
                                    topk, attr_type, const.attrtype2name[attr_type]), accuracy[attr_type - 1], step
                                )
                            print('metrics/attr_top{}_all_recall'.format(topk), ret['attr_recall'][topk])
                            writer.add_scalar('metrics/attr_top{}_all_recall'.format(topk), ret['attr_recall'][topk], step)

                        for i in range(8):
                            print('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i])
                            writer.add_scalar('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i], step)
                        print('metrics/dist_all', ret['lm_dist'])
                        writer.add_scalar('metrics/dist_all', ret['lm_dist'], step)
                    net.train()
        
        epoch += 1
        # learning rate decay
        learning_rate *= const.LEARNING_RATE_DECAY
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

