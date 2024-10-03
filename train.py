import os
import torch
import argparse
import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from dataloader.dataloader import dataloader
from metrics.stats import compute_stats
from utils.utils import viz_map
from models import *

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default="/dataset/CoNIC", help='The path of the data set.')
parser.add_argument('--model', choices=['RepSNet_L', 'RepSNet_S', 'HoverNet', 'DCAN', 'MicroNet', 'ResUNet', 'UNet', 'MaskRCNN', 'RepSNet_noLbq', 'RepSNet_noRepUpsample', 'RepSNet_noRepVgg', 'RepSNet_outside'], default='RepSNet', help='The model to be used for training.')
parser.add_argument('--dataset_name', choices=['kumar', 'cpm17', 'consep', 'CoNIC', 'PanNuke', 'dsb18'], default="CoNIC", help='The name of the data set.')
parser.add_argument('--log_path', type=str, default='./model_log/', help='The path to save the model data after training.')
parser.add_argument('--batch_size', type=int, default=4, help='The size of the batch.')
parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--run_device', choices=['cpu', 'cuda'], default='cuda', help='The device where the program is running.')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
np.random.seed(322)
torch.manual_seed(322)
torch.cuda.manual_seed(322)


def run_once(
    epoch,
    net,
    loader,
    optimizer,
    writer,
    run_mode="train",
    device='cuda',
    net_args={},
):

    pbar_format = "{l_bar}|{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    pbarx = tqdm(total=len(loader[run_mode]), bar_format=pbar_format, ascii=True, position=0)

    if run_mode == "train":
        net.train()
    else:
        net.eval()

    loss_sum = 0

    pred_array = []
    true_array = []

    for step, feed_dict in enumerate(loader[run_mode]):
        imgs = feed_dict["img"].to(device).type(torch.float32)  # to NCHW
        imgs = imgs.permute(0, 3, 1, 2).contiguous()

        if run_mode == "train":
            out = net(imgs)
            loss, loss_info = net.get_loss(out, feed_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        else:
            with torch.no_grad():
                out = net(imgs)
                loss, loss_info = net.get_loss(out, feed_dict)

                # 获取实例
                ann_pred = net.get_ann(out, net_args)
                ann_true = feed_dict["ann"].data.numpy()

                pred_array.append(ann_pred)
                true_array.append(ann_true)

        loss_sum += loss.item()
        pbarx.set_description('EPOCH=%d|%s|loss=%.3f' % (epoch, run_mode, loss_sum / (step + 1)))
        pbarx.update()

        n_iter = (epoch - 1) * len(loader[run_mode]) + step + 1
        writer.add_scalar('%s/loss' % run_mode, loss.to("cpu").data.numpy(), n_iter)
        for key in loss_info:
            writer.add_scalar('%s/%s' % (run_mode, key), loss_info[key].to("cpu").data.numpy(), n_iter)

    torch.save(net.state_dict(), args.log_path + model_name + '/weight_temp.pkl')
        
    out["ann_map"] = net.get_ann(out, net_args)
    output_img = viz_map(out, feed_dict)
    writer.add_images('%s/images' % run_mode, output_img, epoch, dataformats="NWHC")

    pbarx.close()

    other_info = None
    if run_mode != "train":
        pred_array = np.concatenate(pred_array)
        true_array = np.concatenate(true_array)

        other_info = {
            "pred_array": pred_array,
            "true_array": true_array,
        }

    return loss_sum / (step + 1), other_info


if __name__ == "__main__":
    args = parser.parse_args()

    # 模型其他参数
    net_args = {
        "dataset_seed": 0,
        "dataset_name": args.dataset_name,
        "filters": [32, 64, 128, 256, 512], #[128, 256, 512, 1024], #[64, 128, 256, 512] , [96, 192, 384, 768, 1536]
        "with_augs": True,
        "with_instances_aug": False,
        "num_classes": {
            "CoNIC": 7,
            "PanNuke": 6,
            "consep": 5,
            "kumar": 2,
            "cpm17": 2,
            "dsb18": 2,
        },
        "boundary_threshold": {
            "CoNIC": 3,
            "PanNuke": 3,
            "consep": 4,
            "kumar": 5,
            "cpm17": 5,
            "dsb18": 5,
        },
        "instance_threshold": [8, 1000],
        "lr":3e-4,
    }

    model_name = args.dataset_name + '(%d)-' % net_args["dataset_seed"] + args.model
    print(model_name)
    tbpath = os.path.join(args.log_path, model_name, "tensorboard",datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(tbpath, exist_ok=True)
    writer = SummaryWriter(log_dir=tbpath)

    # 加载数据集
    loader = dataloader(
        dataset_name=args.dataset_name,
        dataset_path=args.data_path,
        batch_size=args.batch_size,
        with_augs=net_args["with_augs"],
        with_instances_aug=net_args["with_instances_aug"],
        boundary_mode="outside" if (args.model == "RepSNet_outside" or args.model == "DCAN") else "inside",
    ).getDataloader(seed=net_args["dataset_seed"])

    # 选择的模型
    if args.model == "RepSNet":
        net = RepSNet_L(
            img_channel=3,
            filters=net_args["filters"],
            num_classes=net_args["num_classes"][args.dataset_name],
            deploy=False,
        ).to(args.run_device)
    elif args.model == "RepSNet_S":
        net = RepSNet_S(
            img_channel=3,
            filters=net_args["filters"],
            num_classes=net_args["num_classes"][args.dataset_name],
            deploy=True,
        ).to(args.run_device)
    elif args.model == "HoverNet":
        net = HoVerNetConic(
            num_types=net_args["num_classes"][args.dataset_name],
            pretrained_backbone="./models/HoverNet/resnet50-pretrained.pth",
        ).to(args.run_device)
    elif args.model == "DCAN":
        net = DCAN(
            channel=3,
            num_classes=net_args["num_classes"][args.dataset_name],
            filters=net_args["filters"],
        ).to(args.run_device)
    elif args.model == "MicroNet":
        net = MicroNet(
            num_input_channels=3,
            num_class=net_args["num_classes"][args.dataset_name],
        ).to(args.run_device)
    elif args.model == "UNet":
        net = UNet(
            channel=3,
            num_classes=net_args["num_classes"][args.dataset_name],
            filters=net_args["filters"],
            ResUNet=args.model == "ResUNet",
        ).to(args.run_device)
    elif args.model == "StarDist":
        net = StarDist(
            n_channels=3,
        ).to(args.run_device)

# ----------------------------------
    # save_dict =torch.load("/model_log/best_mpq.pkl")
    # net.load_state_dict(save_dict)

    
    optimizer = torch.optim.Adam(net.parameters(), lr=net_args["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_epochs // 3, gamma=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, verbose=1)
    best_loss = np.inf
    best_mpq =0

   

    for epoch in range(1, args.n_epochs + 1):

        train_loss, _ = run_once(
            epoch=epoch,
            net=net,
            loader=loader,
            optimizer=optimizer,
            writer=writer,
            run_mode="train",
            device=args.run_device,
            net_args=net_args,
        )
      

        valid_loss, other_info = run_once(
            epoch=epoch,
            net=net,
            loader=loader,
            optimizer=optimizer,
            writer=writer,
            run_mode="test",
            device=args.run_device,
            net_args=net_args,
        )
        lr_scheduler.step(valid_loss)
        # 评估模型
        metrics = compute_stats(other_info["pred_array"], other_info["true_array"], net_args["num_classes"][args.dataset_name])
        # lr_scheduler.step(metrics["mpq"])

        for key in metrics.keys():
            writer.add_scalar('metrics/%s' % key, metrics[key], epoch)

        if epoch == 16:
            convert_net = model_convert(net, do_copy=True)
            torch.save(convert_net.state_dict(), args.log_path + model_name + '/weight_epoch16.pkl')
            torch.save(net.state_dict(), args.log_path + model_name + '/weight_epoch16_noconvert.pkl')
        # 保存模型
        if args.model == "RepSNet" or args.model == "RepSNet_noLbq" or args.model == "RepSNet_noRepUpsample":
            convert_net = model_convert(net, do_copy=True)
        else:
            convert_net = net
        torch.save(convert_net.state_dict(), args.log_path + model_name + '/weight_temp.pkl')
        
        if valid_loss < best_loss:
            torch.save(convert_net.state_dict(), args.log_path + model_name + '/weight_best.pkl')
            best_loss = valid_loss
            pickle.dump(other_info, open(args.log_path + model_name + '/other_info.pkl', 'wb'))

        if metrics['mpq'] > best_mpq:
            torch.save(convert_net.state_dict(), args.log_path + model_name + '/best_mpq.pkl')
            best_mpq = metrics['mpq']
            pickle.dump(other_info, open(args.log_path + model_name + '/other_info.pkl', 'wb'))
