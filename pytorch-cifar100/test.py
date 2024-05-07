""" test model using pytorch """

import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from conf import settings
from utils import get_network, get_test_dataloader
import torch.utils.benchmark as benchmark
from ptflops import get_model_complexity_info
import time
from torchsummary import summary

if __name__ == '__main__':

    torch.manual_seed(42) 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
    )
    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.to('cuda')
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    # Warm-up
    for image, label in cifar100_test_loader:
        if args.gpu:
            image = image.cuda()
            label = label.cuda()
        output = net(image)
        break

    # Inference time measurement
    start_time = time.time()
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {} \t total {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
            if args.gpu:
                image = image.cuda()
                label = label.cuda()
            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            correct_5 += correct[:, :5].sum()
            correct_1 += correct[:, :1].sum()
    end_time = time.time()

    inference_time = end_time - start_time
    print()
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
    macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True)
    summary(net, (3, 32, 32))
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print(f"FLOPs: {macs}")
    print(f"Params: {params}")
    print(f"Inference time: {inference_time:.4f} seconds")
    total_steps = len(cifar100_test_loader)
    time_per_step_ms = (inference_time / total_steps) * 1000
    print(f"Time per inference step: {time_per_step_ms:.4f} milliseconds")