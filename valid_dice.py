import os

import torch

import flare_loader

import torchio as tio

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def validate_metricss( val_model):
    # residualUNet3D = ResidualUNet3D()
    # return 0

    val_label_path = os.path.join("data","FLARE22_LabeledCase50","images")+os.sep
    val_ct_path = os.path.join("data","FLARE22_LabeledCase50","labels")+os.sep
    val_ds = flare_loader.CustomValidImageDataset([val_ct_path + i for i in os.listdir(val_ct_path)],
                                                 [val_label_path + i for i in os.listdir(val_label_path)],
                                                 tio.transforms.Compose([tio.Resize(target_shape=(128, 128, 128))]),
                                                 tio.transforms.Compose([tio.Resize(target_shape=(128, 128, 128))]))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1,
        num_workers=4, pin_memory=True)
    val_model.eval()
    losses = AverageMeter("val_loss", ':.4e')
    dice_coefficients = AverageMeter("dice", ':.4e')
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            with torch.cuda.amp.autocast(True):
                pred = val_model(data)
                pred = torch.argmax(pred, 1)
                dice_coefficients.update(flare_loader.compute_dice_coefficient(pred.cpu().numpy().astype(int), target.cpu().numpy()), 1)
    print("DICE:", dice_coefficients)
    # print("validate")
    return dice_coefficients.avg
    pass