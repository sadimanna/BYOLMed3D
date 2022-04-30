from sklearn.manifold import TSNE
import pickle, os, random, torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import math
#%matplotlib inline

def run_command(command):
    sp = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    print(out_str[0].decode("utf-8"))

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def seed_everything(seed, workers = True):
    print(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

def save_model(model, current_epoch, save_path, name):
    out = os.path.join(save_path,name.format(current_epoch))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict':model.optimizer.state_dict(),
                'scheduler_state_dict':model.scheduler.state_dict()}, out)

    return out

def load_model(model, model_path):
    state = torch.load(model_path)
    model.load_state_dict(state['model_state_dict'], strict = False)

def plot_features(feats, labels, num_classes, epoch):
    tsne = TSNE(n_components = 2, perplexity = 50)
    x_feats = tsne.fit_transform(feats)
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1,1,1)
    for i in range(num_classes):
        #plt.scatter(x_feats[val_df['class'].iloc[:num_samples].values==i,1],x_feats[val_df['class'].iloc[:num_samples].values==i,0])
        ax.scatter(x_feats[labels==i,1],x_feats[labels==i,0])
    ax.legend([str(i) for i in range(num_classes)])
    ax.set_title('TSNE-fied fearures of Pre-train model at Epoch : '+str(epoch))
    #plt.show()
    return fig

def plot_metrics(tr, val, title):
    fig= plt.figure(figsize=(50,50))
    ax = fig.add_subplot(1,1)
    t = ax.plot(tr, 'r-', label = 'train')
    v = ax.plot(val, 'b-', label = 'valid')
    ax.legend(handles = [t,v]) #, labels = ['train','valid'])
    ax.set_title(title)
    plt.show()
    return fig

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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
