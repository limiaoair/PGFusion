import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.nn as nn

from evaluator import Eval_thread
from dataloader import EvalDataset

import sys
sys.path.append('..')
from config import Config


styles = ['.-r', '.--b', '.--g', '.--c', '.-m', '.-y', '.-k', '.-c']
lines = ['-', '--', '--', '--', '-', '-', '-', '-']
points = ['*', '.', '.', '.', '.', '.', '.', '.']
colors = ['r', 'b', 'g', 'c', 'm', 'orange', 'k', 'navy']


def main_plot(cfg):
    method_names = cfg.methods.split('+')
    dataset_names = cfg.datasets.split('+')
    os.makedirs(cfg.output_figure, exist_ok=True)
    # plt.style.use('seaborn-white')

    # Plot PR Cureve
    for dataset in dataset_names:
        plt.figure()
        idx_style = 0
        for method in method_names:
            iRes = loadmat(os.path.join(cfg.output_dir, method, 'final', dataset + '.mat'))
            imax = np.argmax(iRes['Fm'])
            plt.plot(
                iRes['Recall'],
                iRes['Prec'],
                #  styles[idx_style],
                color=colors[idx_style],
                linestyle=lines[idx_style],
                marker=points[idx_style],
                markevery=[imax, imax],
                label=method)
            idx_style += 1

        plt.grid(True, zorder=-1)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1.02)
        plt.ylabel('Precision', fontsize=25)
        plt.xlabel('Recall', fontsize=25)

        plt.legend(loc='lower left', prop={'size': 15})
        plt.savefig(os.path.join(cfg.output_figure, 'PR_' + dataset + '.png'),
                    dpi=600,
                    bbox_inches='tight')
        plt.close()

    # Plot Fm Cureve
    for dataset in dataset_names:
        plt.figure()
        idx_style = 0
        for method in method_names:
            iRes = loadmat(os.path.join(cfg.output_dir, method, 'final', dataset + '.mat'))
            imax = np.argmax(iRes['Fm'])
            plt.plot(
                np.arange(0, 255),
                iRes['Fm'],
                #  styles[idx_style],
                color=colors[idx_style],
                linestyle=lines[idx_style],
                marker=points[idx_style],
                label=method,
                markevery=[imax, imax])
            idx_style += 1
        plt.grid(True, zorder=-1)
        # plt.ylim(0, 1)
        plt.ylabel('F-measure', fontsize=25)
        plt.xlabel('Threshold', fontsize=25)

        plt.legend(loc='lower left', prop={'size': 15})
        plt.savefig(os.path.join(cfg.output_figure, 'Fm_' + dataset + '.png'),
                    dpi=600,
                    bbox_inches='tight')
        plt.close()

    # Plot Em Cureve
    for dataset in dataset_names:
        plt.figure()
        idx_style = 0
        for method in method_names:
            iRes = loadmat(os.path.join(cfg.output_dir, method, 'final', dataset + '.mat'))
            imax = np.argmax(iRes['Em'])
            plt.plot(
                np.arange(0, 255),
                iRes['Em'],
                #  styles[idx_style],
                color=colors[idx_style],
                linestyle=lines[idx_style],
                marker=points[idx_style],
                label=method,
                markevery=[imax, imax])
            idx_style += 1
        plt.grid(True, zorder=-1)
        plt.ylim(0, 1)
        plt.ylabel('E-measure', fontsize=16)
        plt.xlabel('Threshold', fontsize=16)

        plt.legend(loc='lower left', prop={'size': 15})
        plt.savefig(os.path.join(cfg.output_figure, 'Em_' + dataset + '.png'),
                    dpi=600,
                    bbox_inches='tight')
        plt.close()

    # Plot ROC Cureve
    for dataset in dataset_names:
        plt.figure()
        idx_style = 0
        for method in method_names:
            iRes = loadmat(os.path.join(cfg.output_dir, method, 'final', dataset + '.mat'))
            imax = np.argmax(iRes['Fm'])
            plt.plot(
                iRes['FPR'],
                iRes['TPR'],
                #  styles[idx_style][1:],
                color=colors[idx_style],
                linestyle=lines[idx_style],
                label=method)
            idx_style += 1

        plt.grid(True, zorder=-1)
        plt.xlim(0, 1)
        plt.ylim(0, 1.02)
        plt.ylabel('TPR', fontsize=16)
        plt.xlabel('FPR', fontsize=16)

        plt.legend(loc='lower right')
        plt.savefig(os.path.join(cfg.output_figure, 'ROC_' + dataset + '.png'),
                    dpi=600,
                    bbox_inches='tight')
        plt.close()

    # Plot Sm-MAE
    for dataset in dataset_names:
        plt.figure()
        plt.gca().invert_xaxis()
        idx_style = 0
        for method in method_names:
            iRes = loadmat(os.path.join(cfg.output_dir, method, 'final', dataset + '.mat'))
            plt.scatter(iRes['MAE'],
                        iRes['Sm'],
                        marker=points[idx_style],
                        c=colors[idx_style],
                        s=120)
            plt.annotate(method,
                         xy=(iRes['MAE'], iRes['Sm']),
                         xytext=(iRes['MAE'] - 0.001, iRes['Sm'] - 0.001),
                         fontsize=14)
            idx_style += 1

        plt.grid(True, zorder=-1)
        # plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.ylabel('S-measure', fontsize=16)
        plt.xlabel('MAE', fontsize=16)
        plt.savefig(os.path.join(cfg.output_figure, 'Sm-MAE_' + dataset + '.png'),
                    bbox_inches='tight')
        plt.close()


def main(cfg):
    if cfg.methods is None:
        method_names = os.listdir(cfg.pred_dir)
    else:
        method_names = cfg.methods.split('+')
    if cfg.datasets is None:
        dataset_names = os.listdir(cfg.gt_dir)
    else:
        dataset_names = cfg.datasets.split('+')

    num_model_eval = Config().val_last
    threads = []
    # model -> ckpt -> dataset
    for method in method_names:
        epochs = os.listdir(os.path.join(cfg.pred_dir, method))[-num_model_eval:][::-1]
        for epoch in epochs:
            continue_eval = True
            for dataset in dataset_names:
                loader = EvalDataset(
                    os.path.join(cfg.pred_dir, method, epoch, dataset),        # preds
                    os.path.join(cfg.gt_dir, dataset)                   # GT
                )
                print('Evaluating predictions from {}'.format(os.path.join(cfg.pred_dir, method, epoch, dataset)))
                thread = Eval_thread(loader, method, dataset, cfg.output_dir, epoch, cfg.cuda)
                info, continue_eval = thread.run(continue_eval=continue_eval)
                print(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='GCoNet_ext')
    parser.add_argument('--datasets', type=str, default='CoCA+CoSOD3k+CoSal2015')

    parser.add_argument('--gt_dir', type=str, default='/home/lm/project/co-sod/DCFM-master/data/gts', help='GT')
    parser.add_argument('--pred_dir', type=str, default='/home/lm/project/co-sod/DCFM-master/results/0828', help='predictions')
    parser.add_argument('--output_dir', type=str, default='./output_measure/details', help='saving measurements here.')
    parser.add_argument('--output_figure', type=str, default='./output_measure/figures', help='saving figures here.')

    parser.add_argument('--cuda', type=bool, default=True)
    config = parser.parse_args()
    main(config)
