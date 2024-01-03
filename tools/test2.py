# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import glob

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from test import trigger_visualization_hook


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument(
        'work_dir', help=('Path of config file, checkpoint file and save dir'))
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def mod_args(args):
    if not os.path.exists(args.work_dir):
        print('work_dir not exist, %s' % args.work_dir)
        exit(0)

    # find config file in work dir
    config_file_pattern = os.path.join(args.work_dir, '*.py')
    config_files = glob.glob(config_file_pattern)
    assert (len(config_files) == 1)
    args.config = config_files[0]

    # find checkpoint
    if args.checkpoint.endswith('.pth'):
        args.checkpoint = os.path.join(args.work_dir, args.checkpoint)
    else:
        ck_pattern = os.path.join(args.work_dir, '*%s*.pth' % args.checkpoint)
        ck_files = glob.glob(ck_pattern)
        assert (len(ck_files) == 1)
        args.checkpoint = ck_files[0]

    args.work_dir = os.path.join(args.work_dir, 'test')
    print('mod_args', '=' * 50)
    print('args.config = %s' % args.config)
    print('args.checkpoint = %s' % args.checkpoint)
    print('args.work_dir = %s' % args.work_dir)
    return args


def mod_cfg(cfg):
    # 设置单线程加载数据，用于debug
    # cfg.test_dataloader.num_workers = 0
    # cfg.test_dataloader.persistent_workers = False

    # 删除wandb
    cfg.visualizer.vis_backends = [bk for bk in cfg.visualizer.vis_backends if bk.type != 'WandbVisBackend']


def main():
    args = parse_args()
    # mod args by alex
    args = mod_args(args)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # mod cfg by alex
    mod_cfg(cfg)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
