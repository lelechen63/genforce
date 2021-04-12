# python3.7
"""A simple tool to synthesize images with pre-trained models."""

import os
import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
import cv2

from models import MODEL_ZOO
from models import build_generator
from utils.misc import bool_parser
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image
from utils.utils import align_face


def postprocess(images):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
    images = images.detach().cpu().numpy()
    images = (images + 1) * 255 / 2
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    return images


def preprocess(images):
    """Post-processes images from `numpy.ndarray` to `torch.Tensor`."""
    images = images.transpose(2,0,1)
    images = (images - 0.5) / 255.0 * 2 - 1
    images = np.clip(images , -1.0, 1.0)
    # images = torch.FloatTensor(images)
    return images

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Synthesize images with pre-trained models.')
    parser.add_argument('model_name', type=str,
                        help='Name to the pre-trained model.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save the results. If not specified, '
                             'the results will be saved to '
                             '`work_dirs/synthesis/` by default. '
                             '(default: %(default)s)')
    parser.add_argument('--num', type=int, default=100,
                        help='Number of samples to synthesize. '
                             '(default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='Batch size. (default: %(default)s)')

    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate. (default: %(default)s)')

    parser.add_argument('--max_iter_num', type=int, default=1000,
                        help='maximun optimization number. (default: %(default)s)')

    parser.add_argument('--device_ids', type=str, default='0')


    parser.add_argument('--generate_html', type=bool_parser, default=True,
                        help='Whether to use HTML page to visualize the '
                             'synthesized results. (default: %(default)s)')
    parser.add_argument('--save_raw_synthesis', type=bool_parser, default=False,
                        help='Whether to save raw synthesis. '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--randomize_noise', type=bool_parser, default=False,
                        help='Whether to randomize the layer-wise noise. This '
                             'is particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    return parser.parse_args()






def load_data():
    """ load the video data"""
    img_path = '/home/cxu-serve/p1/lchen63/nerf/data/mead/001/original'
    img_names = os.listdir(img_path)
    img_names.sort()
    gt_imgs = []
    for i in range(len(img_names)):
        if i == 4:
            break
        img_p = os.path.join( img_path, img_names[i])
        # align_face(img_p)
        aligned_img = cv2.imread(img_p.replace( 'original', 'aligned'))
        # print (aligned_img.shape)
        gt_imgs.append(preprocess(aligned_img))
    gt_imgs = np.asarray(gt_imgs)
    gt_imgs = torch.FloatTensor(gt_imgs)
    return gt_imgs


    


def main():
    """Main function."""
    args = parse_args()
   
    if not args.save_raw_synthesis and not args.generate_html:
        return

    # Parse model configuration.
    if args.model_name not in MODEL_ZOO:
        raise SystemExit(f'Model `{args.model_name}` is not registered in '
                         f'`models/model_zoo.py`!')
    model_config = MODEL_ZOO[args.model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.

    # Get work directory and job name.
    if args.save_dir:
        work_dir = args.save_dir
    else:
        work_dir = os.path.join('work_dirs', 'synthesis')
    os.makedirs(work_dir, exist_ok=True)
    job_name = f'{args.model_name}_{args.num}'
    if args.save_raw_synthesis:
        os.makedirs(os.path.join(work_dir, job_name), exist_ok=True)

    # Build generation and get synthesis kwargs.
    print(f'Building generator for model `{args.model_name}` ...')
    generator = build_generator(**model_config)
    synthesis_kwargs = dict(trunc_psi=args.trunc_psi,
                            trunc_layers=args.trunc_layers,
                            randomize_noise=args.randomize_noise)
    print(f'Finish building generator.')

    # Load pre-trained weights.
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', args.model_name + '.pth')
    print(f'Loading checkpoint from `{checkpoint_path}` ...')
    if not os.path.exists(checkpoint_path):
        print(f'  Downloading checkpoint from `{url}` ...')
        subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
        print(f'  Finish downloading checkpoint.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    
    code_dim = generator.z_space_dim
    # generator = generator.cuda()
    device_ids = [int(i) for i in args.device_ids.split(',')]

    generator = torch.nn.DataParallel(generator, device_ids=device_ids).cuda()

    generator.eval()
    print(f'Finish loading checkpoint.')

    # Set random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load gt images:
    gt_imgs = load_data().cuda()
    print (gt_imgs.shape)

    total_num = gt_imgs.shape[0]
    # define the code
    code = np.zeros( ( total_num, code_dim ), dtype = np.float32)
   

    for batch_idx in tqdm(range(0, total_num, args.batch_size)): 
        frm_idx_start = batch_idx
        frm_idx_end = np.min( [ total_num, frm_idx_start + args.batch_size ] )
        sample_num = frm_idx_end - frm_idx_start

        print( 'Optim lightingcode based on pca batch [%d/%d], frame [%d/%d]'
               % (batch_idx/args.batch_size + 1, total_num, frm_idx_start, frm_idx_end) )

        # get code batch
        batch_code = [ ]
        for sample_id in range( sample_num  ):
            frm_idx = frm_idx_start + sample_id
            batch_code.append( code[ frm_idx ] )
        batch_code = np.asarray(batch_code)
        batch_code = torch.tensor( batch_code, dtype = torch.float32 ).cuda()
        batch_code.requires_grad = True

        # get gt image batch
        batch_gt_img =gt_imgs[frm_idx_start : frm_idx_end]

        # Define the optimizer
        code_optimizer = torch.optim.Adam( [
            { 'params': batch_code, 'lr': args.lr }
        ] )

        # optimize
        for iter_id in range( args. max_iter_num ):
            images = generator(batch_code, **synthesis_kwargs)['image']
            # SIZE: (BATCH_SIZE , 3,1024,1024) 

            # calculate loss:
            global_pix_loss = (batch_gt_img  - images).abs().mean()
            code_reg_loss = (batch_code ** 2).mean()
            loss = global_pix_loss + code_reg_loss * 0.01

            # Optimize
            code_optimizer.zero_grad()
            loss.backward()
            code_optimizer.step()

            # Print errors
            if iter_id == 0 or ( iter_id + 1 ) % 100 == 0:
                print(  ' iter [%d/%d]: global %f, code_reg %f'
                   % (  iter_id + 1, args.max_iter_num ,  global_pix_loss.item(),
                       code_reg_loss.item()), end = '\r' )
    
        # Set the data
        n_code = batch_code.detach().cpu().numpy()
        for sample_id in range( sample_num ):
            frm_idx = frm_idx_start + sample_id
            code[ frm_idx ] = n_code[ sample_id ]


    # Sample and synthesize.
    print(f'Synthesizing {total_num} samples ...')
    indices = list(range(total_num))
    if args.generate_html:
        html = HtmlPageVisualizer(grid_size=total_num)
    for batch_idx in tqdm(range(0, total_num, args.batch_size)):
        sub_indices = indices[batch_idx:batch_idx + args.batch_size]
        code = torch.FloatTensor(code).cuda()
        # code = torch.randn(len(sub_indices), generator.z_space_dim).cuda()
        with torch.no_grad():
            images = generator(code, **synthesis_kwargs)['image']
            # images shape [1,3,1024,1024]
            print (images.shape)

            images = postprocess(images)
        for sub_idx, image in zip(sub_indices, images):
            if args.save_raw_synthesis:
                save_path = os.path.join(
                    work_dir, job_name, f'{sub_idx:06d}.jpg')
                save_image(save_path, image)
            if args.generate_html:
                row_idx, col_idx = divmod(sub_idx, html.num_cols)
                html.set_cell(row_idx, col_idx, image=image,
                              text=f'Sample {sub_idx:06d}')
    if args.generate_html:
        html.save(os.path.join(work_dir, f'{job_name}.html'))
    print(f'Finish synthesizing {total_num} samples.')

     # Sample and ground truth image.
    print(f'Synthesizing {total_num} samples ...')
    indices = list(range(total_num))
    if args.generate_html:
        html = HtmlPageVisualizer(grid_size=total_num)
    for batch_idx in tqdm(range(0, total_num, args.batch_size)):
        sub_indices = indices[batch_idx:batch_idx + args.batch_size]
        code = torch.FloatTensor(code).cuda()
        # code = torch.randn(len(sub_indices), generator.z_space_dim).cuda()
        with torch.no_grad():
            images = generator(code, **synthesis_kwargs)['image']
            # images shape [1,3,1024,1024]
            print (images.shape)

            images = postprocess(images)
        for sub_idx, image in zip(sub_indices, images):
            if args.save_raw_synthesis:
                save_path = os.path.join(
                    work_dir, job_name, f'{sub_idx:06d}.jpg')
                save_image(save_path, image)
            if args.generate_html:
                row_idx, col_idx = divmod(sub_idx, html.num_cols)
                html.set_cell(row_idx, col_idx, image=image,
                              text=f'Sample {sub_idx:06d}')
    if args.generate_html:
        html.save(os.path.join(work_dir, f'{job_name}_gt.html'))
    print(f'Finish synthesizing {total_num} samples.')


if __name__ == '__main__':
    load_data()
    main()
