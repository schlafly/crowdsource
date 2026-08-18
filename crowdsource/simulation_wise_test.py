#!/usr/bin/env python
"""
Injection-recovery test for single-band vs multiband crowdsource.

Functions:
  make_sim() : generate simulated W1+W2 images
  save_sim() : save images and truth catalog to FITS
  run_one()  : run fits and save output catalogs
  run_all()  : loop over crowding levels
"""

import numpy as np
import os
from astropy.io import fits
from astropy.table import Table
from crowdsource import crowdsource_base
import crowdsource.psf as psfmod


def make_sim(nstar, nx=512, ny=512, noise_w1=0.25, noise_w2=0.5, stampsz=19, seed=42):
    """
    Generate simulated W1 and W2 images with shared injected sources.

    Returns
    -------
    im_w1, im_w2 : ndarray (nx, ny)
    wt_w1, wt_w2 : ndarray (nx, ny)
    psf_w1, psf_w2 : SimplePSF objects
    x_true, y_true, flux_true : injected source parameters
    """
    np.random.seed(seed)

    from wise_proc import wise_psf_stamp
    psf_w1 = psfmod.SimplePSF(wise_psf_stamp(1).astype('f4'))
    psf_w2 = psfmod.SimplePSF(wise_psf_stamp(2).astype('f4'))

    im_w1, x_true, y_true, flux_true = crowdsource_base.sim_image(nx, ny, nstar, psf_w1, noise=noise_w1, nskyx=0, nskyy=0, stampsz=stampsz)

    stampszo2 = stampsz // 2
    im_w2_pad = np.pad(np.random.randn(nx, ny).astype('f4') * noise_w2, [stampszo2, stampszo2], constant_values=0., mode='constant')
    
    for i in range(nstar):
        stamp = psf_w2(x_true[i], y_true[i], stampsz=stampsz)
        xl = np.round(x_true[i]).astype('i4')
        yl = np.round(y_true[i]).astype('i4')
        im_w2_pad[xl:xl+stampsz, yl:yl+stampsz] += stamp * flux_true[i]
    im_w2 = im_w2_pad[stampszo2:-stampszo2, stampszo2:-stampszo2]

    wt_w1 = np.ones((nx, ny), dtype='f4') / noise_w1
    wt_w2 = np.ones((nx, ny), dtype='f4') / noise_w2

    return im_w1, im_w2, wt_w1, wt_w2, psf_w1, psf_w2, x_true, y_true, flux_true


def save_sim(im_w1, im_w2, x_true, y_true, flux_true, nstar, noise_w1, noise_w2, outdir):
    """
    Save W1 and W2 images as separate FITS files.
    EXT 0 : image
    EXT 1 (TRUTH) : table with x, y, flux
    """
    imgdir = os.path.join(outdir, 'images')
    os.makedirs(imgdir, exist_ok=True)

    truth = Table({'x':    x_true.astype('f4'),
                   'y':    y_true.astype('f4'),
                   'flux': flux_true.astype('f4')})

    for band, im, noise in [('w1', im_w1, noise_w1), ('w2', im_w2, noise_w2)]:
        outfn = os.path.join(imgdir, f'sim_{band}_nstar{nstar}_noise{noise}.fits')
        fits.HDUList([
            fits.PrimaryHDU(im),
            fits.BinTableHDU(truth, name='TRUTH')
            ]).writeto(outfn, overwrite=True)
        print(f"Saved {band} image to {outfn}")


def run_one(im_w1, im_w2, wt_w1, wt_w2, psf_w1, psf_w2, nstar, noise_w1, noise_w2, outdir):
    """
    Run single-band W1, single-band W2, and multiband fits.
    Save output catalogs to outdir/catalogs/.
    """
    catdir = os.path.join(outdir, 'catalogs')
    os.makedirs(catdir, exist_ok=True)

    fit_kwargs = dict(
                psfderiv=True, nskyx=0, nskyy=0, refit_psf=False, verbose=False, ntilex=1, ntiley=1,
                maxstars=50000, fewstars=50, threshold=5, psfvalsharpcutfac=0.5, psfsharpsat=0.8)

    res_sb_w1 = crowdsource_base.fit_im(im_w1, psf_w1, weights=wt_w1, **fit_kwargs)
    Table(res_sb_w1['stars']).write( os.path.join(catdir, f'sim_w1_nstar{nstar}_noise{noise_w1}.fits'), overwrite=True)
    print(f"  sb_w1: {len(res_sb_w1['stars'])} sources")

    res_sb_w2 = crowdsource_base.fit_im(im_w2, psf_w2, weights=wt_w2, **fit_kwargs)
    Table(res_sb_w2['stars']).write( os.path.join(catdir, f'sim_w2_nstar{nstar}_noise{noise_w2}.fits'), overwrite=True)
    print(f"  sb_w2: {len(res_sb_w2['stars'])} sources")

    res_mb = crowdsource_base.fit_im([im_w1, im_w2], [psf_w1, psf_w2], weights=[wt_w1, wt_w2], **fit_kwargs)
    Table(res_mb['stars']).write( os.path.join(catdir, f'sim_w1w2_nstar{nstar}_noise{noise_w1}_{noise_w2}.fits'), overwrite=True)
    print(f"  mb:    {len(res_mb['stars'])} sources")


def run_all(crowding_levels=(500, 1000, 3000, 5000, 8000),
            nx=512, ny=512, noise_w1=0.25, noise_w2=0.5,
            outdir='/global/cfs/cdirs/desi/users/shreeb/WISE/catalogs/simulated', seed=42):
    """
    Loop over crowding levels, simulate, fit, and save.
    """
    for subdir in ['images', 'catalogs']:
        os.makedirs(os.path.join(outdir, subdir), exist_ok=True)

    for nstar in crowding_levels:
        print(f"\n=== nstar={nstar} ===")

        im_w1, im_w2, wt_w1, wt_w2, psf_w1, psf_w2, x_true, y_true, flux_true = make_sim(nstar, nx=nx, ny=ny, noise_w1=noise_w1, noise_w2=noise_w2, seed=seed)
        save_sim(im_w1, im_w2, x_true, y_true, flux_true, nstar, noise_w1, noise_w2, outdir)
        run_one(im_w1, im_w2, wt_w1, wt_w2, psf_w1, psf_w2, nstar, noise_w1, noise_w2, outdir)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='/global/cfs/cdirs/desi/users/shreeb/WISE/catalogs/simulated')
    parser.add_argument('--seed',     type=int,   default=42)
    parser.add_argument('--noise_w1', type=float, default=0.2)
    parser.add_argument('--noise_w2', type=float, default=0.75)
    args = parser.parse_args()

    run_all(outdir=args.outdir, seed=args.seed, noise_w1=args.noise_w1, noise_w2=args.noise_w2)