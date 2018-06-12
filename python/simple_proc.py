#!/usr/bin/env python

import argparse, os, numpy, pdb
import psf as psfmod
from astropy.io import fits
import crowdsource
import mosaic


def process(imfn, ivarfn, flagfn, psf, nx=1, ny=1, satlimit=numpy.inf, **kw):
    im = fits.getdata(imfn)
    sqivar = numpy.sqrt(fits.getdata(ivarfn))
    flag = fits.getdata(flagfn)
    if numpy.isfinite(satlimit):
        m = imfn > satlimit
        sqivar[satlimit] = 0  # should also change the DQ image?
    if nx != 1 or ny != 1:
        res = mosaic.fit_sections(im, psf, nx, ny, weight=sqivar, dq=flag, **kw)
    else:
        res = crowdsource.fit_im(im, psf, weight=sqivar, dq=flag, **kw)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run crowdsource on image')
    # 3 arguments: image, weight, flags
    parser.add_argument('imagefn', type=str, nargs=1)
    parser.add_argument('ivarfn', type=str, nargs=1)
    parser.add_argument('flagfn', type=str, nargs=1)
    parser.add_argument('outfn', type=str, nargs=1)
    parser.add_argument('--psffn', '-p', type=str, help='file name of image of PSF, centroid at center')
    parser.add_argument('--refit-psf', '-r', default=False, action='store_true')
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    parser.add_argument('--satlimit', '-s', type=float, default=numpy.inf,
                        help='pixel brightness limit for saturation')
    args = parser.parse_args()
    imagefn = args.imagefn[0]
    ivarfn = args.ivarfn[0]
    flagfn = args.flagfn[0]
    if getattr(args, 'psffn', None):
        # stamp = numpy.clip(fits.getdata(args.psffn), 1e-10, numpy.inf)
        stamp = fits.getdata(args.psffn)
        stamp = stamp / numpy.sum(stamp)
        psf = psfmod.SimplePSF(stamp)
        psf.fitfun = psfmod.wise_psf_fit  # should generalize this somehow?
    else:
        print('using moffat')
        psf = psfmod.SimplePSF(psfmod.moffat_psf(2.5, beta=2.5)[0])
    res = process(imagefn, ivarfn, flagfn, psf, refit_psf=args.refit_psf, 
                  verbose=args.verbose, nx=2, ny=2, satlimit=args.satlimit)
    outfn = args.outfn[0]
    fits.writeto(outfn, res[0])
    fits.append(outfn, res[1])
    fits.append(outfn, res[2])

    
