#!/usr/bin/env python

import os
import sys
import pdb
import argparse
import numpy
import mosaic
import crowdsource
from astropy.io import fits
from astropy import wcs


def read(imfn, extname):
    ivarfn = imfn.replace('_ooi_', '_oow_')
    dqfn = imfn.replace('_ooi_', '_ood_')
    return read_data(imfn, ivarfn, dqfn, extname)


def read_data(imfn, ivarfn, dqfn, extname):
    import warnings
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        imdei = fits.getdata(imfn, extname=extname).copy()
        imdew = fits.getdata(ivarfn, extname=extname).copy()
        imded = fits.getdata(dqfn, extname=extname).copy()
    # suppress endless nonstandard keyword warnings on read
    for warning in wlist:
        if 'following header keyword' in str(warning.message):
            continue
        else:
            print(warning)
    mzerowt = (((imded != 0) & (imded != 7)) |
               (imdew < 0.) | ~numpy.isfinite(imdew))
    imdew[mzerowt] = 0.
    imdew[:] = numpy.sqrt(imdew)
    return imdei, imdew, imded


def process_image(imfn, ivarfn, dqfn, outfn=None, clobber=False,
                  outdir=None, verbose=False, nproc=numpy.inf):
    with fits.open(imfn) as hdulist:
        extnames = [hdu.name for hdu in hdulist]
    if 'PRIMARY' not in extnames:
        raise ValueError('No PRIMARY header in file')
    prihdr = fits.getheader(imfn, extname='PRIMARY')
    if outfn is None or len(outfn) == 0:
        outfn = os.path.splitext(os.path.basename(imfn))[0]+'.cat.fits'
    if outdir is not None:
        outfn = os.path.join(outdir, outfn)
    fits.writeto(outfn, None, prihdr, clobber=clobber)
    count = 0
    for name in extnames:
        if name is 'PRIMARY':
            continue
        if verbose:
            print('Fitting %s, extension %s.' % (imfn, name))
            sys.stdout.flush()
        im, wt, dq = read_data(imfn, ivarfn, dqfn, name)
        hdr = fits.getheader(imfn, extname=name)
        fwhm = hdr['FWHM']
        psf = crowdsource.moffat_psf(fwhm, stampsz=59, deriv=False)
        res = mosaic.fit_sections(im, psf, 4, 2, weight=wt, dq=dq,
                                  psfderiv=numpy.gradient(-psf),
                                  refit_psf=True, verbose=verbose)
        cat, modelim, skyim, psfs = res
        wcs0 = wcs.WCS(hdr)
        ra, dec = wcs0.all_pix2world(cat['y'], cat['x'], 0.)
        from matplotlib.mlab import rec_append_fields
        cat = rec_append_fields(cat, ['ra', 'dec'], [ra, dec])
        if verbose:
            print('Writing %s %s, found %d sources.' % (outfn, name, len(cat)))
            sys.stdout.flush()
        hdr['EXTNAME'] = hdr['EXTNAME']+'_HDR'
        # fits.append(outfn, numpy.array(psfs), hdr)
        fits.append(outfn, modelim, hdr)
        fits.append(outfn, cat)
        hdrcat = fits.getheader(outfn, -1)
        hdrcat['EXTNAME'] = hdr['EXTNAME'][:-4] + '_CAT'
        fits.update(outfn, cat, hdrcat, -1)
        count += 1
        if count > nproc:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit DECam frame')

    parser.add_argument('--outfn', '-o', type=str,
                        default=None, help='output file name')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--outdir', '-d', help='output directory',
                        type=str, default=None)
    parser.add_argument('imfn', type=str, help='Image file name')
    parser.add_argument('ivarfn', type=str, help='Inverse variance file name')
    parser.add_argument('dqfn', type=str, help='Data quality file name')
    args = parser.parse_args()
    process_image(args.imfn, args.ivarfn, args.dqfn, outfn=args.outfn,
                  verbose=args.verbose, outdir=args.outdir)
