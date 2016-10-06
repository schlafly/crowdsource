#!/usr/bin/env python

import os
import pdb
import argparse
import numpy
import mosaic
import crowdsource
from astropy.io import fits
from astropy import wcs


def read_data(imfn, ivarfn, dqfn, extname):
    imdei = fits.getdata(imfn, extname=extname).copy()
    imdew = fits.getdata(ivarfn, extname=extname).copy()
    imded = fits.getdata(dqfn, extname=extname).copy()
    # actually need to do each CCD and merge
    # follow policy:
    # HDU 0: primary image header
    # HDU 1: CCD 1 header
    # HDU 2: CCD 1 catalog
    # HDU 3: CCD 2 header
    # HDU 4: CCD 2 catalog
    # ...
    mzerowt = (imded != 0) | (imdew < 0.) | ~numpy.isfinite(imdew)
    imdew[mzerowt] = 0.
    return imdei, imdew, imded


def process_image(imfn, ivarfn, dqfn, outfn=None, clobber=False,
                  verbose=False, nproc=numpy.inf):
    with fits.open(imfn) as hdulist:
        extnames = [hdu.name for hdu in hdulist]
    if 'PRIMARY' not in extnames:
        raise ValueError('No PRIMARY header in file')
    prihdr = fits.getheader(imfn, extname='PRIMARY')
    if outfn is None or len(outfn) == 0:
        outfn = os.path.splitext(os.path.basename(imfn))[0]+'.cat.fits'
    fits.writeto(outfn, None, prihdr, clobber=clobber)
    count = 0
    for name in extnames:
        if name is 'PRIMARY':
            continue
        if verbose:
            print('Fitting %s, extension %s.' % (imfn, name))
        im, wt, dq = read_data(imfn, ivarfn, dqfn, name)
        hdr = fits.getheader(imfn, extname=name)
        fwhm = hdr['FWHM']
        psf, dpsfdx, dpsfdy = crowdsource.moffat_psf(fwhm)
        psf = crowdsource.center_psf(psf)
        res = mosaic.fit_sections(im, psf, 4, 2, weight=wt,
                                  psfderiv=numpy.gradient(-psf),
                                  refit_psf=True)
        cat, modelim, skyim, psfs = res
        wcs0 = wcs.WCS(hdr)
        ra, dec = wcs0.all_pix2world(cat['x'], cat['y'], 0.)
        from matplotlib.mlab import rec_append_fields
        cat = rec_append_fields(cat, ['ra', 'dec'], [ra, dec])
        if verbose:
            print('Writing %s, found %d sources.' % (outfn, len(cat)))
        hdr['EXTNAME'] = hdr['EXTNAME']+'_HDR'
        fits.append(outfn, numpy.array(psfs), hdr)
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
    parser.add_argument('imfn', type=str, help='Image file name')
    parser.add_argument('ivarfn', type=str, help='Inverse variance file name')
    parser.add_argument('dqfn', type=str, help='Data quality file name')
    args = parser.parse_args()
    process_image(args.imfn, args.ivarfn, args.dqfn, outfn=args.outfn)
