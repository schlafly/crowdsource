#!/usr/bin/env python

import os
import pdb
import argparse
import numpy
import psf as psfmod
from astropy.io import fits
from simple_proc import process
import unwise_psf

extrabits = {'crowdsat': 2**24}


def wise_filename(basedir, coadd_id, band, _type, uncompressed=False,
                  drop_first_dir=False):
    # type should be one of:
    # 'img-u', 'img-m', 'invvar-u', 'invvar-m', 'std-u', 'std-m'
    # 'n-u', 'n-m', 'frames', 'msk'

    # -msk is special because the info for both W1/W2 is in same file

    fname = 'unwise-' + coadd_id
    if _type is not 'msk':
        fname += '-w' + str(band)

    fname += ('-' + _type + '.fits')

    path = [basedir, coadd_id[0:3], coadd_id, fname]
    if drop_first_dir:
        del path[1]
    fname = os.path.join(*path)

    if not uncompressed:
        if (_type != 'img-u') and (_type != 'img-m') and (_type != 'frames'):
            fname += '.gz'

    return fname


def read_blist(brightstars, raim, decim, hdr, maxsep):
    from astropy.coordinates.angle_utilities import angular_separation
    sep = angular_separation(numpy.radians(brightstars['ra']),
                             numpy.radians(brightstars['dec']),
                             numpy.radians(raim),
                             numpy.radians(decim))
    sep = numpy.degrees(sep)
    m = (sep < 3) & (brightstars['k_m'] < 5)
    brightstars = brightstars[m]
    from astropy import wcs
    wcs0 = wcs.WCS(hdr)
    yy, xx = wcs0.all_world2pix(brightstars['ra'], brightstars['dec'], 0)
    m = (xx > 0) & (xx < hdr['NAXIS1']) & (yy > 0) & (yy < hdr['NAXIS2'])
    xx, yy = xx[m], yy[m]
    mag = brightstars['k_m'][m]
    if not numpy.any(m):
        return None
    else:
        return [xx, yy, mag]


def massage_isig_and_dim(isig, im, flag, band, nm, fac=None):
    """Construct a WISE inverse sigma image and add saturation to flag.

    unWISE provides nice inverse variance maps.  These however have no
    contribution from Poisson noise from sources, and so underestimate
    the uncertainties dramatically in bright regions.  This can pull the
    whole fit awry in bright areas, since the sky model means that every
    pixel feels every other pixel.

    It's not clear what the best solution is.  We make a goofy inverse
    sigma image from the original image and the inverse variance image.  It
    is intended to be sqrt(ivar) for the low count regime and grow like
    sqrt(1/im) for the high count regime.  The constant of proportionality
    should in principle be worked out; here I set it to 0.15, which worked
    once, and it doesn't seem like this should depend much on which
    WISE exposure the image came from?  It's ultimately something like the gain
    or zero point...
    """

    if fac is None:
        bandfacs = {1: 0.15, 2: 0.3}
        fac = bandfacs[band]

    satbit = 16 if band == 1 else 32
    satlimit = 85000 if band == 1 else 130000
    msat = ((flag & satbit) != 0) | (im > satlimit) | (nm == 0)
    from scipy.ndimage import morphology
    # dilate = morphology.iterate_structure(
    #     morphology.generate_binary_structure(2, 1), 3)
    xx, yy = numpy.mgrid[-3:3+1, -3:3+1]
    dilate = xx**2+yy**2 <= 3**2
    msat = morphology.binary_dilation(msat, dilate)
    isig[msat] = 0
    flag = flag.astype('i8')
    flag[msat] |= extrabits['crowdsat']
    sigma = numpy.sqrt(1./(isig + (isig == 0))**2 +
                       fac**2*numpy.clip(im, 0, numpy.inf))
    sigma[msat] = numpy.inf
    return (1./sigma).astype('f4'), flag


def wise_psf(band, coadd_id):
    # psf noise: ~roughly 0.1 count in outskirts of W1 and W2
    if band >= 3:
        raise ValueError('Need to stare at W3+ PSF more!')
    psfnoise = 0.1
    stamp = unwise_psf.get_unwise_psf(band, coadd_id)
    edges = numpy.concatenate([stamp[0, 1:-1], stamp[-1, 1:-1],
                               stamp[1:-1, 0], stamp[1:-1, -1]])
    if band == 1:
        medval = numpy.median(edges[edges != 0]) / 2
    elif band == 2:
        medval = numpy.median(edges[edges != 0]) / 4
    else:
        medval = 0.
    stamp[stamp == 0] = medval
    stamp -= medval
    # stamp = numpy.clip(stamp, -numpy.median(edges[edges != 0]), numpy.inf)
    # print(numpy.median(edges[edges != 0]))
    # print(numpy.min(stamp))
    from scipy import signal
    stamp[stamp < 0] = 0.
    stamp = signal.wiener(stamp,  11, psfnoise)
    # taper linearly over outer 60 pixels?
    stampszo2 = stamp.shape[0] // 2
    xx, yy = numpy.mgrid[-stampszo2:stampszo2+1, -stampszo2:stampszo2+1]
    edgedist = numpy.clip(stampszo2-numpy.abs(xx), 0,
                          stampszo2-numpy.abs(yy))
    stamp = stamp * numpy.clip(edgedist / 60., stamp < 10, 1)
    stamp = stamp / numpy.sum(stamp)
    psf = psfmod.SimplePSF(stamp)
    from functools import partial
    psf.fitfun = partial(psfmod.wise_psf_fit, psfstamp=stamp)
    return psf


def read_wise(coadd_id, band, basedir, uncompressed=False,
              drop_first_dir=False):
    assert((band == 1) or (band == 2))
    assert(len(coadd_id) == 8)

    imagefn = wise_filename(basedir, coadd_id, band, 'img-m',
                            uncompressed=uncompressed,
                            drop_first_dir=drop_first_dir)
    ivarfn = wise_filename(basedir, coadd_id, band, 'invvar-m',
                           uncompressed=uncompressed,
                           drop_first_dir=drop_first_dir)
    # band isn't actually used, passing it in anyway...
    flagfn = wise_filename(basedir, coadd_id, band, 'msk',
                           uncompressed=uncompressed,
                           drop_first_dir=drop_first_dir)
    nmfn = wise_filename(basedir, coadd_id, band, 'n-m',
                         uncompressed=uncompressed,
                         drop_first_dir=drop_first_dir)

    im = fits.getdata(imagefn)
    sqivar = numpy.sqrt(fits.getdata(ivarfn))
    flag = fits.getdata(flagfn)
    nm = fits.getdata(nmfn)
    sqivar, flag = massage_isig_and_dim(sqivar, im, flag, band, nm)
    return im, sqivar, flag


def brightlist(brightstars, coadd_id, band, basedir, uncompressed=False,
               drop_first_dir=False):
    imagefn = wise_filename(basedir, coadd_id, band, 'img-m',
                            uncompressed=uncompressed,
                            drop_first_dir=drop_first_dir)
    hdr = fits.getheader(imagefn)
    blist = read_blist(brightstars, hdr['CRVAL1'], hdr['CRVAL2'], hdr, 3)
    return blist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run crowdsource on full-depth unWISE coadd image')
    parser.add_argument('coadd_id', type=str, nargs=1)
    parser.add_argument('band', type=int, nargs=1)
    parser.add_argument('outfn', type=str, nargs=1)

    parser.add_argument('basedir', type=str, nargs='?', default='/global/projecta/projectdirs/cosmo/work/wise/outputs/merge/neo4/fulldepth')
    parser.add_argument('--refit-psf', '-r', default=False, action='store_true')
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    parser.add_argument('--satlimit', '-s', type=float, default=numpy.inf,
                        help='pixel brightness limit for saturation')
    parser.add_argument('--uncompressed', '-u', default=False, action='store_true')
    parser.add_argument('--brightcat', '-b',
                        default=os.environ.get('TMASS_BRIGHT'), type=str)

    args = parser.parse_args()

    coadd_id = args.coadd_id[0]
    band = args.band[0]
    basedir = args.basedir

    im, sqivar, flag = read_wise(coadd_id, band, basedir,
                                 uncompressed=args.uncompressed)

    psf = wise_psf(band, coadd_id)

    if len(args.brightcat) > 0:
        brightstars = fits.getdata(args.brightcat)
        blist = brightlist(brightstars, coadd_id, band, basedir,
                           uncompressed=args.uncompressed)
    else:
        print('No bright star catalog, not marking bright stars.')

    res = process(im, sqivar, flag, psf, refit_psf=args.refit_psf,
                  verbose=args.verbose, nx=4, ny=4)
    outfn = args.outfn[0]
    fits.writeto(outfn, res[0])
    fits.append(outfn, res[1])
    fits.append(outfn, res[2])
