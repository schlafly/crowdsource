#!/usr/bin/env python

import argparse, numpy
# didn't need os and pdb imports
import psf as psfmod
from astropy.io import fits
from simple_proc import process
import unwise_psf

def wise_filename(basedir, coadd_id, band, _type, uncompressed=False):
    # type should be one of:
    # 'img-u', 'img-m', 'invvar-u', 'invvar-m', 'std-u', 'std-m'
    # 'n-u', 'n-m', 'frames', 'msk'

    # -msk is special because the info for both W1/W2 is in same file

    fname = 'unwise-' + coadd_id
    if _type is not 'msk':
        fname += '-w' + str(band)

    fname += ('-' + _type + '.fits')

    fname = basedir + '/' + coadd_id[0:3] + '/' + coadd_id + '/' + fname

    if not uncompressed:
        if (_type != 'img-u') and (_type != 'img-m') and (_type != 'frames'):
            fname += '.gz'

    return fname

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run crowdsource on full-depth unWISE coadd image')
    # 3 arguments: image, weight, flags
    #parser.add_argument('imagefn', type=str, nargs=1)
    #parser.add_argument('ivarfn', type=str, nargs=1)
    #parser.add_argument('flagfn', type=str, nargs=1)
    parser.add_argument('coadd_id', type=str, nargs=1)
    parser.add_argument('band', type=int, nargs=1)
    parser.add_argument('outfn', type=str, nargs=1)

    parser.add_argument('basedir', type=str, nargs='?', default='/global/projecta/projectdirs/cosmo/work/wise/outputs/merge/neo4/fulldepth')
    parser.add_argument('--refit-psf', '-r', default=False, action='store_true')
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    parser.add_argument('--satlimit', '-s', type=float, default=numpy.inf,
                        help='pixel brightness limit for saturation')
    parser.add_argument('--uncompressed', '-u', default=False, action='store_true')

    args = parser.parse_args()

    coadd_id = args.coadd_id[0]
    band = args.band[0]
    basedir = args.basedir

    assert((band == 1) or (band == 2))
    assert(len(coadd_id) == 8)

    imagefn = wise_filename(basedir, coadd_id, band, 'img-m', uncompressed=args.uncompressed)
    ivarfn = wise_filename(basedir, coadd_id, band, 'invvar-m', uncompressed=args.uncompressed)
    # band isn't actually used, passing it in anyway...
    flagfn = wise_filename(basedir, coadd_id, band, 'msk', uncompressed=args.uncompressed)

    stamp = unwise_psf.get_unwise_psf(band, coadd_id)
    stamp[stamp < 0] = 0.
    stamp = stamp / numpy.sum(stamp)
    psf = psfmod.SimplePSF(stamp)
    from functools import partial
    psf.fitfun = partial(psfmod.wise_psf_fit, psfstamp=unwise_psf.get_unwise_psf(band, coadd_id))

    res = process(imagefn, ivarfn, flagfn, psf, refit_psf=args.refit_psf, 
                  verbose=args.verbose, nx=4, ny=4, satlimit=args.satlimit)
    outfn = args.outfn[0]
    fits.writeto(outfn, res[0])
    fits.append(outfn, res[1])
    fits.append(outfn, res[2])
