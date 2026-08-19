#!/usr/bin/env python

import os
import sys
import time
import pdb
import argparse
import numpy as np
import crowdsource.psf as psfmod
from astropy.io import fits
from crowdsource import crowdsource_base
from unwise_psf import unwise_psf
# implicit dependency for WISE runs only
# https://github.com/legacysurvey/unwise_psf
import crowdsource.unwise_primary as unwise_primary
from astropy import wcs
from collections import OrderedDict
from pkg_resources import resource_filename


extrabits = {'crowdsat': 2**25,
             'nebulosity': 2**26,
             'w1brightoffedge': 2**7,
             'w2brightoffedge': 2**8,
             'hyperleda': 2**9}

nodeblend_bits = extrabits['hyperleda']
sharp_bits = (extrabits['w1brightoffedge'] | extrabits['w2brightoffedge'])

def wise_filename(basedir, coadd_id, band, _type, uncompressed=False,
                  drop_first_dir=False, epoch=-1):
    # type should be one of:
    # 'img-u', 'img-m', 'invvar-u', 'invvar-m', 'std-u', 'std-m'
    # 'n-u', 'n-m', 'frames', 'msk'

    fname = 'unwise-' + coadd_id
    if _type != 'msk':
        fname += '-w' + str(band)

    fname += ('-' + _type + '.fits')

    path = [basedir, coadd_id[0:3], coadd_id, fname]
    if drop_first_dir:
        del path[1]
    if epoch >= 0:
        epochstr = 'e%03d' % epoch if _type != 'msk' else 'fulldepth'
        path = path[0:1] + [epochstr] + path[1:]
    fname = os.path.join(*path)

    if not uncompressed or _type == 'msk':
        if (_type not in ['img-u', 'img-m', 'frames']):
            # Prefer .gz if it exists
            gzname = fname + '.gz'
            if os.path.exists(gzname):
                return gzname
            # Otherwise fall back to plain .fits
    return fname



def read_blist(brightstars, raim, decim, hdr, maxsep):
    from astropy.coordinates.angle_utilities import angular_separation
    sep = angular_separation(np.radians(brightstars['ra']),
                             np.radians(brightstars['dec']),
                             np.radians(raim),
                             np.radians(decim))
    sep = np.degrees(sep)
    m = (sep < 3) & (brightstars['k_m'] < 5)
    brightstars = brightstars[m]
    wcs0 = wcs.WCS(hdr)
    yy, xx = wcs0.all_world2pix(brightstars['ra'], brightstars['dec'], 0)
    m = (xx > 0) & (xx < hdr['NAXIS1']) & (yy > 0) & (yy < hdr['NAXIS2'])
    xx, yy = xx[m], yy[m]
    mag = brightstars['k_m'][m]
    if not np.any(m):
        return None
    else:
        return [xx, yy, mag]


def massage_isig_and_dim(isig, im, flag, band, nm, nu, fac=None):
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
        bandfloors = {1: 0.5, 2: 2}
        fac = bandfacs[band]
        floor = bandfloors[band]

    satbit = 16 if band == 1 else 32
    satlimit = 85000  # if band == 1 else 130000
    msat = ((flag & satbit) != 0) | (im > satlimit) | ((nm == 0) & (nu > 1))
    from scipy.ndimage import morphology
    # dilate = morphology.iterate_structure(
    #     morphology.generate_binary_structure(2, 1), 3)
    xx, yy = np.mgrid[-3:3+1, -3:3+1]
    dilate = xx**2+yy**2 <= 3**2
    msat = morphology.binary_dilation(msat, dilate)
    isig[msat] = 0
    flag = flag.astype('i8')
    # zero out these bits; we claim them for our own purposes.
    massagebits = (extrabits['crowdsat'] | crowdsource_base.nodeblend_maskbit |
                   crowdsource_base.sharp_maskbit | extrabits['nebulosity'])
    flag &= ~massagebits
    flag[msat] |= extrabits['crowdsat']
    flag[(flag & nodeblend_bits) != 0] |= crowdsource_base.nodeblend_maskbit
    flag[(flag & sharp_bits) != 0] |= crowdsource_base.sharp_maskbit

    sigma = np.sqrt(1./(isig + (isig == 0))**2 + floor**2 +
                       fac**2*np.clip(im, 0, np.inf))
    sigma[msat] = np.inf
    sigma[isig == 0] = np.inf
    return (1./sigma).astype('f4'), flag


def wise_psf_stamp(band, nosmooth=False):
    # psf noise: ~roughly 0.1 count in outskirts of W1 and W2
    if band >= 3:
        raise ValueError('Need to stare at W3+ PSF more!')
    psfnoise = 0.1
    stampfn = resource_filename('unwise_psf',
                                'data/psf_model_w'+str(band)+'.fits')
    stamp = fits.getdata(stampfn)
    edges = np.concatenate([stamp[0, 1:-1], stamp[-1, 1:-1],
                               stamp[1:-1, 0], stamp[1:-1, -1]])
    medval = np.median(edges[edges != 0]) / 2
    stamp[stamp == 0] = medval
    stamp -= medval
    from scipy import signal
    stamp[stamp < 0] = 0.
    # suppress spurious warnings in signal.wiener
    olderr = np.seterr(invalid='ignore', divide='ignore')
    # update to scipy.signal means that Wiener filter uses an FFT
    # to perform the various convolutions, which causes bad errors
    # here unless we cast to f8.  It's not that hard to do something
    # a bit better than scipy.signal.wiener---morally we really want to do
    # something like smooth in log space on radial lines---but I don't
    # want to go further down that rabbit hole today.
    stamp = signal.wiener(stamp.astype('f8'),  11, psfnoise)
    stamp = stamp.astype('f4')
    np.seterr(**olderr)
    # taper linearly over outer 60 pixels?
    stampszo2 = stamp.shape[0] // 2
    xx, yy = np.mgrid[-stampszo2:stampszo2+1, -stampszo2:stampszo2+1]
    edgedist = np.clip(stampszo2-np.abs(xx), 0,
                          stampszo2-np.abs(yy))
    stamp = stamp * np.clip(edgedist / 60., stamp < 10, 1)
    import psf
    stamp = psf.center_psf(stamp, censize=19)
    stamp = stamp / np.sum(stamp)
    return stamp


def wise_psf(band, coadd_id):
    stamp = wise_psf_stamp(band)
    stamp = unwise_psf.rotate_using_rd(stamp, coadd_id)
    psf = psfmod.SimplePSF(stamp)
    from functools import partial
    psf.fitfun = partial(psfmod.wise_psf_fit, psfstamp=stamp)
    return psf


def wise_psf_grid(band, coadd_id, basedir, uncompressed=False,
                  drop_first_dir=False, ngrid=None, epoch=-1):
    imagefn = wise_filename(basedir, coadd_id, band, 'img-m',
                            uncompressed=uncompressed,
                            drop_first_dir=drop_first_dir, epoch=epoch)
    hdr = fits.getheader(imagefn)
    if ngrid is None:
        rr, dd = hdr['CRVAL1'], hdr['CRVAL2']
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        coord = SkyCoord(ra=rr*u.deg, dec=dd*u.deg, frame='icrs')
        coord = coord.geocentrictrueecliptic
        lam, bet = coord.lon.deg, coord.lat.deg
        dlam = 1.4/(np.abs(np.cos(np.radians(bet)))+1e-6)
        ngrid = np.floor(np.clip(dlam / 1, 4, 16)).astype('i4')
    x = np.linspace(0, 2047, ngrid)
    y = np.linspace(0, 2047, ngrid)
    wcs0 = wcs.WCS(hdr)
    stamp = wise_psf_stamp(band).astype('f4')
    stamps = np.zeros((len(x), len(y))+stamp.shape, dtype=stamp.dtype)
    unwise_psf.rotate_using_convolution.cache = None  # clear cache
    for i in range(len(x)):
        for j in range(len(y)):
            rr, dd = wcs0.all_pix2world(y[j], x[i], 0)
            stamps[i, j, ...] = unwise_psf.rotate_using_rd(
                stamp, coadd_id, ra=rr, dec=dd, cache=True)
    psf = psfmod.GridInterpPSF(stamps, x, y)
    from functools import partial
    psf.fitfun = partial(psfmod.wise_psf_fit, psfstamp=(stamps, x, y),
                         grid=True)
    return psf


def read_wise(coadd_id, band, basedir, uncompressed=False,
              drop_first_dir=False, epoch=-1):
    assert((band == 1) or (band == 2))
    assert(len(coadd_id) == 8)

    imagefn = wise_filename(basedir, coadd_id, band, 'img-m',
                            uncompressed=uncompressed,
                            drop_first_dir=drop_first_dir, epoch=epoch)
    ivarfn = wise_filename(basedir, coadd_id, band, 'invvar-m',
                           uncompressed=uncompressed,
                           drop_first_dir=drop_first_dir, epoch=epoch)
    flagfn = wise_filename(basedir, coadd_id, band, 'msk',
                           uncompressed=uncompressed,
                           drop_first_dir=drop_first_dir, epoch=epoch)
    nmfn = wise_filename(basedir, coadd_id, band, 'n-m',
                         uncompressed=uncompressed,
                         drop_first_dir=drop_first_dir, epoch=epoch)
    nufn = wise_filename(basedir, coadd_id, band, 'n-u',
                         uncompressed=uncompressed,
                         drop_first_dir=drop_first_dir, epoch=epoch)

    im, hdr = fits.getdata(imagefn, header=True)
    sqivar = np.sqrt(fits.getdata(ivarfn))
    flag = fits.getdata(flagfn)
    nm = fits.getdata(nmfn)
    nu = fits.getdata(nufn)
    sqivar, flag = massage_isig_and_dim(sqivar, im, flag, band, nm, nu)
    return im, sqivar, flag, hdr


def ivarmap(isig, psfstamp):
    from scipy.signal import fftconvolve
    ivarim = fftconvolve(isig**2., psfstamp[::-1, ::-1]**2., mode='same')
    return ivarim


def brightlist(brightstars, coadd_id, band, basedir, uncompressed=False,
               drop_first_dir=False, epoch=-1):
    imagefn = wise_filename(basedir, coadd_id, band, 'img-m',
                            uncompressed=uncompressed,
                            drop_first_dir=drop_first_dir, epoch=epoch)
    hdr = fits.getheader(imagefn)
    blist = read_blist(brightstars, hdr['CRVAL1'], hdr['CRVAL2'], hdr, 3)
    return blist


def collapse_unwise_bitmask(bitmask, band):
    # 2^0 = bright star core and wings
    # 2^1 = PSF-based diffraction spike
    # 2^2 = optical ghost
    # 2^3 = first latent
    # 2^4 = second latent
    # 2^5 = AllWISE-like circular halo
    # 2^6 = bright star saturation
    # 2^7 = geometric diffraction spike

    assert((band == 1) or (band == 2))

    bits_w1 = OrderedDict([('core_wings', 2**0 + 2**1),
                           ('psf_spike', 2**27),
                           ('ghost', 2**25 + 2**26),
                           ('first_latent', 2**13 + 2**14),
                           ('second_latent', 2**17 + 2**18),
                           ('circular_halo', 2**23),
                           ('saturation', 2**4),
                           ('geom_spike', 2**29)])

    bits_w2 = OrderedDict([('core_wings', 2**2 + 2**3),
                           ('psf_spike', 2**28),
                           ('ghost', 2**11 + 2**12),
                           ('first_latent', 2**15 + 2**16),
                           ('second_latent', 2**19 + 2**20),
                           ('circular_halo', 2**24),
                           ('saturation', 2**5),
                           ('geom_spike', 2**30)])

    bits = (bits_w1 if (band == 1) else bits_w2)

    # hack to handle both scalar and array inputs
    result = 0*bitmask

    for i, feat in enumerate(bits.keys()):
        result += (2**i)*(np.bitwise_and(bitmask, bits[feat]) != 0)

    # int8 would be fine here, but astropy.io.fits seems to read this
    # as a boolean... so we waste the extra 8 bits.
    return result.astype('i2')


def collapse_extraflags(bitmask, band):
    bits_w1 = OrderedDict([('bright_off_edge', 2**7),
                           ('resolved_galaxy', 2**9),
                           ('big_object', 2**10),
                           ('possible_bright_star_centroid', 2**21),
                           ('crowdsat', extrabits['crowdsat']),
                           ('nebulosity', extrabits['nebulosity']),
                           ('nodeblend', crowdsource_base.nodeblend_maskbit),
                           ('sharp', crowdsource_base.sharp_maskbit)])

    bits_w2 = OrderedDict([('bright_off_edge', 2**8),
                           ('resolved_galaxy', 2**9),
                           ('big_object', 2**10),
                           ('possible_bright_star_centroid', 2**22),
                           ('crowdsat', extrabits['crowdsat']),
                           ('nebulosity', extrabits['nebulosity']),
                           ('nodeblend', crowdsource_base.nodeblend_maskbit),
                           ('sharp', crowdsource_base.sharp_maskbit)])

    bits = (bits_w1 if (band == 1) else bits_w2)

    # hack to handle both scalar and array inputs
    result = 0*bitmask

    for i, feat in enumerate(bits.keys()):
        result += (2**i)*(np.bitwise_and(bitmask, bits[feat]) != 0)

    # could fit in a byte, but astropy.io.fits reads these as booleans,
    # so we waste the byte...
    return result.astype('i2')

def merge_per_band_fields(cat, bands):
    """
    Merge flux_bi -> flux[:,i] and remove per-band fields.
    bands : list of actual WISE bands, e.g. [1,2]

    Returns: New catalog array with merged fields.
    """
    import re

    # ---- Convert WISE bands to 0-based internal indices ----
    # Example: bands=[1,2] --> idx_bands=[0,1]
    idx_bands = list(range(len(bands)))
    pat = re.compile(r"^(.*)_b(\d+)$")

    #detect per-band fields
    mapping = {}; per_band_fields = set()
    for name in cat.dtype.names:
        m = pat.match(name)
        if m:
            base = m.group(1); band_idx = int(m.group(2))
            per_band_fields.add(name)
            mapping.setdefault(base, {})[band_idx] = name

    #build new dtype (keep non-per-band)
    new_dtype = []
    for name in cat.dtype.names:
        if name not in per_band_fields: new_dtype.append((name, cat[name].dtype))

    #add merged fields
    B = len(idx_bands)
    for base, band_dict in mapping.items():
        # Only merge if all internal indices appear
        if all(i in band_dict for i in idx_bands):
            dt = cat[band_dict[idx_bands[0]]].dtype
            new_dtype.append((base, dt, (B,)))

    out = np.zeros(len(cat), dtype=new_dtype)

    # Copy non-per-band fields
    for name in out.dtype.names:
        if name in cat.dtype.names: out[name] = cat[name]

    # fill merged arrays
    for base, band_dict in mapping.items():
        if not all(i in band_dict for i in idx_bands):
            continue
        merged = np.vstack([cat[band_dict[i]] for i in idx_bands]).T
        out[base] = merged

    return out


if __name__ == "__main__":
    try:
        print('Running on host: ' + str(os.environ.get('HOSTNAME')))
    except Exception:
        print("Couldn't retrieve hostname!")

    parser = argparse.ArgumentParser(description='Run crowdsource on WISE coadd image(s)')
    parser.add_argument('coadd_id', type=str, nargs=1)
    parser.add_argument('bands', type=int, nargs='+', help='Bands to process, e.g. 1 2 or 1 2 3 4 ...')
    parser.add_argument('--bandweights', type=float, nargs='+', default=None)
    parser.add_argument('outdir', type=str, nargs=1, help='Base output directory (will create cat/, mod/, iminfo/, log/ subdirs)')
    parser.add_argument('--basedir', type=str, nargs='?', default='/global/cfs/cdirs/cosmo/work/wise/outputs/merge/neo8/fulldepth', help='Input directory to images')
    parser.add_argument('--outfn', '-o', default=None, type=str, help='Catalog file path (default: auto-generated in outdir/cat/)')
    parser.add_argument('--modelfn', '-m', default=None, type=str, help='Model file path (default: auto-generated in outdir/mod/)')
    parser.add_argument('--infoimfn', '-i', default=None, type=str, help='Info image file path (default: auto-generated in outdir/iminfo/)')
    parser.add_argument('--refit-psf', '-r', default=False, action='store_true')
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    parser.add_argument('--uncompressed', '-u', default=False, action='store_true')
    parser.add_argument('--brightcat', '-b', default=os.environ.get('TMASS_BRIGHT', ''), type=str)
    parser.add_argument('--masknebulosity', '-n', action='store_true')
    parser.add_argument('--startsky', type=str, default='')
    parser.add_argument('--startpsf', type=str, default='')
    parser.add_argument('--noskyfit', default=False, action='store_true')
    parser.add_argument('--threshold', default=5, type=float)
    parser.add_argument('--epoch', type=int, default=-1)
    parser.add_argument('--release', type=str, default='')
    parser.add_argument('--save_snapshot', default=False, action='store_true')
    

    args = parser.parse_args()

    coadd_id = args.coadd_id[0]
    basedir = args.basedir
    outdir = args.outdir[0]
    
    for subdir in ["", "cat", "mod", "iminfo", "log"]: os.makedirs(os.path.join(outdir, subdir), exist_ok=True)
    bands_str = ''.join(str(b) for b in args.bands)  # e.g. '2' or '12' or '1234'
    if args.bandweights == None: bw_str = ''
    else: bw_str = '.'+''.join(str(b) for b in args.bandweights)
    
    if args.outfn is None: outfn = f'{coadd_id}.{bands_str}{bw_str}.cat.fits'
    else: outfn = args.outfn
    if args.modelfn is None: modelfn = f'{coadd_id}.{bands_str}{bw_str}.mod.fits'
    else: modelfn = args.modelfn
    if args.infoimfn is None: infoimfn = f'{coadd_id}.{bands_str}{bw_str}.info.fits'
    else: infoimfn = args.infoimfn
        
    outfn = os.path.join(outdir, 'cat', outfn)
    modelfn = os.path.join(outdir,'mod', modelfn)
    infoimfn = os.path.join(outdir, 'iminfo', infoimfn)

    ims, sqivars, psfs, flags, hdrs, blists = [], [], [], [], [], []
    for band in args.bands:
        im, sqivar, flag, hdr = read_wise(coadd_id, band, basedir, uncompressed=args.uncompressed, epoch=args.epoch)
    
        if len(args.startsky) > 0: startsky = fits.getdata(args.startsky, 'SKY')
        else: startsky = np.nan
    
        flag_orig = fits.getdata(wise_filename(basedir, coadd_id, band, 'msk', uncompressed=args.uncompressed, epoch=args.epoch))
    
        if args.masknebulosity:
            import nebulosity_mask
            nebfn = os.path.join(os.environ['WISE_DIR'], 'dat', 'nebnet', 'weights1', '1st_try')
            nebmod = nebulosity_mask.load_model(nebfn)
            nebmask = nebulosity_mask.gen_mask_wise(nebmod, im) == 2
            if np.any(nebmask):
                # mark those pixels as bad in the flag map
                flag |= nebmask * extrabits['nebulosity']
                flag |= nebmask * crowdsource_base.sharp_maskbit
                print('Masking nebulosity, %5.2f' % (np.sum(nebmask)/1./np.sum(np.isfinite(nebmask))))
    
        psf = wise_psf_grid(band, coadd_id, basedir, epoch=args.epoch)

        if len(args.startpsf) > 0:
            startpsf = fits.getdata(args.startpsf, 'PSF').astype('f4')
            # there can be some endianness issues; astype('f4') converts to native
            modpsf = psf(1024, 1024, stampsz=psf.stamp.shape[-1])
            resid = startpsf - modpsf
            # need not sum to zero.
            newstamps = psf.stamp / psf.normstamp[:, :, None, None]
            newstamps += resid
            psf = psfmod.GridInterpPSF(newstamps, psf.x, psf.y)
            from functools import partial
            psf.fitfun = partial(psfmod.wise_psf_fit,
                                 psfstamp=(newstamps, psf.x, psf.y), grid=True)
    
        if len(args.brightcat) > 0:
            brightstars = fits.getdata(args.brightcat)
            blist = brightlist(brightstars, coadd_id, band, basedir, uncompressed=args.uncompressed, epoch=args.epoch)
        else:
            print(f'No bright star catalog in band{band}, not marking bright stars.')
            blist = None
    
        ims.append(im); sqivars.append(sqivar); psfs.append(psf); flags.append(flag); hdrs.append(hdr); blists.append(blist)

    # print("Input image shapes:", [im.shape for im in ims])
    # print("Weight shapes:", [sq.shape for sq in sqivars])
    # print("Flag shapes:", [fl.shape for fl in flags])
    # print("PSF shapes:", [psf.stamp.shape for psf in np.array(psfs)])

    if args.verbose:
        logfn = os.path.join(outdir, 'log', f'{coadd_id}.{bands_str}.log')
        sys.stdout = open(logfn, "w")
        sys.stderr = sys.stdout
        t0 = time.time()
        print('Starting %s, bands %s, at %s' % (coadd_id, bands_str, time.ctime()))
        print('Band weights:', args.bandweights)
        sys.stdout.flush()

    flag_combined = np.bitwise_or.reduce(np.stack(flags, axis=0))
    # if len(args.forcecat) == 0:
    
    res = crowdsource_base.fit_im(
        ims, psfs, weights=sqivars, dq=flag_combined,
        band_weights=args.bandweights,
        refit_psf=args.refit_psf, verbose=args.verbose,
        ntilex=4, ntiley=4, maxstars=30000*16, fewstars=50*16,
        threshold=args.threshold,
        psfvalsharpcutfac=0.5, psfsharpsat=0.8
    )

    cat = res["stars"]
    model = res["model"]
    sky = res["sky"]
    psf = res["psfs"]
    snapshots = res["iter_history"]

    if args.save_snapshot:
        snapfn = os.path.join(outdir, "snapshot", f"{coadd_id}.{bands_str}{bw_str}.snapshots.npz")
        os.makedirs(os.path.join(outdir, "snapshot"), exist_ok=True)    
        snapshots_obj = np.array(snapshots, dtype=object)  
        np.savez_compressed(snapfn, snapshots=snapshots_obj)
        print("Wrote snapshots to:", snapfn)

    if args.verbose:
        print(f'Finishing {coadd_id}, bands {args.bands}; elapsed {time.time() - t0:.1f}sec.')

    x = cat['x']; y = cat['y']

    if len(args.bands) == 1:
        hdr = hdrs[0] 
        hdr['BAND'] = args.bands[0]
        band_val = args.bands[0]
        id_prefix = f"{coadd_id}w{band_val}"
    
    else:
        hdr = hdrs[0].copy()
        hdr['BANDS'] = ','.join(str(b) for b in args.bands)
        # id_prefix = f"{coadd_id}w{''.join(str(b) for b in args.bands)}"   #Band format : w123...
        id_prefix = f"{coadd_id}{''.join(f'w{b}' for b in args.bands)}"    #Band format : w1w2w3...

    
    # ---- Now compute RA/Dec using hdr ----
    wcs0 = wcs.WCS(hdr)
    ra, dec = wcs0.all_pix2world(y, x, 0)
    coadd_ids = np.full(len(ra), coadd_id, dtype='a8')
    
    # ---- Build bands_col ----
    if len(args.bands) == 1: bands_col = np.full(len(ra), band_val, dtype='i4')
    else: bands_col = np.full(len(ra), -1, dtype='i4')  # -1 = joint fit

    if len(args.release) == 0: ids = [f"{id_prefix}o{num:07d}" for num in range(len(ra))]
    else: ids = [f"{id_prefix}o{num:07d}r{args.release}" for num in range(len(ra))]

    nms_all, flags_unwise_all, flags_info_all = [], [], []
    for i, band in enumerate(args.bands):
        # nm per band
        nmfn = wise_filename(basedir, coadd_id, band, 'n-m', uncompressed=args.uncompressed, epoch=args.epoch)
        nmim = fits.getdata(nmfn)
        nms_all.append(crowdsource_base.extract_im(cat['x'], cat['y'], nmim).astype('i2'))
    
        flag_orig = fits.getdata(wise_filename(basedir, coadd_id, band, 'msk',
                                               uncompressed=args.uncompressed,
                                               epoch=args.epoch))
    
        # unwise flags per band
        fu = crowdsource_base.extract_im(cat['x'], cat['y'],collapse_unwise_bitmask(flag_orig, band)).astype('i2')
        flags_unwise_all.append(fu)
    
        # info flags per band
        fi = crowdsource_base.extract_im(cat['x'], cat['y'],collapse_extraflags(flags[i], band)).astype('i2')
        flags_info_all.append(fi)
    
    # Stack into (nbands, nsource) arrays
    nms_all          = np.asarray(nms_all, dtype='i2')
    flags_unwise_all = np.asarray(flags_unwise_all, dtype='i2')
    flags_info_all   = np.asarray(flags_info_all, dtype='i2')
    
    # cast to i2; astropy.io.fits seems to fail for bools?
    primary = unwise_primary.is_primary(coadd_id, ra, dec).astype('i2')
    
    import numpy.lib.recfunctions as rfn
    cat = rfn.drop_fields(cat, ['flags'])
    # Append the common columns
    cat = rfn.append_fields(
        cat,
        ['ra', 'dec', 'coadd_id', 'band', 'unwise_detid', 'primary'],
        [ra, dec, coadd_ids, bands_col, ids, primary],
        usemask=False
    )
    # append per-band info as separate columns
    from numpy.lib import recfunctions as rfn
    seen = set()
    for ib, band in enumerate(args.bands):
        if band in seen:      # skip duplicate bands (e.g., W1 in W1+W1)
            continue
        seen.add(band)
    
        cat = rfn.append_fields(cat, [f'nm_b{ib}'],           [nms_all[ib, :]],         usemask=False)
        cat = rfn.append_fields(cat, [f'flags_unwise_b{ib}'], [flags_unwise_all[ib, :]], usemask=False)
        cat = rfn.append_fields(cat, [f'flags_info_b{ib}'],   [flags_info_all[ib, :]],  usemask=False)

    print(cat.dtype.names)
    #Merge the individual bands to have the output with the structure as objcat
    cat_out = merge_per_band_fields(cat, args.bands)
    print(cat_out.dtype.names)
    

    hdr['EXTNAME'] = 'PRIMARY'
    fits.writeto(outfn, None, hdr, overwrite=True)
    fits.append(outfn, cat_out)


    if modelfn is not None:
        fits.writeto(modelfn, None, hdr, overwrite=True)
        with fits.open(modelfn, mode='append', memmap=False) as hdulist:
            if isinstance(model, (list, tuple)):
                model_arr = np.stack(model, axis=0)
                sky_arr   = np.stack(sky, axis=0)
                tile_shape = model[0].shape
            else:
                model_arr = model
                sky_arr   = sky
                tile_shape = model.shape
    
            compkw = dict(compression_type='GZIP_1',
                          quantize_method=2, quantize_level=-0.5,
                          tile_shape=tile_shape)
    
            h = hdr.copy(); h['EXTNAME'] = 'model'
            hdulist.append(fits.CompImageHDU(model_arr.astype('f4'), header=h, **compkw))
    
            h = hdr.copy(); h['EXTNAME'] = 'sky'
            hdulist.append(fits.CompImageHDU(sky_arr.astype('f4'), header=h, **compkw))


    if infoimfn is not None:
        fits.writeto(infoimfn, None, hdr, overwrite=True)
    
        with fits.open(infoimfn, mode="append", memmap=False) as hdulist:
    
            psffluxivar_all, flags_infoim_all, psfstamp_all = [], [], []
    
            for ib, band in enumerate(args.bands):
                psf_obj = psfs[ib]          # <-- ALWAYS use callable PSF object
                sqivar  = sqivars[ib]
                flagmap = flags[ib]
    
                psffluxivar_all.append(
                    ivarmap(sqivar, psf_obj(1024, 1024, stampsz=59)).astype("f4")
                )
                flags_infoim_all.append(
                    collapse_extraflags(flagmap, band).astype("u1")
                )
                psfstamp_all.append(
                    psf_obj(1024, 1024, stampsz=325)
                )
    
            psffluxivar_all = np.asarray(psffluxivar_all)   # (B, ny, nx)
            flags_infoim_all = np.asarray(flags_infoim_all) # (B, ny, nx)
            psfstamp_all     = np.asarray(psfstamp_all)     # (B, nypsf, nxpsf)
    
            # If single-band, write 2D images (not stacked) to match old behavior
            if len(args.bands) == 1:
                psffluxivar_to_write = psffluxivar_all[0]
                flags_to_write       = flags_infoim_all[0]
                psf_to_write         = psfstamp_all[0]
                tile_shape = psffluxivar_to_write.shape
            else:
                psffluxivar_to_write = psffluxivar_all
                flags_to_write       = flags_infoim_all
                psf_to_write         = psfstamp_all
                tile_shape = psffluxivar_to_write.shape[-2:]
    
            compkw = dict(compression_type="GZIP_1",
                          quantize_method=2,
                          tile_shape=tile_shape)
    
            h = hdr.copy(); h["EXTNAME"] = "psffluxivar"
            hdulist.append(fits.CompImageHDU(psffluxivar_to_write.astype("f4"), header=h, **compkw))
    
            h = hdr.copy(); h["EXTNAME"] = "infoflags"
            hdulist.append(fits.CompImageHDU(flags_to_write.astype("u1"), header=h,
                                             compression_type="GZIP_1", tile_shape=tile_shape))
    
            h = hdr.copy(); h["EXTNAME"] = "psf"
            hdulist.append(fits.ImageHDU(psf_to_write, header=h))

        hdulist.close(closed=True)


   