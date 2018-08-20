#!/usr/bin/env python

import os
import sys
import pdb
import argparse
import numpy
import psf as psfmod
from astropy.io import fits
from astropy import wcs
from functools import partial
import crowdsource


badpixmaskfn = '/n/fink2/www/eschlafly/decam/badpixmasksefs_comp.fits'

extrabits = ({'badpix': 2**20,
              'diffuse': 2**21,
              's7unstable': 2**22,
              'brightstar': 2**23})


def read(imfn, extname, **kw):
    ivarfn = imfn.replace('_ooi_', '_oow_')
    dqfn = imfn.replace('_ooi_', '_ood_')
    return read_data(imfn, ivarfn, dqfn, extname, **kw)


def read_data(imfn, ivarfn, dqfn, extname, badpixmask=None,
              maskdiffuse=True, corrects7=True):
    import warnings
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        imh = fits.getheader(imfn)
        imdei = fits.getdata(imfn, extname=extname).copy()
        imdew = fits.getdata(ivarfn, extname=extname).copy()
        imded = fits.getdata(dqfn, extname=extname).copy()
    # suppress endless nonstandard keyword warnings on read
    for warning in wlist:
        if 'following header keyword' in str(warning.message):
            continue
        else:
            print(warning)
    # support old versions of CP with different DQ meanings
    from distutils.version import LooseVersion
    if LooseVersion(imh['PLVER']) < LooseVersion('V3.5'):
        imdedo = imded
        imded = numpy.zeros_like(imded)
        imded[(imdedo & 2**7) != 0] = 7
        imded[(imdedo & 2**4) != 0] = 5
        imded[(imdedo & 2**6) != 0] = 4
        imded[(imdedo & 2**1) != 0] = 3
        imded[(imdedo & 2**0) != 0] = 1
        # values 2, 8 don't seem to exist in early CP images;
        # likewise bits 2, 3, 5 don't really have meaning in recent CP images
        # (interpolated, unused, unused)
    imded = 2**imded
    # flag 7 does not seem to indicate problems with the pixels.
    mzerowt = (((imded & ~(2**0 | 2**7)) != 0) |
               (imdew < 0.) | ~numpy.isfinite(imdew))
    if badpixmask is None:
        badpixmask = os.path.join(os.environ['DECAM_DIR'], 'data', 
                                  'badpixmasksefs_comp.fits')
    badmask = fits.getdata(badpixmask, extname=extname)
    imded |= ((badmask != 0) * extrabits['badpix'])
    mzerowt = mzerowt | (badmask != 0)
    imdew[mzerowt] = 0.
    imdew[:] = numpy.sqrt(imdew)
    if maskdiffuse:
        import nebulosity_mask
        nebmod = getattr(read_data, 'nebmod', None)
        if nebmod is None:
            modfn = os.path.join(os.environ['DECAM_DIR'], 'data', 'nebmaskmod',
                                 'weights', '27th_try')
            nebmod = nebulosity_mask.load_model(modfn)
            read_data.nebmod = nebmod
        nebmask = nebulosity_mask.gen_mask(nebmod, imdei) == 0
        if numpy.any(nebmask):
            imded |= (nebmask * extrabits['diffuse'])
            print('Masking nebulosity, %5.2f' % (
                numpy.sum(nebmask)/1./numpy.sum(numpy.isfinite(nebmask))))
    if corrects7 and (extname == 'S7'):
        imdei = correct_sky_offset(imdei, weight=imdew)
        half = imded.shape[1] // 2
        imded[:, half:] |= extrabits['s7unstable']
    return imdei, imdew, imded


def process_image(imfn, ivarfn, dqfn, outfn=None, clobber=False,
                  outdir=None, verbose=False, nproc=numpy.inf, resume=False,
                  outmodelfn=None, profile=False, maskdiffuse=True):
    if profile:
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.enable()

    with fits.open(imfn) as hdulist:
        extnames = [hdu.name for hdu in hdulist]
    if 'PRIMARY' not in extnames:
        raise ValueError('No PRIMARY header in file')
    prihdr = fits.getheader(imfn, extname='PRIMARY')
    if 'CENTRA' in prihdr:
        bstarfn = os.path.join(os.environ['DECAM_DIR'], 'data',
                               'tyc2brighttrim.fits')
        brightstars = fits.getdata(bstarfn)
        from astropy.coordinates.angle_utilities import angular_separation
        sep = angular_separation(numpy.radians(brightstars['ra']),
                                 numpy.radians(brightstars['dec']),
                                 numpy.radians(prihdr['CENTRA']),
                                 numpy.radians(prihdr['CENTDEC']))
        sep = numpy.degrees(sep)
        m = sep < 3
        brightstars = brightstars[m]
        dmjd = prihdr['MJD-OBS'] - 51544.5  # J2000 MJD.
        cosd = numpy.cos(numpy.radians(numpy.clip(brightstars['dec'],
                                                  -89.9999, 89.9999)))
        brightstars['ra'] += dmjd*brightstars['pmra']/365.25/cosd/1000/60/60
        brightstars['dec'] += dmjd*brightstars['pmde']/365.25/1000/60/60
    else:
        brightstars = None
    filt = prihdr['filter']
    if outfn is None or len(outfn) == 0:
        outfn = os.path.splitext(os.path.basename(imfn))[0]
        if outfn[-5:] == '.fits':
            outfn = outfn[:-5]
        outfn = outfn + '.cat.fits'
    if outdir is not None:
        outfn = os.path.join(outdir, outfn)
    if not resume or not os.path.exists(outfn):
        fits.writeto(outfn, None, prihdr, clobber=clobber)
        extnamesdone = None
    else:
        hdulist = fits.open(outfn)
        extnamesdone = []
        for hdu in hdulist:
            if hdu.name == 'PRIMARY':
                continue
            ext, exttype = hdu.name.split('_')
            if exttype != 'CAT':
                continue
            extnamesdone.append(ext)
        hdulist.close()
    if outmodelfn and (not resume or not os.path.exists(outmodelfn)):
        fits.writeto(outmodelfn, None, prihdr, clobber=clobber)
    count = 0
    fwhms = []
    for name in extnames:
        if name is 'PRIMARY':
            continue
        hdr = fits.getheader(imfn, extname=name)
        if 'FWHM' in hdr:
            fwhms.append(hdr['FWHM'])
    fwhms = numpy.array(fwhms)
    fwhms = fwhms[fwhms > 0]
    for name in extnames:
        if name is 'PRIMARY':
            continue
        if extnamesdone is not None and name in extnamesdone:
            print('Skipping %s, extension %s; already done.' % (imfn, name))
            continue
        if verbose:
            print('Fitting %s, extension %s.' % (imfn, name))
            sys.stdout.flush()
        im, wt, dq = read_data(imfn, ivarfn, dqfn, name, 
                               maskdiffuse=maskdiffuse)
        hdr = fits.getheader(imfn, extname=name)
        fwhm = hdr.get('FWHM', numpy.median(fwhms))
        if fwhm <= 0.:
            fwhm = 4.
        fwhmmn, fwhmsd = numpy.mean(fwhms), numpy.std(fwhms)
        if fwhmsd > 0.4:
            fwhm = fwhmmn
        psf = decam_psf(filt[0], fwhm)
        wcs0 = wcs.WCS(hdr)
        if brightstars is not None:
            sep = angular_separation(numpy.radians(brightstars['ra']),
                                     numpy.radians(brightstars['dec']),
                                     numpy.radians(hdr['CENRA1']),
                                     numpy.radians(hdr['CENDEC1']))
            sep = numpy.degrees(sep)
            m = sep < 0.2
            # CCD is 4094 pix wide => everything is at most 0.15 deg
            # from center
            if numpy.any(m):
                yb, xb = wcs0.all_world2pix(brightstars['ra'][m],
                                            brightstars['dec'][m], 0)
                vmag = brightstars['vtmag'][m]
                # WCS module and I order x and y differently...
                m = ((xb > 0) & (xb < im.shape[0]) &
                     (yb > 0) & (yb < im.shape[1]))
                if numpy.any(m):
                    xb, yb = xb[m], yb[m]
                    vmag = vmag[m]
                    blist = [xb, yb, vmag]
                else:
                    blist = None
            else:
                blist = None
        else:
            blist = None

        if blist is not None:
            dq = mask_very_bright_stars(dq, blist)

        # the actual fit
        res = crowdsource.fit_im(im, psf, ntilex=4, ntiley=2,
                                 weight=wt, dq=dq,
                                 psfderiv=True, refit_psf=True,
                                 verbose=verbose, blist=blist,
                                 maxstars=320000)

        cat, modelim, skyim, psf = res
        if len(cat) > 0:
            ra, dec = wcs0.all_pix2world(cat['y'], cat['x'], 0.)
        else:
            ra = numpy.zeros(0, dtype='f8')
            dec = numpy.zeros(0, dtype='f8')
        from matplotlib.mlab import rec_append_fields
        decapsid = numpy.zeros(len(cat), dtype='i8')
        decapsid[:] = (prihdr['EXPNUM']*2**32*2**7 +
                       hdr['CCDNUM']*2**32 +
                       numpy.arange(len(cat), dtype='i8'))
        if verbose:
            print('Writing %s %s, found %d sources.' % (outfn, name, len(cat)))
            sys.stdout.flush()
        hdr['EXTNAME'] = hdr['EXTNAME']+'_HDR'
        if numpy.any(wt > 0):
            hdr['GAINCRWD'] = numpy.nanmedian((im*wt**2.)[wt > 0])
            hdr['SKYCRWD'] = numpy.nanmedian(skyim[wt > 0])
        else:
            hdr['GAINCRWD'] = 4
            hdr['SKYCRWD'] = 0
        if len(cat) > 0:
            hdr['FWHMCRWD'] = numpy.nanmedian(cat['fwhm'])
        else:
            hdr['FWHMCRWD'] = 0.0
        gain = hdr['GAINCRWD']*numpy.ones(len(cat), dtype='f4')
        cat = rec_append_fields(cat, ['ra', 'dec', 'decapsid', 'gain'],
                                [ra, dec, decapsid, gain])
        fits.append(outfn, numpy.zeros(0), hdr)
        hdupsf = fits.BinTableHDU(psf.serialize())
        hdupsf.name = hdr['EXTNAME'][:-4] + '_PSF'
        hducat = fits.BinTableHDU(cat)
        hducat.name = hdr['EXTNAME'][:-4] + '_CAT'
        hdulist = fits.open(outfn, mode='append')
        hdulist.append(hdupsf)
        hdulist.append(hducat)
        hdulist.close(closed=True)
        if outmodelfn:
            modhdulist = fits.open(outmodelfn, mode='append')
            hdr['EXTNAME'] = hdr['EXTNAME'][:-4] + '_MOD'
            # RICE should be significantly better here and supported in
            # mrdfits?, but compression_type=RICE_1 seems to cause
            # quantize_level to be ignored.
            compkw = {'compression_type': 'GZIP_1',
                      'quantize_method': 1, 'quantize_level': -4,
                      'tile_size': modelim.shape}
            modhdulist.append(fits.CompImageHDU(modelim, hdr, **compkw))
            hdr['EXTNAME'] = hdr['EXTNAME'][:-4] + '_SKY'
            modhdulist.append(fits.CompImageHDU(skyim, hdr, **compkw))
            modhdulist.close(closed=True)
        count += 1
        if count > nproc:
            break
    if profile:
        pr.disable()
        pstats.Stats(pr).sort_stats('cumulative').print_stats(60)


def decam_psf(filt, fwhm):
    if filt not in 'ugrizY':
        tpsf = psfmod.moffat_psf(fwhm, stampsz=511, deriv=False)
        return psfmod.SimplePSF(tpsf)
    fname = os.path.join(os.environ['DECAM_DIR'], 'data', 'psfs',
                         'psf_%s_deconv_mod.fits.gz' % filt[0])
    normalizesz = 59
    tpsf = fits.getdata(fname).T.copy()
    tpsf /= numpy.sum(psfmod.central_stamp(tpsf, normalizesz))
    # omitting central_stamp here places too much
    # emphasis on the wings relative to the pipeline estimate.
    tpsffwhm = psfmod.neff_fwhm(psfmod.central_stamp(tpsf))
    from scipy.ndimage.filters import convolve
    if tpsffwhm < fwhm:
        convpsffwhm = numpy.sqrt(fwhm**2.-tpsffwhm**2.)
        convpsf = psfmod.moffat_psf(convpsffwhm, stampsz=39, deriv=False)
        tpsf = convolve(tpsf, convpsf, mode='constant', cval=0., origin=0)
    else:
        convpsffwhm = 0.
    tpsf = psfmod.stamp2model(numpy.array([tpsf, tpsf, tpsf, tpsf]),
                              normalize=normalizesz)
    nlinperpar = 3
    pixsz = 9
    extraparam = numpy.zeros(
        1, dtype=[('convparam', 'f4', 3*nlinperpar+1),
                  ('resparam', 'f4', (nlinperpar, pixsz, pixsz))])
    extraparam['convparam'][0, 0:4] = [convpsffwhm, 1., 0., 1.]
    extraparam['resparam'][0, :, :, :] = 0.
    tpsf.extraparam = extraparam
    tpsf.fitfun = partial(psfmod.fit_linear_static_wing, filter=filt)
    return tpsf


def correct_sky_offset(im, weight=None):
    xx = numpy.arange(im.shape[0], dtype='f4')
    xx -= numpy.median(xx)
    xx = xx.reshape(-1, 1)
    if weight is None:
        weight = numpy.ones_like(im)*10
    half = (im.shape[1] // 2)
    bdy = 10
    use = ((weight[:, half+bdy:half:-1] > 0) &
           (weight[:, half-bdy:half] > 0))
    if numpy.sum(use) == 0:
        return im
    import psf
    delta = im[:, half+bdy:half:-1] - im[:, half-bdy:half]
    weight = numpy.min([weight[:, half+bdy:half:-1],
                        weight[:, half-bdy:half]], axis=0)

    def objective(par):
        return psf.damper(((delta - par[0] - par[1]*xx)*weight)[use], 5)
    guessoff = numpy.median(delta[use])
    from scipy.optimize import leastsq
    par = leastsq(objective, [guessoff, 0.])[0]
    im[:, half:] -= (par[0] + par[1]*xx)
    return im


def mask_very_bright_stars(dq, blist):
    dq = dq.copy()
    maskradpermag = 50
    for x, y, mag in zip(*blist):
        maskrad = maskradpermag*(11-mag)
        if maskrad < 50:
            continue
        maskrad = numpy.clip(maskrad, 0, 500)
        xl, xr = numpy.clip([x-maskrad, x+maskrad], 0,
                            dq.shape[0]-1).astype('i4')
        yl, yr = numpy.clip([y-maskrad, y+maskrad], 0,
                            dq.shape[1]-1).astype('i4')
        dq[xl:xr, yl:yr] |= extrabits['brightstar']
    return dq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit DECam frame')

    parser.add_argument('--outfn', '-o', type=str,
                        default=None, help='output file name')
    parser.add_argument('--outmodelfn', '-m', type=str,
                        default=None, help='output model file name')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--outdir', '-d', help='output directory',
                        type=str, default=None)
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume if file already exists')
    parser.add_argument('--profile', '-p', action='store_true',
                        help='print profiling statistics')
    parser.add_argument('--no-mask-diffuse', action='store_true',
                        help='turn off nebulosity masking')
    parser.add_argument('imfn', type=str, help='Image file name')
    parser.add_argument('ivarfn', type=str, help='Inverse variance file name')
    parser.add_argument('dqfn', type=str, help='Data quality file name')
    args = parser.parse_args()
    process_image(args.imfn, args.ivarfn, args.dqfn, outfn=args.outfn,
                  outmodelfn=args.outmodelfn,
                  verbose=args.verbose, outdir=args.outdir,
                  resume=args.resume, profile=args.profile, 
                  maskdiffuse=(not args.no_mask_diffuse))
