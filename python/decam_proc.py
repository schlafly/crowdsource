#!/usr/bin/env python

import os
import sys
import pdb
import argparse
import numpy
import numpy as np
import psf as psfmod
from astropy.io import fits
from astropy import wcs
from functools import partial
import crowdsource
from pqdm.processes import pqdm
import copy

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
              maskdiffuse=True, corrects7=True,wcutoff=0.0,contmask=False):
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
    if wcutoff != 0.0:
        print("weight_cutoff = {}".format(wcutoff))
    mzerowt = (((imded & ~(2**0 | 2**7)) != 0) |
               (imdew < wcutoff) | ~numpy.isfinite(imdew))
    if badpixmask is None:
        badpixmask = os.path.join(os.environ['DECAM_DIR'], 'data',
                                  'badpixmasksefs_comp.fits')
    badmask = fits.getdata(badpixmask, extname=extname)
    imded |= ((badmask != 0) * extrabits['badpix'])
    mzerowt = mzerowt | (badmask != 0)
    imdew[mzerowt] = 0.
    imdew[:] = numpy.sqrt(imdew)
    if corrects7 and (extname == 'S7'):
        imdei = correct_sky_offset(imdei, weight=imdew)
        half = imded.shape[1] // 2
        imded[:, half:] |= extrabits['s7unstable']
    if maskdiffuse:
        import nebulosity_mask
        nebmod = getattr(read_data, 'nebmod', None)
        if nebmod is None:
            modfn = os.path.join(os.environ['DECAM_DIR'], 'data', 'nebmaskmod',
                                 'weights', '27th_try')
            nebmod = nebulosity_mask.load_model(modfn)
            read_data.nebmod = nebmod
        if contmask == False:
            nebmask = nebulosity_mask.gen_mask(nebmod, imdei) == 0
        elif contmask == True:
            nebprob = nebulosity_mask.gen_prob(nebmod, imdei)
            # hard code decision boundary for now
            alpha = 2.0
            gam = 0.5
            decnum = np.empty(imdei.shape[0],imdei.shape[1],dtype=numpy.float32)
            decnum = numpy.divide(nebprob[:,:,0] + gam*nebprob[:,:,1],nebprob[:,:,1] + nebprob[:,:,2] + nebprob[:,:,3],out=decnum)
            nebmask = (decnum > alpha)
        else:
            raise ValueError("contmask must be bool")

        if numpy.any(nebmask):
            imded |= (nebmask * extrabits['diffuse'])
            imded |= (nebmask * (crowdsource.nodeblend_maskbit |
                                 crowdsource.sharp_maskbit))
            print('Masking nebulosity fraction, %5.2f' % (
                numpy.sum(nebmask)/1./numpy.sum(numpy.isfinite(nebmask))))
    if maskdiffuse:
        if contmask == True:
            return imdei, imdew, imded, nebmask, nebprob
        return imdei, imdew, imded, nebmask, None
    return imdei, imdew, imded, None, None

def process_image(imfn, ivarfn, dqfn, outfn=None, overwrite=False,
                  outdir=None, verbose=False, nproc=numpy.inf, resume=False,
                  outmodelfn=None, profile=False, maskdiffuse=True, wcutoff=0.0,
                  bin_weights_on=False, plot=False, miniter=4, maxiter=10,titer_thresh=2,
                  pixsz=9,contmask=False):
    if profile:
        import cProfile
        import pstats
        from guppy import hpy
        hp = hpy()
        before = hp.heap()
        pr = cProfile.Profile()
        pr.enable()
    if bin_weights_on == True:
        print("caution, weights are binarized")
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
        fits.writeto(outfn, None, prihdr, overwrite=overwrite)
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
        fits.writeto(outmodelfn, None, prihdr, overwrite=overwrite)
    count = 0
    fwhms = []
    for name in extnames:
        if name == 'PRIMARY':
            continue
        hdr = fits.getheader(imfn, extname=name)
        if 'FWHM' in hdr:
            fwhms.append(hdr['FWHM'])
    fwhms = numpy.array(fwhms)
    fwhms = fwhms[fwhms > 0]
    for name in extnames: #CCD for loop
        if name == 'PRIMARY':
            continue
        if extnamesdone is not None and name in extnamesdone:
            print('Skipping %s, extension %s; already done.' % (imfn, name))
            continue
        if verbose:
            print('Fitting %s, extension %s.' % (imfn, name))
            sys.stdout.flush()
        im, wt, dq, msk, prb = read_data(imfn, ivarfn, dqfn, name,
                               maskdiffuse=maskdiffuse,wcutoff=wcutoff,contmask=contmask)
        hdr = fits.getheader(imfn, extname=name)
        fwhm = hdr.get('FWHM', numpy.median(fwhms))
        if fwhm <= 0.:
            fwhm = 4.
        fwhmmn, fwhmsd = numpy.mean(fwhms), numpy.std(fwhms)
        if fwhmsd > 0.4:
            fwhm = fwhmmn
        psf = decam_psf(filt[0], fwhm,pixsz=pixsz)
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
                                 maxstars=320000,bin_weights_on=bin_weights_on,
                                 ccd=name, plot=plot, miniter=miniter, maxiter=maxiter,
                                 titer_thresh=titer_thresh,msk=msk,prb=prb)

        cat, modelim, skyim, psf = res
        if len(cat) > 0:
            ra, dec = wcs0.all_pix2world(cat['y'], cat['x'], 0.)
        else:
            ra = numpy.zeros(0, dtype='f8')
            dec = numpy.zeros(0, dtype='f8')
        from numpy.lib.recfunctions import rec_append_fields
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
        fits.append(outfn, numpy.zeros(0), hdr) # append some header
        hdupsf = fits.BinTableHDU(psf.serialize())
        hdupsf.name = hdr['EXTNAME'][:-4] + '_PSF'
        hducat = fits.BinTableHDU(cat)
        hducat.name = hdr['EXTNAME'][:-4] + '_CAT'
        hdulist = fits.open(outfn, mode='append')
        hdulist.append(hdupsf) #append the psf field for the ccd
        hdulist.append(hducat) #append the cat field for the ccd
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
            if msk is not None:
                hdr['EXTNAME'] = hdr['EXTNAME'][:-4] + '_MSK'
                modhdulist.append(fits.CompImageHDU(msk.astype('i4'), hdr, **compkw))
            modhdulist.close(closed=True)
        count += 1
        if count > nproc:
            break
    if profile:
        pr.disable()
        pstats.Stats(pr).sort_stats('cumulative').print_stats(60)
        after = hp.heap()
        leftover = after - before
        print(leftover)

def process_image_p(imfn, ivarfn, dqfn, outfn=None, overwrite=False,
                  outdir=None, verbose=False, nproc=numpy.inf, resume=False,
                  outmodelfn=None, profile=False, maskdiffuse=True, wcutoff=0.0,
                  bin_weights_on=False, plot=False, miniter=4, maxiter=10,titer_thresh=2, num_procs=1,pixsz=9):
    if profile:
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.enable()
    if bin_weights_on == True:
        print("caution, weights are binarized")
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
        fits.writeto(outfn, None, prihdr, overwrite=overwrite)
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
        fits.writeto(outmodelfn, None, prihdr, overwrite=overwrite)
    fwhms = []
    for name in extnames:
        if name == 'PRIMARY':
            continue
        hdr = fits.getheader(imfn, extname=name)
        if 'FWHM' in hdr:
            fwhms.append(hdr['FWHM'])
    fwhms = numpy.array(fwhms)
    fwhms = fwhms[fwhms > 0]

    newexts = numpy.setdiff1d(numpy.setdiff1d(extnames,extnamesdone),['PRIMARY'])

    if nproc != numpy.inf:
        max_nproc = numpy.min([nproc, len(newexts)])
        nargs = [(n, outfn, imfn, ivarfn, dqfn, outmodelfn, maskdiffuse, wcutoff, fwhms, bin_weights_on, verbose, filt, brightstars, prihdr, plot, miniter, maxiter,titer_thresh,pixsz) for n in newexts[0:max_nproc]]
    else:
        nargs = [(n, outfn, imfn, ivarfn, dqfn, outmodelfn, maskdiffuse, wcutoff, fwhms, bin_weights_on, verbose, filt, brightstars, prihdr, plot, miniter, maxiter,titer_thresh,pixsz) for n in newexts]

    result = pqdm(nargs, sub_process, n_jobs=num_procs)

    for s in result:
        hdr = fits.Header.fromstring(s[0])
        fits.append(outfn, numpy.zeros(0), hdr) # append some header

        hdupsf = fits.BinTableHDU(s[1])
        hdupsf.name = hdr['EXTNAME'][:-4] + '_PSF'

        hducat = fits.BinTableHDU(s[2])
        hducat.name = hdr['EXTNAME'][:-4] + '_CAT'

        hdulist = fits.open(outfn, mode='append')
        hdulist.append(hdupsf) #append the psf field for the ccd
        hdulist.append(hducat) #append the cat field for the ccd
        hdulist.close(closed=True)

        if outmodelfn:
            hdr['EXTNAME'] = hdr['EXTNAME'][:-4] + '_MOD'
            compkw = {'compression_type': 'GZIP_1',
                      'quantize_method': 1, 'quantize_level': -4,
                      'tile_size': s[3].shape}
            model = fits.CompImageHDU(s[3], hdr, **compkw)
            hdr['EXTNAME'] = hdr['EXTNAME'][:-4] + '_SKY'
            sky = fits.CompImageHDU(s[4], hdr, **compkw)

            modhdulist = fits.open(outmodelfn, mode='append')
            modhdulist.append(model)
            modhdulist.append(sky)
            modhdulist.close(closed=True)

    if profile:
        pr.disable()
        pstats.Stats(pr).sort_stats('cumulative').print_stats(60)

def sub_process(args):
    name, outfn, imfn, ivarfn, dqfn, outmodelfn, maskdiffuse, wcutoff, fwhms, bin_weights_on, verbose, filt, brightstars, prihdr, plot, miniter, maxiter,titer_thresh,pixsz = args
    if verbose:
        print('Fitting %s, extension %s.' % (imfn, name))
        sys.stdout.flush()
    im, wt, dq = read_data(imfn, ivarfn, dqfn, name,
                           maskdiffuse=maskdiffuse,wcutoff=wcutoff)
    hdr = fits.getheader(imfn, extname=name)
    fwhm = hdr.get('FWHM', numpy.median(fwhms))
    if fwhm <= 0.:
        fwhm = 4.
    fwhmmn, fwhmsd = numpy.mean(fwhms), numpy.std(fwhms)
    if fwhmsd > 0.4:
        fwhm = fwhmmn
    psf = decam_psf(filt[0], fwhm, pixsz)
    wcs0 = wcs.WCS(hdr)
    from astropy.coordinates.angle_utilities import angular_separation
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
                             maxstars=320000,bin_weights_on=bin_weights_on,
                             ccd=name, plot=plot,miniter=miniter, maxiter=maxiter,titer_thresh=titer_thresh)

    cat, modelim, skyim, psf = res
    if len(cat) > 0:
        ra, dec = wcs0.all_pix2world(cat['y'], cat['x'], 0.)
    else:
        ra = numpy.zeros(0, dtype='f8')
        dec = numpy.zeros(0, dtype='f8')
    from numpy.lib.recfunctions import rec_append_fields
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

    if outmodelfn:
        return [hdr.tostring(), psf.serialize(), cat, modelim, skyim]

    return [hdr.tostring(), psf.serialize(), cat]

def decam_psf(filt, fwhm, pixsz = 9, nlinperpar = 3):
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
    pixsz = pixsz
    extraparam = numpy.zeros(
        1, dtype=[('convparam', 'f4', 3*nlinperpar+1),
                  ('resparam', 'f4', (nlinperpar, pixsz, pixsz))])
    extraparam['convparam'][0, 0:4] = [convpsffwhm, 1., 0., 1.]
    extraparam['resparam'][0, :, :, :] = 0.
    tpsf.extraparam = extraparam
    tpsf.fitfun = partial(psfmod.fit_linear_static_wing, filter=filt, pixsz=pixsz)
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
        dq[xl:xr, yl:yr] |= (crowdsource.nodeblend_maskbit |
                             crowdsource.sharp_maskbit)
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
    parser.add_argument('--parallel', type=int,
                        default=1, help='num of parallel processors')
    parser.add_argument('--miniter', type=int,
                        default=4, help='min fit im iterations')
    parser.add_argument('--maxiter', type=int,
                        default=10, help='max fit im iterations')
    parser.add_argument('--titer_thresh', type=int,
                        default=2, help='threshold for deblending increase')
    parser.add_argument('--pixsz', type=int,
                        default=9, help='size of pixelized psf stamp')
    parser.add_argument('--ccd_num', type=int,
                        default=numpy.inf, help='limit to num ccds run')
    parser.add_argument('--profile', '-p', action='store_true',
                        help='print profiling statistics')
    parser.add_argument('--no-mask-diffuse', action='store_true',
                        help='turn off nebulosity masking')
    parser.add_argument('--wcutoff', type=float,
                        default=0.0, help='cutoff for inverse variances')
    parser.add_argument('--bin_weights_on', action='store_true',
                        help='make WLS depend on binary weights only')
    parser.add_argument('--contmask', action='store_true',
                        help='make WLS depend on binary weights only')
    parser.add_argument('--plot_on', action='store_true',
                        help='save psf diagonsitic plots at each titer')
    parser.add_argument('imfn', type=str, help='Image file name')
    parser.add_argument('ivarfn', type=str, help='Inverse variance file name')
    parser.add_argument('dqfn', type=str, help='Data quality file name')
    args = parser.parse_args()
    if args.parallel > 1:
        process_image_p(args.imfn, args.ivarfn, args.dqfn, outfn=args.outfn,
                      outmodelfn=args.outmodelfn,
                      verbose=args.verbose, outdir=args.outdir,
                      resume=args.resume, profile=args.profile,
                      maskdiffuse=(not args.no_mask_diffuse),wcutoff=args.wcutoff,
                      bin_weights_on=args.bin_weights_on, num_procs=args.parallel,
                      nproc=args.ccd_num,plot=args.plot_on, miniter=args.miniter,
                      maxiter=args.maxiter, titer_thresh=args.titer_thresh,pixsz=args.pixsz,
                      contmask=args.contmask)
    else:
        process_image(args.imfn, args.ivarfn, args.dqfn, outfn=args.outfn,
                      outmodelfn=args.outmodelfn,
                      verbose=args.verbose, outdir=args.outdir,
                      resume=args.resume, profile=args.profile,
                      maskdiffuse=(not args.no_mask_diffuse),wcutoff=args.wcutoff,
                      bin_weights_on=args.bin_weights_on,nproc=args.ccd_num,
                      plot=args.plot_on,miniter=args.miniter,maxiter=args.maxiter,
                      titer_thresh=args.titer_thresh,pixsz=args.pixsz,
                      contmask=args.contmask)
