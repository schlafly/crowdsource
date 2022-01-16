#!/usr/bin/env python

import os
import sys
import pdb
import argparse
import numpy
import numpy as np
import crowdsource.psf as psfmod
from astropy.io import fits
from astropy import wcs
from functools import partial
from crowdsource import crowdsource_base
from scipy.ndimage import zoom

import os
if 'DECAM_DIR' not in os.environ:
    decam_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"decam_dir")
    os.environ['DECAM_DIR'] = decam_dir

extrabits = ({'badpix': 2**20,
              'diffuse': 2**21,
              's7unstable': 2**22,
              'brightstar': 2**23,
              'galaxy': 2**24})


# this is a convenience function for developer access, not used in processing
def read(imfn, extname, **kw):
    ivarfn = imfn.replace('_ooi_', '_oow_')
    dqfn = imfn.replace('_ooi_', '_ood_')
    return read_data(imfn, ivarfn, dqfn, extname, **kw)


# wrapper to make file reading easier using the decam pattern
def decaps_filenames(base, date, filtf, vers):
    imfn = base+date+"_ooi_"+filtf+"_"+vers+".fits.fz"
    ivarfn = base+date+"_oow_"+filtf+"_"+vers+".fits.fz"
    dqfn = base+date+"_ood_"+filtf+"_"+vers+".fits.fz"
    return imfn, ivarfn, dqfn


# actual read function
def read_data(imfn, ivarfn, dqfn, extname, badpixmask=None,
              maskdiffuse=True, corrects7=True, wcutoff=0.0, contmask=False,
              maskgal=False, verbose=False):
    import warnings
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        imh = fits.getheader(imfn)
        hdr = fits.getheader(imfn, extname=extname)
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
               (imdew < wcutoff) | ~numpy.isfinite(imdew))
    if badpixmask is None:
        badpixmask = os.path.join(os.environ['DECAM_DIR'], 'data',
                                  'badpixmasksefs_comp.fits')
    if "I" in extname:
        extidx = extname.index("I")
        bextname = extname[:extidx]
    else:
        bextname = extname
    badmask = fits.getdata(badpixmask, extname=bextname)
    imded |= ((badmask != 0) * extrabits['badpix'])
    mzerowt = mzerowt | (badmask != 0)
    imdew[mzerowt] = 0.
    imdew[:] = numpy.sqrt(imdew)
    if corrects7 and (extname[:2] == 'S7'):
        imdei = correct_sky_offset(imdei, weight=imdew)
        half = imded.shape[1] // 2
        imded[:, half:] |= extrabits['s7unstable']
    if maskgal:
        if imh["WCSCAL"] == "Successful":
            import galaxy_mask
            leda = getattr(read_data, 'leda', None)
            if leda is None:
                leda = galaxy_mask.read_leda_decaps()
                read_data.leda = leda
            gmsk = galaxy_mask.galaxy_mask(hdr, leda)
            if numpy.any(gmsk):
                imded |= (gmsk * extrabits['galaxy'])
                imded |= (gmsk * crowdsource_base.nodeblend_maskbit)
        else:
            if verbose:
                print("WCSCAL Unsucessful, Skipping galaxy masking...")

    if maskdiffuse:
        import nebulosity_mask
        nebmod = getattr(read_data, 'nebmod', None)
        if nebmod is None:
            modfn = os.path.join(os.environ['DECAM_DIR'], 'data', 'nebmaskmod',
                                 'weights', '27th_try')
            nebmod = nebulosity_mask.load_model(modfn)
            read_data.nebmod = nebmod
        if not contmask:
            nebmask = nebulosity_mask.gen_mask(nebmod, imdei) == 0
            nebprob = None
        else:
            nebmask, nebprob = nebulosity_mask.gen_prob(
                nebmod, imdei, return_prob=True)

        if numpy.any(nebmask):
            imded |= (nebmask * extrabits['diffuse'])
            imded |= (nebmask * (crowdsource_base.nodeblend_maskbit |
                                 crowdsource_base.sharp_maskbit))
            if verbose:
                print('Masking nebulosity fraction, %5.2f' % (
                    numpy.sum(nebmask)/1./numpy.sum(numpy.isfinite(nebmask))))
    else:
        nebprob = None

    return imdei, imdew, imded, nebprob


# work function to process each ccd
def process_one_ccd(name, bigdict):
    imfn = bigdict['imfn']
    ivarfn = bigdict['ivarfn']
    dqfn = bigdict['dqfn']
    maskdiffuse = bigdict['maskdiffuse']
    maskgal = bigdict['maskgal']
    verbose = bigdict['verbose']
    wcutoff = bigdict['wcutoff']
    contmask = bigdict['contmask']
    fwhms = bigdict['fwhms']
    filt = bigdict['filt']
    pixsz = bigdict['pixsz']
    brightstars = bigdict['brightstars']
    bmask_deblend = bigdict['bmask_deblend']
    plot = bigdict['plot']
    miniter = bigdict['miniter']
    maxiter = bigdict['maxiter']
    titer_thresh = bigdict['titer_thresh']
    expnum = bigdict['expnum']
    if verbose:
        print('Fitting %s, extension %s.' % (imfn, name))
        sys.stdout.flush()
    im, wt, dq, prb = read_data(
        imfn, ivarfn, dqfn, name, maskdiffuse=maskdiffuse,
        wcutoff=wcutoff, contmask=contmask, maskgal=maskgal,
        verbose=verbose)
    hdr = fits.getheader(imfn, extname=name)
    fwhm = hdr.get('FWHM', numpy.median(fwhms))
    if fwhm <= 0.:
        fwhm = 4.
    fwhmmn, fwhmsd = numpy.mean(fwhms), numpy.std(fwhms)
    if fwhmsd > 0.4:
        fwhm = fwhmmn
    psf = decam_psf(filt[0], fwhm, pixsz=pixsz)
    wcs0 = wcs.WCS(hdr)
    from astropy.coordinates.angle_utilities import angular_separation
    if brightstars is not None:
        raccdcen, decccdcen = wcs0.all_pix2world(
            im.shape[1]//2, im.shape[0]//2, 0)
        sep = angular_separation(numpy.radians(brightstars['ra']),
                                 numpy.radians(brightstars['dec']),
                                 numpy.radians(raccdcen),
                                 numpy.radians(decccdcen))
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
                if not bmask_deblend:
                    dq = mask_very_bright_stars(dq, blist)
            else:
                blist = None
        else:
            blist = None
    else:
        blist = None

    # the actual fit (which has a nested iterative fit)
    res = crowdsource_base.fit_im(
        im, psf, ntilex=4, ntiley=2, weight=wt, dq=dq, psfderiv=True,
        refit_psf=True, verbose=verbose, blist=blist, maxstars=320000,
        ccd=name, plot=plot, miniter=miniter, maxiter=maxiter,
        titer_thresh=titer_thresh)
    cat, modelim, skyim, psf = res
    if len(cat) > 0:
        ra, dec = wcs0.all_pix2world(cat['y'], cat['x'], 0.)
    else:
        ra = numpy.zeros(0, dtype='f8')
        dec = numpy.zeros(0, dtype='f8')
    from numpy.lib.recfunctions import rec_append_fields
    decapsid = numpy.zeros(len(cat), dtype='i8')
    decapsid[:] = (expnum*2**32*2**7 +
                   hdr['CCDNUM']*2**32 +
                   numpy.arange(len(cat), dtype='i8'))
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
    if prb is not None:
        prnebdat = [
            crowdsource_base.extract_im(cat['x'], cat['y'], prb[:, :, i])
            for i in range(prb.shape[2])]
        prnebnames = ['prN', 'prL', 'prR', 'prE']
        prbexport = zoom(prb, (1/8, 1/8, 1), order=1)
    else:
        prnebdat = []
        prnebnames = []
        prbexport = None
    cat = rec_append_fields(
        cat,
        ['ra', 'dec', 'decapsid', 'gain'] + prnebnames,
        [ra, dec, decapsid, gain] + prnebdat)
    msk = (dq & extrabits['diffuse']) != 0
    return cat, modelim, skyim, psf, hdr.tostring(), msk, prbexport, name


# main processing function for all decam handling
def process_image(base, date, filtf, vers, outfn=None, overwrite=False,
                  outmodel=False, outdirc=None, outdirm=None, verbose=False,
                  resume=False, bmask_deblend=False,
                  maskgal=False, maskdiffuse=True, contmask=False,
                  nproc=numpy.inf,
                  extnamelist=None, plot=False, profile=False, miniter=4,
                  maxiter=10, titer_thresh=2, pixsz=9, wcutoff=0.0,
                  nthreads=1,
                  inject = 0, injextnamelist = None, injectfrac = 0.1,
                  modsaveonly=False, donefrommod=False,noModsave=False):
    if profile:
        import cProfile
        import pstats
        from guppy import hpy
        hp = hpy()
        before = hp.heap()
        pr = cProfile.Profile()
        pr.enable()

    imfn, ivarfn, dqfn = decaps_filenames(base, date, filtf, vers)
    with fits.open(imfn) as hdulist:
        extnames = [hdu.name for hdu in hdulist]
    if 'PRIMARY' not in extnames:
        raise ValueError('No PRIMARY header in file')
    prihdr = fits.getheader(imfn, extname='PRIMARY')

    bstarfn = os.path.join(os.environ['DECAM_DIR'], 'data',
                           'tyc2brighttrim.fits')
    brightstars = fits.getdata(bstarfn)
    from astropy.coordinates.angle_utilities import angular_separation
    from astropy.coordinates import SkyCoord
    from astropy import units
    coordcen = SkyCoord(
        ra=prihdr['RA'], dec=prihdr['DEC'],
        unit=(units.hourangle, units.deg))
    sep = angular_separation(numpy.radians(brightstars['ra']),
                             numpy.radians(brightstars['dec']),
                             coordcen.ra.to(units.radian).value,
                             coordcen.dec.to(units.radian).value)
    sep = numpy.degrees(sep)
    m = sep < 3
    brightstars = brightstars[m]
    dmjd = prihdr['MJD-OBS'] - 51544.5  # J2000 MJD.
    cosd = numpy.cos(numpy.radians(numpy.clip(brightstars['dec'],
                                              -89.9999, 89.9999)))
    brightstars['ra'] += dmjd*brightstars['pmra']/365.25/cosd/1000/60/60
    brightstars['dec'] += dmjd*brightstars['pmde']/365.25/1000/60/60
    filt = prihdr['filter']
    # cat filename handling
    if outfn is None or len(outfn) == 0:
        outfn = os.path.splitext(os.path.basename(imfn))[0]
        if outfn[-5:] == '.fits':
            outfn = outfn[:-5]
        outfn = outfn + '.cat.fits'
    if outdirc is not None:
        outfn = os.path.join(outdirc, outfn)
    if not resume or not os.path.exists(outfn):
        fits.writeto(outfn, None, prihdr, overwrite=overwrite)
        extnamesdone = None
    else:
        hdulist = fits.open(outfn)
        extnamesdone = []
        for hdu in hdulist:
            if hdu.name == 'PRIMARY':
                continue
            extfull = hdu.name.split('_')
            ext = "_".join(extfull[:-1])
            if extfull[-1] != 'CAT':
                continue
            extnamesdone.append(ext)
        hdulist.close()
    # model filename handling
    if outmodel:
        outmodelfn = os.path.splitext(os.path.basename(imfn))[0]
        if outmodelfn[-5:] == '.fits':
            outmodelfn = outmodelfn[:-5]
        outmodelfn = outmodelfn + '.mod.fits'
        if outdirm is not None:
            outmodelfn = os.path.join(outdirm, outmodelfn)
        if (not resume or not os.path.exists(outmodelfn)):
            fits.writeto(outmodelfn, None, prihdr, overwrite=overwrite)
        else:
            if donefrommod:
                hdulist = fits.open(outmodelfn)
                extnamesdone = []
                for hdu in hdulist:
                    if hdu.name == 'PRIMARY':
                        continue
                    ext, exttype = hdu.name.split('_')
                    if exttype != 'SKY':
                        continue
                    extnamesdone.append(ext)
                hdulist.close()
    # fwhm scrape all the ccds
    fwhms = []
    for name in extnames:
        if name == 'PRIMARY':
            continue
        hdr = fits.getheader(imfn, extname=name)
        if 'FWHM' in hdr:
            fwhms.append(hdr['FWHM'])
    fwhms = numpy.array(fwhms)
    fwhms = fwhms[fwhms > 0]

    # Prepare main CCD for loop
    if extnamelist is not None:
        if verbose:
            s = ("Only running CCD subset: [%s]" %
                 ', '.join(extnamelist))
            print(s)

    if extnamesdone is not None:
        alreadydone = [n for n in extnames if n in extnamesdone]
        extnames = [n for n in extnames if n not in extnamesdone]
        if verbose:
            print('Skipping %s, extension %s; already done.' %
                  (imfn, ' '.join(alreadydone)))
    if extnamelist is not None:
        extnames = [n for n in extnames if n in extnamelist]
    extnames = [n for n in extnames if n != 'PRIMARY']
    if np.isfinite(nproc):
        extnames = extnames[:nproc]

    bigdict = dict(imfn=imfn, ivarfn=ivarfn, dqfn=dqfn,
                   maskdiffuse=maskdiffuse, maskgal=maskgal, verbose=verbose,
                   wcutoff=wcutoff, contmask=contmask, fwhms=fwhms, filt=filt,
                   pixsz=pixsz, brightstars=brightstars,
                   bmask_deblend=bmask_deblend, plot=plot, miniter=miniter,
                   maxiter=maxiter, titer_thresh=titer_thresh,
                   expnum=prihdr['EXPNUM'],outmodel=outmodel,
                   outfn=outfn, outmodelfn=outmodelfn, modsaveonly=modsaveonly,
                   noModsave=noModsave)

    run_fxn(bigdict, extnames, nthreads)

    ### This is the (optional) synthetic injection pipeline ###
    if inject != 0:
        import decam_inject
        imfnI, ivarfnI, dqfnI, injextnames = decam_inject.write_injFiles(imfn, ivarfn,
            dqfn, outfn, inject, injextnamelist, filt, pixsz, wcutoff, verbose, resume,
            date, overwrite, injectfrac=injectfrac)

        bigdict['imfn'] = imfnI
        bigdict['ivarfn'] = ivarfnI
        bigdict['dqfn'] = dqfnI

        run_fxn(bigdict, injextnames, nthreads)

    if profile:
        pr.disable()
        pstats.Stats(pr).sort_stats('cumulative').print_stats(60)
        after = hp.heap()
        leftover = after - before
        print(leftover)
## END of main processing wrapper


def run_fxn(bigdict, extnames, nthreads):
    # Main CCD for loop
    if nthreads > 1:
        import concurrent.futures
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=nthreads)
        iterator = concurrent.futures.as_completed(
            (pool.submit(process_one_ccd, x, bigdict) for x in extnames))
    else:
        iterator = (process_one_ccd(x, bigdict) for x in extnames)

    for res in iterator:
        if nthreads > 1:
            try:
                res = res.result()
            except Exception as e:
                print('Exception running ccd.', e)
                continue
        if res is None:  # no need to process this extension.
            continue
        save_fxn(res, bigdict)
    return


def save_fxn(res, bigdict):
    outmodel = bigdict['outmodel']
    verbose = bigdict['verbose']
    contmask = bigdict['contmask']
    outfn=bigdict['outfn']
    outmodelfn=bigdict['outmodelfn']
    modsaveonly=bigdict['modsaveonly']
    noModsave=bigdict['noModsave']
    cat, modelim, skyim, psf, hdr, msk, prbexport, name = res
    hdr = fits.Header.fromstring(hdr)
    # Data Saving
    if verbose:
        print('Writing %s %s, found %d sources.' % (outfn, name, len(cat)))
        sys.stdout.flush()
    # primary extension includes only header.
    if not modsaveonly:
        fits.append(outfn, numpy.zeros(0), hdr)
    hdupsf = fits.BinTableHDU(psf.serialize())
    hdupsf.name = hdr['EXTNAME'][:-4] + '_PSF'
    hducat = fits.BinTableHDU(cat)
    hducat.name = hdr['EXTNAME'][:-4] + '_CAT'
    if not modsaveonly:
        hdulist = fits.open(outfn, mode='append')
        hdulist.append(hdupsf)  # append the psf field for the ccd
        hdulist.append(hducat)  # append the cat field for the ccd
        hdulist.close(closed=True)
    if outmodel:
        hdr['EXTNAME'] = hdr['EXTNAME'][:-4] + '_MOD'
        # RICE should be significantly better here and supported in
        # mrdfits?, but compression_type=RICE_1 seems to cause
        # quantize_level to be ignored.
        compkw = {'compression_type': 'GZIP_1',
                  'quantize_method': 1, 'quantize_level': -4,
                  'tile_size': modelim.shape}
        modhdulist = fits.open(outmodelfn, mode='append')
        if not noModsave:
            modhdulist.append(fits.CompImageHDU(modelim, hdr, **compkw))
        hdr['EXTNAME'] = hdr['EXTNAME'][:-4] + '_SKY'
        modhdulist.append(fits.CompImageHDU(skyim, hdr, **compkw))
        if msk is not None:
            hdr['EXTNAME'] = hdr['EXTNAME'][:-4] + '_MSK'
            modhdulist.append(fits.CompImageHDU(msk.astype('i4'), hdr,
                                                **compkw))
        if contmask == True:
            prnebnames = ['prN', 'prL', 'prR', 'prE']
            compkw = {'compression_type': 'GZIP_1',
                      'quantize_method': 1, 'quantize_level': 2,
                      'tile_size': (prbexport.shape[0],prbexport.shape[1])}
            for i in range(prbexport.shape[2]):
                hdr['EXTNAME'] = hdr['EXTNAME'][:-4] + '_' + prnebnames[i]
                modhdulist.append(fits.CompImageHDU(prbexport[:,:,i], hdr, **compkw))
        modhdulist.close(closed=True)
    return


def decam_psf(filt, fwhm, pixsz=9):
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
    nlinperpar = 3
    extraparam = numpy.zeros(
        1, dtype=[('convparam', 'f4', 3*nlinperpar+1),
                  ('resparam', 'f4', (nlinperpar, pixsz, pixsz))])
    extraparam['convparam'][0, 0:4] = [convpsffwhm, 1., 0., 1.]
    extraparam['resparam'][0, :, :, :] = 0.
    tpsf.extraparam = extraparam
    tpsf.fitfun = partial(psfmod.fit_linear_static_wing, filter=filt,
                          pixsz=pixsz)
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
    delta = im[:, half+bdy:half:-1] - im[:, half-bdy:half]
    weight = numpy.min([weight[:, half+bdy:half:-1],
                        weight[:, half-bdy:half]], axis=0)

    def objective(par):
        return psfmod.damper(((delta - par[0] - par[1]*xx)*weight)[use], 5)
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
        dq[xl:xr, yl:yr] |= (crowdsource_base.nodeblend_maskbit |
                             crowdsource_base.sharp_maskbit)
    return dq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit DECam frame')

    # Required file name information
    parser.add_argument('base', type=str, help='Base names for files')
    parser.add_argument('date', type=str, help='File name date')
    parser.add_argument('filtf', type=str, help='File name filter')
    parser.add_argument('vers', type=str, help='File name version')
    # Optional file naming directions
    parser.add_argument('--outfn', '-o', type=str,
                        default=None, help='output file name')
    parser.add_argument('--outmodel', '-m', action='store_true',
                        default=False, help='output model file?')

    parser.add_argument('--outdirc', '-d', help='cat output directory',
                        type=str, default=None)
    parser.add_argument('--outdirm', '-e', help='mod output directory',
                        type=str, default=None)
    parser.add_argument('--modsaveonly', action='store_true',
                        help="saves only the model") #not recommended
    parser.add_argument('--donefrommod', action='store_true',
                        help="reads done extensions from mod, not cat")
    parser.add_argument('--noModsave', action='store_true',  #not recommended
                        help="save all model files other than _MOD")
    # Run options
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="prints lots of nice info to cmd line")
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume if file already exists')
    parser.add_argument('--bmask_deblend', '-q', action='store_true',
                        help='turn deblending around bright stars back on')
    parser.add_argument('--maskgal', '-g', action='store_true',
                        help='turn on galaxy masking from leda catalogue')
    parser.add_argument('--no-mask-diffuse', action='store_true',
                        help='turn off nebulosity masking')
    parser.add_argument('--contmask', '-c', action='store_true',
                        help='use continuous nebulosity masking model')
    # Fast short run options
    parser.add_argument('--nthreads', type=int,
                        default=1, help='num of parallel processes (not threads)')
    parser.add_argument('--nccds', type=int,
                        default=numpy.inf, help='run only first nccds ccds')
    parser.add_argument('--ccdlist', nargs='+', default=None,
                        help='limit run to subset of ccds listed')
    # Diagnostic options
    parser.add_argument('--plot_on', type=int,
                        default=0, help='plot psf diagonsitic plots at each titer; \
                        0 off, 1 interactive, 2 save')
    parser.add_argument('--profile', '-p', action='store_true',
                        help='print profiling statistics')
    parser.add_argument('--miniter', type=int,
                        default=4, help='min fit im iterations')
    parser.add_argument('--maxiter', type=int,
                        default=10, help='max fit im iterations')
    parser.add_argument('--titer_thresh', type=int,
                        default=2, help='threshold for deblending increase')
    parser.add_argument('--pixsz', type=int,
                        default=9, help='size of pixelized psf stamp')
    # Experimental options
    parser.add_argument('--wcutoff', type=float,
                        default=0.0, help='cutoff for inverse variances')
    # Calibration run options
    parser.add_argument('--inject', type=int,
                        default=0, help='number of ccd to synthetic inject and rerun \
                        chosen at random from completed ccds; -1 runs all completed ccds \
                        or the full injccdlist')
    parser.add_argument('--injccdlist', nargs='+', default=None,
                        help='limit injection run to subset of ccds listed')
    parser.add_argument('--injectfrac', type=float,
                        default=0.1, help='fraction of sources to reinject')

    args = parser.parse_args()
    process_image(args.base, args.date, args.filtf, args.vers,
                  outfn=args.outfn, outmodel=args.outmodel,
                  outdirc=args.outdirc, outdirm=args.outdirm,
                  verbose=args.verbose, resume=args.resume,
                  bmask_deblend=args.bmask_deblend,
                  maskgal=args.maskgal,
                  maskdiffuse=(not args.no_mask_diffuse),
                  contmask=args.contmask,
                  nproc=args.nccds, extnamelist=args.ccdlist,
                  plot=args.plot_on, profile=args.profile,
                  miniter=args.miniter, maxiter=args.maxiter,
                  titer_thresh=args.titer_thresh, pixsz=args.pixsz,
                  wcutoff=args.wcutoff,
                  nthreads=args.nthreads,
                  inject=args.inject, injextnamelist=args.injccdlist,
                  injectfrac=args.injectfrac,modsaveonly=args.modsaveonly,
                  donefrommod=args.donefrommod,noModsave=args.noModsave)
