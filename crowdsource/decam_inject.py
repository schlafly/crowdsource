from astropy.io import fits
from scipy.interpolate import interp1d
import copy
import numpy as np
from functools import partial
import crowdsource.psf as psfmod
import crowdsource.decam_proc as decam_proc
from collections import OrderedDict
import os

def write_injFiles(imfn, ivarfn, dqfn, outfn, inject, injextnamelist, filt, pixsz,
                   wcutoff, verbose, resume, date, overwrite, injectfrac=0.1,extadd=None):
    # Updated the completed ccds
    hdulist = fits.open(outfn)
    extnamesdone = []
    injnamescat = []
    for hdu in hdulist:
        if hdu.name == 'PRIMARY':
            continue
        extfull = hdu.name.split('_')
        ext = "_".join(extfull[:-1])
        if extfull[-1] != 'CAT':
            continue
        if "I" in ext:
            injnamescat.append(ext)
        else:
            extnamesdone.append(ext)
    hdulist.close()

    # Prepare injection CCD for loop ext list
    if extnamesdone is not None:
        injextnames = [n for n in extnamesdone]
    else:
        raise ValueError('No CCDs are done. Please fit at least one CCD before injection test.')
    if injextnamelist is not None:
        injextnames = [n for n in injextnames if n in injextnamelist]
    injextnames = [n for n in injextnames if n != 'PRIMARY']
    if inject != -1:
        rng = np.random.default_rng(int(date))
        injextnames = rng.choice(injextnames, inject, replace=False)
    if verbose:
        s = 'Injecting sources into [%s]' %', '.join(injextnames)
        print(s)

    # create files with injected sources in the decapsi directory
    ## this might need to be more robust if we port to a different cluster/user
    imfnI = injectRename(imfn)
    ivarfnI = injectRename(ivarfn)
    dqfnI = injectRename(dqfn)

    # intialize the injected images
    prihdr = fits.getheader(imfn, extname='PRIMARY')
    if not resume or not os.path.exists(imfnI):
        fits.writeto(imfnI, None, prihdr, overwrite=overwrite)

    prihdr = fits.getheader(ivarfn, extname='PRIMARY')
    if not resume or not os.path.exists(ivarfnI):
        fits.writeto(ivarfnI, None, prihdr, overwrite=overwrite)

    prihdr = fits.getheader(dqfn, extname='PRIMARY')
    if not resume or not os.path.exists(dqfnI):
        fits.writeto(dqfnI, None, prihdr, overwrite=overwrite)

    import warnings
    with warnings.catch_warnings(record=True) as wlist:
        hdulist = fits.open(dqfnI)
        injnamesdone = []
        for hdu in hdulist:
            if hdu.name == 'PRIMARY':
                continue
            injnamesdone.append(hdu.name)
        hdulist.close()
    # suppress endless nonstandard keyword warnings on read
    for warning in wlist:
        if 'following header keyword' in str(warning.message):
            continue
        else:
            print(warning)

    injextnamesI = [i+"I" for i in injextnames]
    injextnamesI = [i for i in injextnamesI if i not in injnamescat]

    injextnames = [i for i in injextnames if i not in injnamesdone]

    rng = np.random.default_rng(int(date))
    for key in injextnames:
        scatter_stars(outfn, imfn, ivarfn, dqfn, key, filt, pixsz, wcutoff, verbose, rng, injectfrac=injectfrac,extadd=extadd)

    return imfnI, ivarfnI, dqfnI, injextnamesI

#seed on date here too
def scatter_stars(outfn, imfn, ivarfn, dqfn, key, filt, pixsz, wcutoff, verbose, rng, injectfrac=0.1, extadd=None):
    keyadd = "I"
    if extadd is not None:
        keyadd+=("_"+str(extadd).zfill(3))
    ## imports
    hdr = fits.getheader(outfn,key+"_HDR")
    gain = hdr['GAINCRWD']

    f = fits.open(outfn)
    table = f[key+"_CAT"].data
    flux_stars = table["flux"]
    flags_stars = table["flags"]
    f.close()

    psfmodel = load_psfmodel(outfn, key, filt[0], pixsz=pixsz)

    import warnings
    with warnings.catch_warnings(record=True) as wlist:
        im = fits.getdata(imfn, extname=key).copy()
        wt = fits.getdata(ivarfn, extname=key).copy()
        dq = fits.getdata(dqfn, extname=key).copy()
    # suppress endless nonstandard keyword warnings on read
    for warning in wlist:
        if 'following header keyword' in str(warning.message):
            continue
        else:
            print(warning)

    nx, ny = im.shape

    # this requres stars to be "good" and in a reasonable flux range (0 flux to 17th mag)
    badflags = 2**1+2**3+2**4+2**5+2**7+2**20+2**23+2**24
    maskf = ((flags_stars & badflags) == 0) & (flux_stars>0) & (flux_stars<158489.3192461114);
    nstars_tot=flux_stars.shape[0]
    nstarg=flux_stars[maskf].shape[0]
    nstars=np.round(injectfrac*nstarg).astype(int)
    if nstarg < 2:
        if verbose:
            print("skipping injection, %d good stars out of %d total stars" % (nstarg, nstars_tot))
        return
    if nstars == 0:
        if verbose:
            print("skipping injection, 0 stars would have been injected")
            print("for the record, based on %d good stars out of %d total stars" % (nstarg, nstars_tot))
        return

    flux_samples = sample_stars(flux_stars[maskf],nstars,rng)
    nstars = flux_samples.shape[0]
    # stay 33 pixels away from edge for injections
    centxl = rng.uniform(33,nx-33,nstars)
    centyl = rng.uniform(33,ny-33,nstars)
    xcenl = centxl.astype(int)
    ycenl = centyl.astype(int)
    mhn = 255 # this is the radius of the model stamp
    mszn = 511 # this is the size of the model stamp
    mock_cat = np.zeros((nstars,6))
    new_flux = np.zeros((nx, ny))
    for i in range(nstars):
        amp = flux_samples[i]
        centx = centxl[i]
        centy = centyl[i]
        xcen = xcenl[i]
        ycen = ycenl[i]

        psf_shift = psfmodel(centx,centy,stampsz=511)
        draw = rng.poisson(lam=amp*gain*psf_shift)/gain

        new_flux[np.clip(xcen-mhn,a_min=0,a_max=None):np.clip(xcen+mhn+1,a_min=None,a_max=nx),
           np.clip(ycen-mhn,a_min=0,a_max=None):np.clip(ycen+mhn+1,a_min=None,a_max=ny)] += draw[
        np.clip(mhn-xcen,a_min=0,a_max=None):np.clip(nx-xcen+mhn,a_min=None,a_max=mszn),
             np.clip(mhn-ycen,a_min=0,a_max=None):np.clip(ny-ycen+mhn,a_min=None,a_max=mszn)]

        mock_cat[i,:] = [centx, centy, np.sum(psfmod.central_stamp(draw,censize=59)), np.sum(draw), np.sum(np.multiply(draw,psf_shift))/np.sum(np.square(psf_shift)), amp]

    im += new_flux
    wt = (1./(wt + (wt == 0) * 1e14) + np.divide(new_flux,gain))**(-1)

    # save our injections
    ## Eddie thinks we should compare compressing with the
    ## same or different seed
    imfnI = injectRename(imfn)
    ivarfnI = injectRename(ivarfn)
    dqfnI = injectRename(dqfn)

    with warnings.catch_warnings(record=True) as wlist:
        hdr = fits.getheader(dqfn, extname=key)
        hdr['EXTNAME'] = hdr['EXTNAME'] + keyadd
        compkw = {'quantize_method': 1,
                  'quantize_level': 4,
                 }
        f = fits.open(dqfnI, mode='append')
        f.append(fits.CompImageHDU(dq, hdr, **compkw))
        f.close(closed=True)

        hdr = fits.getheader(ivarfn, extname=key)
        new_seed = hdr["ZDITHER0"]+1
        if new_seed > 10000:
            new_seed -= 10000
        hdr['EXTNAME'] = hdr['EXTNAME'] + keyadd
        compkw = {'quantize_method': 1,
                  'quantize_level': 4,
                  'dither_seed': new_seed,
                 }
        f = fits.open(ivarfnI, mode='append')
        f.append(fits.CompImageHDU(wt, hdr, **compkw))
        f.close(closed=True)

        hdr = fits.getheader(imfn, extname=key)
        new_seed = hdr["ZDITHER0"]+1
        if new_seed > 10000:
            new_seed -= 10000
        hdr['EXTNAME'] = hdr['EXTNAME'] + keyadd
        compkw = {'quantize_method': 1,
                  'quantize_level': 4,
                  'dither_seed': new_seed,
                 }
        f = fits.open(imfnI, mode='append')
        f.append(fits.CompImageHDU(im, hdr, **compkw))
        f.close(closed=True)
    for warning in wlist:
        if 'following header keyword' in str(warning.message):
            continue
        else:
            print(warning)

    #mock catalogue export
    stars = OrderedDict([('x', mock_cat[:,0]), ('y', mock_cat[:,1]), ('flux', mock_cat[:,2]),
                     ('fluxfull', mock_cat[:,3]), ('psfwt_flux', mock_cat[:,4]), ('amp', mock_cat[:,5])])
    dtypenames = list(stars.keys())
    dtypeformats = [stars[n].dtype for n in dtypenames]
    dtype = dict(names=dtypenames, formats=dtypeformats)
    cat = np.fromiter(zip(*stars.values()),
                           dtype=dtype, count=len(stars['x']))

    hducat = fits.BinTableHDU(cat)
    hducat.name = hdr['EXTNAME'] + '_MCK'
    hdulist = fits.open(outfn, mode='append')
    hdulist.append(hducat)  # append the cat field for the ccd
    hdulist.close(closed=True)
    return


def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys


def sample_stars(flux_list, nstars, rng):
    fx, fy = ecdf(flux_list)
    inv_ecdf = interp1d(fy, fx, fill_value="extrapolate")
    sampler = rng.uniform(0,1,nstars)
    sflux = inv_ecdf(sampler)
    return sflux[sflux>0]


def load_psfmodel(outfn, key, filter, pixsz=9):
    f = fits.open(outfn)
    psfmodel = psfmod.linear_static_wing_from_record(f[key+"_PSF"].data[0],filter=filter)
    f.close()
    psfmodel.fitfun = partial(psfmod.fit_linear_static_wing, filter=filter, pixsz=pixsz)
    return psfmodel


def injectRename(fname):
    spltname = fname.split("/")
    spltname[-2] = "decapsi"
    fname = "/".join(spltname)
    return fname[:-7]+"I.fits.fz"
