from astropy.io import fits
from scipy.interpolate import interp1d
import copy
import numpy as np
from functools import partial
import psf as psfmod
import decam_proc


def scatter_stars(outfn, imfn, ivarfn, dqfn, key, filt, pixsz, wcutoff, verbose, frac=0.1, seed=2021):
    rng = np.random.default_rng(seed)

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
            
    ny, nx = im.shape

    # this requres stars to be "good" and in a reasonable flux range (0 flux to 17th mag)
    maskf = ((flags_stars==1) | (flags_stars==2097153)) & (flux_stars>0) & (flux_stars<158489.3192461114);
    nstars=np.round(0.1*flux_stars[maskf].shape[0]).astype(int)

    flux_samples = sample_stars(flux_stars[maskf],nstars,rng)
    # stay 33 pixels away from edge for injections
    centyl = rng.uniform(33,ny-33,nstars)
    centxl = rng.uniform(33,nx-33,nstars)
    ycenl = centyl.astype(int)
    xcenl = centxl.astype(int)
    diffyl = ycenl - centyl
    diffxl = xcenl - centxl
    mhn = 255 # this is the radius of the model stamp
    mszn = 511 # this is the size of the model stamp

    mock_cat = np.zeros((nstars,5))
    new_flux = np.zeros((ny, nx))
    for i in range(nstars):
        amp = flux_samples[i]
        centy = centyl[i]
        centx = centxl[i]
        ycen = ycenl[i]
        xcen = xcenl[i]
        diffy = diffyl[i]
        diffx = diffxl[i]

        psf = psfmodel.render_model(centy,centx,stampsz=511)
        psf_shift = psfmod.shift(psf,(-diffy,-diffx));
        draw = rng.poisson(lam=amp*gain*psf_shift)/gain

        new_flux[np.clip(ycen-mhn,a_min=0,a_max=None):np.clip(ycen+mhn+1,a_min=None,a_max=ny),
           np.clip(xcen-mhn,a_min=0,a_max=None):np.clip(xcen+mhn+1,a_min=None,a_max=nx)] += draw[
        np.clip(mhn-ycen,a_min=0,a_max=None):np.clip(ny-ycen+mhn,a_min=None,a_max=mszn),
             np.clip(mhn-xcen,a_min=0,a_max=None):np.clip(nx-xcen+mhn,a_min=None,a_max=mszn)]

        mock_cat[i,:] = [centx, centy, np.sum(draw), np.sum(np.multiply(draw,psf_shift))/np.sum(np.square(psf_shift)), amp]

    im += new_flux
    wt = (wt**(-1) + np.divide(new_flux,gain))**(-1)

    # save our injections
    ## Eddie thinks we should compare compressing with the
    ## same or different seed
    imfnI = injectRename(imfn)
    ivarfnI = injectRename(ivarfn)
    dqfnI = injectRename(dqfn)

    hdr = fits.getheader(dqfn, extname=key)
    hdr['EXTNAME'] = hdr['EXTNAME'] + 'I'
    compkw = {'quantize_method': 1,
              'quantize_level': 4,
             }
    f = fits.open(dqfnI, mode='append')
    f.append(fits.CompImageHDU(dq, hdr, **compkw))
    f.close(closed=True)

    hdr = fits.getheader(ivarfn, extname=key)
    hdr['EXTNAME'] = hdr['EXTNAME'] + 'I'
    compkw = {'quantize_method': 1,
              'quantize_level': 4,
              'dither_seed': hdr["ZDITHER0"],
             }
    f = fits.open(ivarfnI, mode='append')
    f.append(fits.CompImageHDU(wt, hdr, **compkw))
    f.close(closed=True)

    hdr = fits.getheader(imfn, extname=key)
    hdr['EXTNAME'] = hdr['EXTNAME'] + 'I'
    compkw = {'quantize_method': 1,
              'quantize_level': 4,
              'dither_seed': hdr["ZDITHER0"],
             }
    f = fits.open(imfnI, mode='append')
    f.append(fits.CompImageHDU(im, hdr, **compkw))
    f.close(closed=True)

    #mock catalogue export
    stars = OrderedDict([('centx', mock_cat[:,0]), ('centy', mock_cat[:,1]), ('flux', mock_cat[:,2]),
                     ('psfwt_flux', mock_cat[:,3]), ('amp', mock_cat[:,4])])
    dtypenames = list(stars.keys())
    dtypeformats = [stars[n].dtype for n in dtypenames]
    dtype = dict(names=dtypenames, formats=dtypeformats)
    stars = numpy.fromiter(zip(*stars.values()),
                           dtype=dtype, count=len(stars['centx']))

    hducat = fits.BinTableHDU(cat)
    hducat.name = hdr['EXTNAME'] + '_CAT'
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
    return inv_ecdf(sampler)


def load_psfmodel(outfn, key, filter, pixsz=9):
    f = fits.open(outfn)
    psfmodel = psfmod.linear_static_wing_from_record(f[key+"_PSF"].data[0],filter=filter)
    f.close()
    psfmodel.fitfun = partial(psfmod.fit_linear_static_wing, filter=filter, pixsz=pixsz)
    return psfmodel


def injectRename(fname):
    spltname = fname.split("/")
    spltname[3] = "decapsi"
    fname = "/".join(spltname)
    return fname[:-7]+"I.fits.fz"
