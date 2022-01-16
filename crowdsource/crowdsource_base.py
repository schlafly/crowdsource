"""Crowded field photometry pipeline.

This module fits positions, fluxes, PSFs, and sky backgrounds of images.
Intended usage is:
>>> x, y, flux, model, psf = fit_im(im, psf_initial, weight=wim,
                                    psfderiv=numpy.gradient(-psf),
                                    nskyx=3, nskyy=3, refit_psf=True)
which returns the best fit positions (x, y), fluxes (flux), model image
(model), and improved psf (psf) to the image im, with an initial psf guess
(psf_initial), an inverse-variance image wim, and a variable sky background.

See mosaic.py for how to use this on a large image that is too big to be fit
entirely simultaneously.
"""

import sys
import os
import numpy
import pdb
import crowdsource.psf as psfmod
import scipy.ndimage.filters as filters
from collections import OrderedDict

nodeblend_maskbit = 2**30
sharp_maskbit = 2**31


def shift(im, offset, **kw):
    """Wrapper for scipy.ndimage.interpolation.shift"""
    from scipy.ndimage.interpolation import shift
    if 'order' not in kw:
        kw['order'] = 4
        # 1" Gaussian: 60 umag; 0.75": 0.4 mmag; 0.5": 4 mmag
        # order=3 roughly 5x worse.
    if 'mode' not in kw:
        kw['mode'] = 'nearest'
    if 'output' not in kw:
        kw['output'] = im.dtype
    return shift(im, offset, **kw)


def sim_image(nx, ny, nstar, psf, noise, nskyx=3, nskyy=3, stampsz=19):
    im = numpy.random.randn(nx, ny).astype('f4')*noise
    stampszo2 = stampsz // 2
    im = numpy.pad(im, [stampszo2, stampszo2], constant_values=-1e6,
                   mode='constant')
    x = numpy.random.rand(nstar).astype('f4')*(nx-1)
    y = numpy.random.rand(nstar).astype('f4')*(ny-1)
    flux = 1./numpy.random.power(1.0, nstar)
    for i in range(nstar):
        stamp = psf(x[i], y[i], stampsz=stampsz)
        xl = numpy.round(x[i]).astype('i4')
        yl = numpy.round(y[i]).astype('i4')
        im[xl:xl+stampsz, yl:yl+stampsz] += stamp*flux[i]
    if (nskyx != 0) or (nskyy != 0):
        im += sky_model(100*numpy.random.rand(nskyx, nskyy).astype('f4'),
                        im.shape[0], im.shape[1])
    ret = im[stampszo2:-stampszo2, stampszo2:-stampszo2], x, y, flux
    return ret


def significance_image(im, model, isig, psf, sz=19):
    """Significance of a PSF at each point, without local background fit."""
    # assume, for the moment, the image has already been sky-subtracted
    def convolve(im, kernel):
        from scipy.signal import fftconvolve
        return fftconvolve(im, kernel[::-1, ::-1], mode='same')
        # identical to 1e-8 or so
        # from scipy.ndimage.filters import convolve
        # return convolve(im, kernel[::-1, ::-1], mode='nearest')
    psfstamp = psfmod.central_stamp(psf, sz).copy()
    sigim = convolve(im*isig**2., psfstamp)
    varim = convolve(isig**2., psfstamp**2.)
    modim = convolve(model*isig**2., psfstamp)
    varim[varim <= 1e-14] = 0.  # numerical noise starts to set in around here.
    ivarim = 1./(varim + (varim == 0) * 1e14)
    return sigim*numpy.sqrt(ivarim), modim*numpy.sqrt(ivarim)


def significance_image_lbs(im, model, isig, psf, sz=19):
    """Give significance of PSF at each point, with local background fits."""

    def convolve(im, kernel):
        from scipy.signal import fftconvolve
        return fftconvolve(im, kernel[::-1, ::-1], mode='same')

    def convolve_flat(im, sz):
        from scipy.ndimage.filters import convolve
        filt = numpy.ones(sz, dtype='f4')
        c1 = convolve(im, filt.reshape(1, -1), mode='constant', origin=0)
        return convolve(c1, filt.reshape(-1, 1), mode='constant', origin=0)

    # we need: * convolution of ivar with P^2
    #          * convolution of ivar with flat
    #          * convolution of ivar with P
    #          * convolution of b*ivar with P
    #          * convolution of b*ivar with flat
    ivar = isig**2.
    if sz is None:
        psfstamp = psfmod.central_stamp(psf).copy()
    else:
        psfstamp = psfmod.central_stamp(psf, censize=sz).copy()
    ivarp2 = convolve(ivar, psfstamp**2.)
    ivarp2[ivarp2 < 0] = 0.
    ivarimsimple = 1./(ivarp2 + (ivarp2 == 0) * 1e12)
    ivarf = convolve_flat(ivar, psfstamp.shape[0])
    ivarp = convolve(ivar, psfstamp)
    bivarp = convolve(im*ivar, psfstamp)
    bivarf = convolve_flat(im*ivar, psfstamp.shape[0])
    atcinvadet = ivarp2*ivarf-ivarp**2.
    atcinvadet[atcinvadet <= 0] = 1.e-12
    ivarf[ivarf <= 0] = 1.e-12
    fluxest = (bivarp*ivarf-ivarp*bivarf)/atcinvadet
    fluxisig = numpy.sqrt(atcinvadet/ivarf)
    fluxsig = fluxest*fluxisig
    modim = convolve(model*ivar, psfstamp)
    return fluxsig, modim*numpy.sqrt(ivarimsimple)


def peakfind(im, model, isig, dq, psf, keepsat=False, threshold=5,
             blendthreshold=0.3, psfvalsharpcutfac=0.7, psfsharpsat=0.7):
    psfstamp = psf(int(im.shape[0]/2.), int(im.shape[1]/2.), deriv=False,
                   stampsz=59)
    sigim, modelsigim = significance_image(im, model, isig, psfstamp,
                                           sz=59)
    sig_max = filters.maximum_filter(sigim, 3)
    x, y = numpy.nonzero((sig_max == sigim) & (sigim > threshold) &
                         (keepsat | (isig > 0)))
    fluxratio = im[x, y]/numpy.clip(model[x, y], 0.01, numpy.inf)
    sigratio = (im[x, y]*isig[x, y])/numpy.clip(modelsigim[x, y], 0.01,
                                                numpy.inf)
    sigratio2 = sigim[x, y]/numpy.clip(modelsigim[x, y], 0.01, numpy.inf)
    keepsatcensrc = keepsat & (isig[x, y] == 0)
    m = ((isig[x, y] > 0) | keepsatcensrc)  # ~saturated, or saturated & keep
    if dq is not None and numpy.any(dq[x, y] & nodeblend_maskbit):
        nodeblend = (dq[x, y] & nodeblend_maskbit) != 0
        blendthreshold = numpy.ones_like(x)*blendthreshold
        blendthreshold[nodeblend] = 100
    if dq is not None and numpy.any(dq[x, y] & sharp_maskbit):
        sharp = (dq[x, y] & sharp_maskbit) != 0
        msharp = ~sharp | psfvalsharpcut(
            x, y, sigim, isig, psfstamp, psfvalsharpcutfac=psfvalsharpcutfac,
            psfsharpsat=psfsharpsat)
        # keep if not nebulous region or sharp peak.
        m = m & msharp

    m = m & ((sigratio2 > blendthreshold*2) |
             ((fluxratio > blendthreshold) & (sigratio > blendthreshold/4.) &
              (sigratio2 > blendthreshold)))

    return x[m], y[m]


def psfvalsharpcut(x, y, sigim, isig, psf, psfvalsharpcutfac=0.7,
                   psfsharpsat=0.7):
    xl = numpy.clip(x-1, 0, sigim.shape[0]-1)
    xr = numpy.clip(x+1, 0, sigim.shape[0]-1)
    yl = numpy.clip(y-1, 0, sigim.shape[1]-1)
    yr = numpy.clip(y+1, 0, sigim.shape[1]-1)
    # sigim[x, y] should always be >0 from threshold cut.
    psfval1 = 1-(sigim[xl, y]+sigim[xr, y])/(2*sigim[x, y])
    psfval2 = 1-(sigim[x, yl]+sigim[x, yr])/(2*sigim[x, y])
    psfval3 = 1-(sigim[xl, yl]+sigim[xr, yr])/(2*sigim[x, y])
    psfval4 = 1-(sigim[xl, yr]+sigim[xr, yl])/(2*sigim[x, y])
    # in nebulous region, there should be a peak of these around the PSF
    # size, plus a bunch of diffuse things (psfval ~ 0).
    from scipy.signal import fftconvolve
    pp = fftconvolve(psf, psf[::-1, ::-1], mode='same')
    half = psf.shape[0] // 2
    ppcen = pp[half, half]
    psfval1pp = 1-(pp[half-1, half]+pp[half+1, half])/(2*ppcen)
    psfval2pp = 1-(pp[half, half-1]+pp[half, half+1])/(2*ppcen)
    psfval3pp = 1-(pp[half-1, half-1]+pp[half+1, half+1])/(2*ppcen)
    psfval4pp = 1-(pp[half-1, half+1]+pp[half+1, half-1])/(2*ppcen)
    fac = psfvalsharpcutfac*(1-psfsharpsat*(isig[x, y] == 0))
    # more forgiving if center is masked.
    res = ((psfval1 > psfval1pp*fac) & (psfval2 > psfval2pp*fac) &
           (psfval3 > psfval3pp*fac) & (psfval4 > psfval4pp*fac))
    return res


def build_model(x, y, flux, nx, ny, psf=None, psflist=None, psfderiv=False):
    if psf is None and psflist is None:
        raise ValueError('One of psf and psflist must be set')
    if psf is not None and psflist is not None:
        raise ValueError('Only one of psf and psflist must be set')
    if psflist is None:
        stampsz = 59
        psflist = build_psf_list(x, y, psf, stampsz, psfderiv=psfderiv)
        sz = numpy.ones(len(x), dtype='i4')*stampsz
    else:
        sz = numpy.array([tpsf[0].shape[-1] for tpsf in psflist[0]])
        if len(sz) > 0:
            stampsz = numpy.max(sz)
        else:
            stampsz = 59

    stampszo2 = stampsz//2
    im = numpy.zeros((nx, ny), dtype='f4')
    im = numpy.pad(im, [stampszo2, stampszo2], constant_values=0.,
                   mode='constant')
    xp = numpy.round(x).astype('i4')
    yp = numpy.round(y).astype('i4')
    # _subtract_ stampszo2 to move from the center of the PSF to the edge
    # of the stamp.
    # _add_ it back to move from the original image to the padded image.
    xe = xp - sz//2 + stampszo2
    ye = yp - sz//2 + stampszo2
    repeat = 3 if psfderiv else 1
    for i in range(len(x)):
        for j in range(repeat):
            im[xe[i]:xe[i]+sz[i], ye[i]:ye[i]+sz[i]] += (
                psflist[j][i][:, :]*flux[i*repeat+j])
    im = im[stampszo2:-stampszo2, stampszo2:-stampszo2]
    return im


def build_psf_list(x, y, psf, sz, psfderiv=True):
    """Make a list of PSFs of the right size, hopefully efficiently."""

    psflist = {}
    for tsz in numpy.unique(sz):
        m = sz == tsz
        res = psf(x[m], y[m], stampsz=tsz, deriv=psfderiv)
        if not psfderiv:
            res = [res]
        psflist[tsz] = res
    counts = {tsz: 0 for tsz in numpy.unique(sz)}
    out = [[] for i in range(3 if psfderiv else 1)]
    for i in range(len(x)):
        for j in range(len(out)):
            out[j].append(psflist[sz[i]][j][counts[sz[i]]])
        counts[sz[i]] += 1
    return out


def in_padded_region(flatcoord, imshape, pad):
    coord = numpy.unravel_index(flatcoord, imshape)
    m = numpy.zeros(len(flatcoord), dtype='bool')
    for c, length in zip(coord, imshape):
        m |= (c < pad) | (c >= length - pad)
    return m


def fit_once(im, x, y, psfs, weight=None,
             psfderiv=False, nskyx=0, nskyy=0,
             guess=None):
    """Fit fluxes for psfs at x & y in image im.

    Args:
        im (ndarray[NX, NY] float): image to fit
        x (ndarray[NS] float): x coord
        y (ndarray[NS] float): y coord
        psf (ndarray[sz, sz] float): psf stamp
        weight (ndarray[NX, NY] float): weight for image
        psfderiv (tuple(ndarray[sz, sz] float)): x, y derivatives of psf image
        nskyx (int): number of sky pixels in x direction (0 or >= 3)
        nskyy (int): number of sky pixels in y direction (0 or >= 3)

    Returns:
        tuple(flux, model, sky)
        flux: output of optimization routine; needs to be refined
        model (ndarray[NX, NY]): best fit model image
        sky (ndarray(NX, NY]): best fit model sky
    """
    # sparse matrix, with rows at first equal to the fluxes at each peak
    # later add in the derivatives at each peak
    sz = numpy.array([tpsf[0].shape[-1] for tpsf in psfs[0]])
    if len(sz) > 0:
        stampsz = numpy.max(sz)
    else:
        stampsz = 19
    stampszo2 = stampsz // 2
    szo2 = sz // 2
    nx, ny = im.shape
    pad = stampszo2 + 1
    im = numpy.pad(im, [pad, pad], constant_values=0.,
                   mode='constant')
    if weight is None:
        weight = numpy.ones_like(im)
    weight = numpy.pad(weight, [pad, pad], constant_values=0.,
                       mode='constant')
    weight[weight == 0.] = 1.e-20
    pix = numpy.arange(stampsz*stampsz, dtype='i4').reshape(stampsz, stampsz)
    # convention: x is the first index, y is the second
    # sorry.
    xpix = pix // stampsz
    ypix = pix % stampsz
    xp = numpy.round(x).astype('i4')
    yp = numpy.round(y).astype('i4')
    # _subtract_ stampszo2 to move from the center of the PSF to the edge
    # of the stamp.
    # _add_ pad back to move from the original image to the padded image.
    xe = xp - stampszo2 + pad
    ye = yp - stampszo2 + pad
    repeat = 1 if not psfderiv else 3
    nskypar = nskyx * nskyy
    npixim = im.shape[0]*im.shape[1]
    xloc = numpy.zeros(repeat*numpy.sum(sz*sz).astype('i4') +
                       nskypar*npixim, dtype='i4')
    # yloc = numpy.zeros(len(xloc), dtype='i4')
    # no longer need yloc; csc entries are built directly.
    values = numpy.zeros(len(xloc), dtype='f4')
    colnorm = numpy.zeros(len(x)*repeat+nskypar, dtype='f4')
    first = 0
    for i in range(len(xe)):
        f = stampszo2-szo2[i]
        l = stampsz - f
        wt = weight[xe[i]:xe[i]+stampsz, ye[i]:ye[i]+stampsz][f:l, f:l]
        for j in range(repeat):
            xloc[first:first+sz[i]**2] = (
                numpy.ravel_multi_index(((xe[i]+xpix[f:l, f:l]),
                                         (ye[i]+ypix[f:l, f:l])),
                                        im.shape)).reshape(-1)
            # yloc[first:first+sz[i]**2] = i*repeat+j
            values[first:first+sz[i]**2] = (
                (psfs[j][i][:, :]*wt).reshape(-1))
            colnorm[i*repeat+j] = numpy.sqrt(
                numpy.sum(values[first:first+sz[i]**2]**2.))
            colnorm[i*repeat+j] += (colnorm[i*repeat+j] == 0)
            values[first:first+sz[i]**2] /= colnorm[i*repeat+j]
            first += sz[i]**2

    if nskypar != 0:
        sxloc, syloc, svalues = sky_parameters(nx+pad*2, ny+pad*2,
                                               nskyx, nskyy, weight)
        startidx = len(x)*repeat
        nskypix = len(sxloc[0])
        for i in range(len(sxloc)):
            xloc[first:first+nskypix] = sxloc[i]
            # yloc[first:first+nskypix] = startidx+syloc[i]
            colnorm[startidx+i] = numpy.sqrt(numpy.sum(svalues[i]**2.))
            colnorm[startidx+i] += (colnorm[startidx+i] == 0.)
            values[first:first+nskypix] = svalues[i] / colnorm[startidx+i]
            first += nskypix
    shape = (im.shape[0]*im.shape[1], len(x)*repeat+nskypar)

    from scipy import sparse
    csc_indptr = numpy.cumsum([sz[i]**2 for i in range(len(x))
                               for j in range(repeat)])
    csc_indptr = numpy.concatenate([[0], csc_indptr])
    if nskypar != 0:
        csc_indptr = numpy.concatenate([csc_indptr, [
            csc_indptr[-1] + i*nskypix for i in range(1, nskypar+1)]])
    mat = sparse.csc_matrix((values, xloc, csc_indptr), shape=shape,
                            dtype='f4')
    if guess is not None:
        # guess is a guess for the fluxes and sky; no derivatives.
        guessvec = numpy.zeros(len(xe)*repeat+nskypar, dtype='f4')
        guessvec[0:len(xe)*repeat:repeat] = guess[0:len(xe)]
        if nskypar > 0:
            guessvec[-nskypar:] = guess[-nskypar:]
        guessvec *= colnorm
    else:
        guessvec = None
    flux = lsqr_cp(mat, (im*weight).ravel(), atol=1.e-4, btol=1.e-4,
                   guess=guessvec)
    model = mat.dot(flux[0]).reshape(*im.shape)
    flux[0][:] = flux[0][:] / colnorm
    im = im[pad:-pad, pad:-pad]
    model = model[pad:-pad, pad:-pad]
    weight = weight[pad:-pad, pad:-pad]
    if nskypar != 0:
        sky = sky_model(flux[0][-nskypar:].reshape(nskyx, nskyy),
                        nx+pad*2, ny+pad*2)
        sky = sky[pad:-pad, pad:-pad]
    else:
        sky = model * 0
    model = model / (weight + (weight == 0))
    res = (flux, model, sky)
    return res


def unpack_fitpar(guess, nsource, psfderiv):
    """Extract fluxes and sky parameters from fit parameter vector."""
    repeat = 3 if psfderiv else 1
    return guess[0:nsource*repeat:repeat], guess[nsource*repeat:]


def lsqr_cp(aa, bb, guess=None, **kw):
    # implement two speed-ups:
    # 1. "column preconditioning": make sure each column of aa has the same
    #    norm
    # 2. allow guesses

    # column preconditioning is important (substantial speedup), and has
    # been implemented directly in fit_once.

    # allow guesses: solving Ax = b is the same as solving A(x-x*) = b-Ax*.
    # => A(dx) = b-Ax*.  So we can solve for dx instead, then return dx+x*.
    # This improves speed if we reduce the tolerance.
    from scipy.sparse import linalg

    if guess is not None:
        bb2 = bb - aa.dot(guess)
        if 'btol' in kw:
            fac = numpy.sum(bb**2.)**(0.5)/numpy.sum(bb2**2.)**0.5
            kw['btol'] = kw['btol']*numpy.clip(fac, 0.1, 10.)
    else:
        bb2 = bb.copy()

    normbb = numpy.sum(bb2**2.)
    bb2 /= normbb**(0.5)
    par = linalg.lsqr(aa, bb2, **kw)
    # for some reason, everything ends up as double precision after this
    # or lsmr; lsqr seems to be better
    # par[0][:] *= norm**(-0.5)*normbb**(0.5)
    par[0][:] *= normbb**0.5
    if guess is not None:
        par[0][:] += guess
    par = list(par)
    par[0] = par[0].astype('f4')
    par[9] = par[9].astype('f4')
    return par


def compute_centroids(x, y, psflist, flux, im, resid, weight,
                      derivcentroids=False, centroidsize=19):
    # define c = integral(x * I * P * W) / integral(I * P * W)
    # x = x/y coordinate, I = isolated stamp, P = PSF model, W = weight
    # Assuming I ~ P(x-y) for some small offset y and expanding,
    # integrating by parts gives:
    # y = 2 / integral(P*P*W) * integral(x*(I-P)*W)
    # that is the offset we want.

    # we want to compute the centroids on the image after the other sources
    # have been subtracted off.
    # we construct this image by taking the residual image, and then
    # star-by-star adding the model back.
    psfs = [numpy.zeros((len(x), centroidsize, centroidsize), dtype='f4')
            for i in range(len(psflist))]
    for j in range(len(psflist)):
        for i in range(len(x)):
            psfs[j][i, :, :] = psfmod.central_stamp(psflist[j][i],
                                                    censize=centroidsize)
    stampsz = psfs[0].shape[-1]
    stampszo2 = (stampsz-1)//2
    dx = numpy.arange(stampsz, dtype='i4')-stampszo2
    dx = dx.reshape(-1, 1)
    dy = dx.copy().reshape(1, -1)
    xp = numpy.round(x).astype('i4')
    yp = numpy.round(y).astype('i4')
    # subtracting to get to the edge of the stamp, adding back to deal with
    # the padded image.
    xe = xp - stampszo2 + stampszo2
    ye = yp - stampszo2 + stampszo2
    resid = numpy.pad(resid, [stampszo2, stampszo2], constant_values=0.,
                      mode='constant')
    weight = numpy.pad(weight, [stampszo2, stampszo2], constant_values=0.,
                       mode='constant')
    im = numpy.pad(im, [stampszo2, stampszo2], constant_values=0.,
                   mode='constant')
    repeat = len(psflist)
    residst = numpy.array([resid[xe0:xe0+stampsz, ye0:ye0+stampsz]
                           for (xe0, ye0) in zip(xe, ye)])
    weightst = numpy.array([weight[xe0:xe0+stampsz, ye0:ye0+stampsz]
                            for (xe0, ye0) in zip(xe, ye)])
    psfst = psfs[0] * flux[:len(x)*repeat:repeat].reshape(-1, 1, 1)
    imst = numpy.array([im[xe0:xe0+stampsz, ye0:ye0+stampsz]
                        for (xe0, ye0) in zip(xe, ye)])
    if len(x) == 0:
        weightst = psfs[0].copy()
        residst = psfs[0].copy()
        imst = psfs[0].copy()
    modelst = psfst.copy()
    if len(psflist) > 1:
        modelst += psfs[1]*flux[1:len(x)*repeat:repeat].reshape(-1, 1, 1)
        modelst += psfs[2]*flux[2:len(x)*repeat:repeat].reshape(-1, 1, 1)
    cen = []
    ppw = numpy.sum(modelst*modelst*weightst, axis=(1, 2))
    pp = numpy.sum(modelst*modelst, axis=(1, 2))
    for dc in (dx, dy):
        xrpw = numpy.sum(dc[None, :, :]*residst*modelst*weightst, axis=(1, 2))
        xmmpm = numpy.sum(dc[None, :, :]*(modelst-psfst)*modelst, axis=(1, 2))
        cen.append(2*xrpw/(ppw + (ppw == 0.))*(ppw != 0.) +
                   2*xmmpm/(pp + (pp == 0.))*(pp != 0.))
    xcen, ycen = cen
    norm = numpy.sum(modelst, axis=(1, 2))
    norm = norm + (norm == 0)
    psfqf = numpy.sum(modelst*(weightst > 0), axis=(1, 2)) / norm
    # how should we really be doing this?  derivcentroids is the first order
    # approximation to the right thing.  the centroid computation that I do
    # otherwise should be unbiased but noisier than optimal for significantly
    # offset peaks.  Vakili, Hogg (2016) say that I should convolve with the
    # PSF and interpolate to the brightest point with some polynomial.  I
    # expected this to be slow (convolving thousands of stamps individually
    # with the PSF each iteration), but the spread_model code worked pretty
    # well, so this is probably a worthwhile thing to try.  if it worked, it
    # would obviate some of the code mess above, and be optimal, so that
    # sounds worthwhile.
    if not derivcentroids:
        m = psfqf < 0.5
    else:
        m = numpy.ones(len(xcen), dtype='bool')
    xcen[m] = 0.
    ycen[m] = 0.
    if (len(psflist) > 1) and numpy.sum(m) > 0:
        ind = numpy.flatnonzero(m)
        # just use the derivative-based centroids for this case.
        fluxnz = flux[repeat*ind]
        fluxnz = fluxnz + (fluxnz == 0)
        xcen[ind] = flux[repeat*ind+1]/fluxnz
        ycen[ind] = flux[repeat*ind+2]/fluxnz
    # stamps: 0: neighbor-subtracted images,
    # 1: images,
    # 2: psfs with shifts
    # 3: psfs without shifts
    res = (xcen, ycen, (modelst+residst, imst, modelst, weightst, psfst))
    return res


def estimate_sky_background(im):
    """Find peak of count distribution; pretend this is the sky background."""
    # for some reason, I have found this hard to work robustly.  Replace with
    # median at the moment.

    return numpy.median(im)


def sky_im(im, weight=None, npix=20, order=1):
    """Remove sky from image."""
    nbinx, nbiny = (numpy.ceil(sh/1./npix).astype('i4') for sh in im.shape)
    xg = numpy.linspace(0, im.shape[0], nbinx+1).astype('i4')
    yg = numpy.linspace(0, im.shape[1], nbiny+1).astype('i4')
    val = numpy.zeros((nbinx, nbiny), dtype='f4')
    usedpix = numpy.zeros((nbinx, nbiny), dtype='f4')
    if weight is None:
        weight = numpy.ones_like(im, dtype='f4')
    if numpy.all(weight == 0):
        return im*0
    # annoying!
    for i in range(nbinx):
        for j in range(nbiny):
            use = weight[xg[i]:xg[i+1], yg[j]:yg[j+1]] > 0
            usedpix[i, j] = numpy.sum(use)
            if usedpix[i, j] > 0:
                val[i, j] = estimate_sky_background(
                    im[xg[i]:xg[i+1], yg[j]:yg[j+1]][use])
    val[usedpix < 20] = 0.
    usedpix[usedpix < 20] = 0.
    from scipy.ndimage.filters import gaussian_filter
    count = 0
    while numpy.any(usedpix == 0):
        sig = 0.4
        valc = gaussian_filter(val*(usedpix > 0), sig, mode='constant')
        weightc = gaussian_filter((usedpix != 0).astype('f4'), sig,
                                  mode='constant')
        m = (usedpix == 0) & (weightc > 1.e-10)
        val[m] = valc[m]/weightc[m]
        usedpix[m] = 1
        count += 1
        if count > 100:
            m = usedpix == 0
            val[m] = numpy.median(im)
            print('Sky estimation failed badly.')
            break
    x = numpy.arange(im.shape[0])
    y = numpy.arange(im.shape[1])
    xc = (xg[:-1]+xg[1:])/2.
    yc = (yg[:-1]+yg[1:])/2.
    from scipy.ndimage import map_coordinates
    xp = numpy.interp(x, xc, numpy.arange(len(xc), dtype='f4'))
    yp = numpy.interp(y, yc, numpy.arange(len(yc), dtype='f4'))
    xpa = xp.reshape(-1, 1)*numpy.ones(len(yp)).reshape(1, -1)
    ypa = yp.reshape(1, -1)*numpy.ones(len(xp)).reshape(-1, 1)
    coord = [xpa.ravel(), ypa.ravel()]
    bg = map_coordinates(val, coord, mode='nearest', order=order)
    bg = bg.reshape(im.shape)
    return bg


def get_sizes(x, y, imbs, weight=None, blist=None):
    x = numpy.round(x).astype('i4')
    y = numpy.round(y).astype('i4')
    peakbright = imbs[x, y]
    sz = numpy.zeros(len(x), dtype='i4')
    cutoff = 1000
    sz[peakbright > cutoff] = 59
    sz[peakbright <= cutoff] = 19  # for the moment...
    # for very bright things, use a bigger PSF
    # but if there are too many of these, don't bother.
    cutoff2 = 20000
    if ((numpy.sum(peakbright > cutoff2) < numpy.sum(peakbright > cutoff)/2)
            or (numpy.sum(peakbright > cutoff) < 100)):
        sz[peakbright > cutoff2] = 149
    else:
        print('Too many bright sources, using smaller PSF stamp size...')

    if weight is not None:
        sz[weight[x, y] == 0] = 149  # saturated/off edge sources get big PSF

    # sources near listed sources get very big PSF
    if blist is not None and len(x) > 0:
        for xb, yb in zip(blist[0], blist[1]):
            dist2 = (x-xb)**2 + (y-yb)**2
            indclose = numpy.argmin(dist2)
            if dist2[indclose] < 5**2:
                sz[indclose] = 299
    return sz


def fit_im_force(im, x, y, psf, weight=None, dq=None, psfderiv=True,
                 nskyx=0, nskyy=0, refit_psf=False,
                 niter=4, blist=None, derivcentroids=False, refit_sky=True,
                 startsky=numpy.nan):
    repeat = 3 if psfderiv else 1
    guessflux = None
    msky = 0
    model = 0

    if len(x) == 0:
        raise ValueError('must force some sources')

    if derivcentroids and not psfderiv:
        raise ValueError('derivcentroids only makes sense when psfderiv '
                         'is true')

    for titer in range(niter):
        for c, s in zip((x, y), im.shape):
            if numpy.any((c < -0.499) | (c > s-0.501)):
                c[:] = numpy.clip(c, -0.499, s-0.501)
                print('Some positions within 0.01 pix of edge of image '
                      'clipped back to 0.01 pix inside image.')
        if (refit_sky and
                ((titer > 0) or numpy.any(~numpy.isfinite(startsky)))):
            sky = sky_im(im-model, weight=weight, npix=100)
        else:
            sky = startsky
        sz = get_sizes(x, y, im-sky-msky, weight=weight, blist=blist)
        minsz = numpy.min(sz)
        psfs = [numpy.zeros((len(x), minsz, minsz), dtype='f4')
                for i in range(repeat)]
        if guessflux is not None:
            guess = guessflux.copy()
        else:
            guess = None
        # should really only be done once in refit_psf=False case
        psfsfull = build_psf_list(x, y, psf, sz, psfderiv=psfderiv)
        # need to package some "tiling" around this eventually, probably?
        flux, model, msky = fit_once(
                im-sky, x, y, psfsfull,
                psfderiv=psfderiv, weight=weight, guess=guess,
                nskyx=nskyx, nskyy=nskyy)
        import gc
        gc.collect()
        flux = flux[0]
        skypar = flux[len(x)*repeat:]
        guessflux = flux[:len(x)*repeat:repeat]
        for i in range(repeat):
            psfs[i][...] = [psfmod.central_stamp(psfsfull[i][j], minsz)
                            for j in range(len(psfsfull[i]))]
        centroids = compute_centroids(x, y, psfs, flux, im-(sky+msky),
                                      im-model-sky,
                                      weight, derivcentroids=derivcentroids)
        xcen, ycen, stamps = centroids
        if refit_psf:
            psf, x, y = refit_psf_from_stamps(psf, x, y, xcen, ycen,
                                              stamps)
            # we are letting the positions get updated, even when
            # psfderiv is false, only for the mean shift that
            # gets introduced when we recentroid all the stars.
            # we could eliminate this by replacing the above with
            # psf, _, _ = refit_psf_from_stamps(...)
            # for WISE at the moment, this should _mostly_ introduce
            # a mean shift, and potentially also a small subpixel-offset
            # related shift.
        if psfderiv:
            if derivcentroids:
                maxstep = 1
            else:
                maxstep = 3
            dcen = numpy.sqrt(xcen**2 + ycen**2)
            m = dcen > maxstep
            xcen[m] /= dcen[m]
            ycen[m] /= dcen[m]
            x, y = (numpy.clip(c, -0.499, s-0.501)
                    for c, s in zip((x+xcen, y+ycen), im.shape))
        print('Iteration %d, median sky %6.2f' %
              (titer+1, numpy.median(sky+msky)))

    stats = compute_stats(x-numpy.round(x), y-numpy.round(y),
                          stamps[0], stamps[2], stamps[3], stamps[1], flux)
    if dq is not None:
        stats['flags'] = extract_im(x, y, dq).astype('i4')
    stats['sky'] = extract_im(x, y, sky+msky).astype('f4')

    stars = OrderedDict([('x', x), ('y', y), ('flux', flux),
                         ('deltx', xcen), ('delty', ycen)] +
                        [(f, stats[f]) for f in stats])
    dtypenames = list(stars.keys())
    dtypeformats = [stars[n].dtype for n in dtypenames]
    dtype = dict(names=dtypenames, formats=dtypeformats)
    stars = numpy.fromiter(zip(*stars.values()),
                           dtype=dtype, count=len(stars['x']))
    res = (stars, model+sky, sky+msky, psf)
    return res


def refit_psf_from_stamps(psf, x, y, xcen, ycen, stamps, name=None,
                          plot=False):
    # how far the centroids of the model PSFs would
    # be from (0, 0) if instantiated there
    # this initial definition includes the known offset (since
    # we instantiated off a pixel center), and the model offset
    xe, ye = psfmod.simple_centroid(
        psfmod.central_stamp(stamps[4], censize=stamps[0].shape[-1]))
    # now we subtract the known offset
    xe -= x-numpy.round(x)
    ye -= y-numpy.round(y)
    if hasattr(psf, 'fitfun'):
        psffitfun = psf.fitfun
        npsf = psffitfun(x, y, xcen+xe, ycen+ye, stamps[0],
                         stamps[1], stamps[2], stamps[3], nkeep=200,
                         name=name, plot=plot)
        if npsf is not None:
            npsf.fitfun = psffitfun
    else:
        shiftx = xcen + xe + x - numpy.round(x)
        shifty = ycen + ye + y - numpy.round(y)
        npsf = find_psf(x, shiftx, y, shifty,
                        stamps[0], stamps[3], stamps[1])
        # we removed the centroid offset of the model PSFs;
        # we need to correct the positions to compensate
    if npsf is not None:
        xnew = x + xe
        ynew = y + ye
        psf = npsf
    else:
        xnew = x
        ynew = y
    return psf, xnew, ynew


def fit_im(im, psf, weight=None, dq=None, psfderiv=True,
           nskyx=0, nskyy=0, refit_psf=False,
           verbose=False, miniter=4, maxiter=10, blist=None,
           maxstars=40000, derivcentroids=False,
           ntilex=1, ntiley=1, fewstars=100, threshold=5,
           ccd=None, plot=False, titer_thresh=2, blendthreshu=2,
           psfvalsharpcutfac=0.7, psfsharpsat=0.7):
    if isinstance(weight, int):
        weight = numpy.ones_like(im)*weight

    model = numpy.zeros_like(im)
    xa = numpy.zeros(0, dtype='f4')
    ya = xa.copy()
    lsky = numpy.median(im[weight > 0])
    hsky = numpy.median(im[weight > 0])
    msky = numpy.zeros_like(im)
    passno = numpy.zeros(0, dtype='i4')
    guessflux, guesssky = None, None
    titer = -1
    lastiter = -1
    skypar = {}  # best sky parameters so far.

    roughfwhm = psfmod.neff_fwhm(psf(im.shape[0]//2, im.shape[1]//2))
    roughfwhm = numpy.max([roughfwhm, 3.])

    while True:
        titer += 1
        hsky = sky_im(im-model, weight=weight, npix=20)
        lsky = sky_im(im-model, weight=weight, npix=50*roughfwhm)
        if titer != lastiter:
            # in first passes, do not split sources!
            blendthresh = blendthreshu if titer < titer_thresh else 0.2
            xn, yn = peakfind(im-model-hsky,
                              model-msky, weight, dq, psf,
                              keepsat=(titer == 0),
                              blendthreshold=blendthresh,
                              threshold=threshold,
                              psfvalsharpcutfac=psfvalsharpcutfac,
                              psfsharpsat=psfsharpsat)
            if len(xa) > 0 and len(xn) > 0:
                keep = neighbor_dist(xn, yn, xa, ya) > 1.5
                xn, yn = (c[keep] for c in (xn, yn))
            if (titer == 0) and (blist is not None):
                xnb, ynb = add_bright_stars(xn, yn, blist, im)
                xn = numpy.concatenate([xn, xnb]).astype('f4')
                yn = numpy.concatenate([yn, ynb]).astype('f4')

            xa, ya = (numpy.concatenate([xa, xn]).astype('f4'),
                      numpy.concatenate([ya, yn]).astype('f4'))
            passno = numpy.concatenate([passno, numpy.zeros(len(xn))+titer])
        else:
            xn, yn = numpy.zeros(0, dtype='f4'), numpy.zeros(0, dtype='f4')

        if titer != lastiter:
            if (titer == maxiter-1) or (
                    (titer >= miniter-1) and (len(xn) < fewstars)) or (
                    len(xa) > maxstars):
                lastiter = titer + 1
        # we probably don't want the sizes to change very much.  hsky certainly
        # will change a bit from iteration to iteration, though.
        sz = get_sizes(xa, ya, im-hsky-msky, weight=weight, blist=blist)
        if guessflux is not None:
            guess = numpy.concatenate([guessflux, numpy.zeros_like(xn)])
        else:
            guess = None
        sky = hsky if titer >= 2 else lsky

        # in final iteration, no longer allow shifting locations; just fit
        # centroids.
        tpsfderiv = psfderiv if lastiter != titer else False
        repeat = 1+tpsfderiv*2
        if len(sz) != 0:
            minsz = numpy.min(sz)
        else:
            minsz = 19
        psfs = [numpy.zeros((len(xa), minsz, minsz), dtype='f4')
                for i in range(repeat)]
        flux = numpy.zeros(len(xa)*repeat, dtype='f4')
        for (bdxf, bdxl, bdxaf, bdxal, bdyf, bdyl, bdyaf, bdyal) in (
                subregions(im.shape, ntilex, ntiley)):
            mbda = in_bounds(xa, ya, [bdxaf-0.5, bdxal-0.5],
                             [bdyaf-0.5, bdyal-0.5])
            mbd = in_bounds(xa, ya, [bdxf-0.5, bdxl-0.5],
                            [bdyf-0.5, bdyl-0.5])
            psfsbda = build_psf_list(xa[mbda], ya[mbda], psf, sz[mbda],
                                     psfderiv=tpsfderiv)
            sall = numpy.s_[bdxaf:bdxal, bdyaf:bdyal]
            spri = numpy.s_[bdxf:bdxl, bdyf:bdyl]
            dx, dy = bdxal-bdxaf, bdyal-bdyaf
            sfit = numpy.s_[bdxf-bdxaf:dx+bdxl-bdxal,
                            bdyf-bdyaf:dy+bdyl-bdyal]
            weightbda = weight[sall] if weight is not None else None
            guessmbda = guess[mbda] if guess is not None else None
            guesssky = skypar.get((bdxf, bdyf))
            guessmbda = (numpy.concatenate([guessmbda, guesssky])
                         if guessmbda is not None else None)
            tflux, tmodel, tmsky = fit_once(
                im[sall]-sky[sall], xa[mbda]-bdxaf, ya[mbda]-bdyaf, psfsbda,
                psfderiv=tpsfderiv, weight=weightbda, guess=guessmbda,
                nskyx=nskyx, nskyy=nskyy)
            model[spri] = tmodel[sfit]
            msky[spri] = tmsky[sfit]
            ind = numpy.flatnonzero(mbd)
            ind2 = numpy.flatnonzero(mbd[mbda])
            for i in range(repeat):
                flux[ind*repeat+i] = tflux[0][ind2*repeat+i]
            skypar[(bdxf, bdyf)] = flux[numpy.sum(mbda)*repeat:]
            for i in range(repeat):
                if len(ind2) == 0:
                    continue
                psfs[i][mbd] = [psfmod.central_stamp(psfsbda[i][tind], minsz)
                                for tind in ind2]
            # try to free memory!  Not sure where the circular reference
            # could be, but this makes a factor of a few difference
            # in peak memory usage on fields with lots of stars with
            # large models...
            del psfsbda
            import gc
            gc.collect()

        centroids = compute_centroids(xa, ya, psfs, flux, im-(sky+msky),
                                      im-model-sky,
                                      weight, derivcentroids=derivcentroids)

        xcen, ycen, stamps = centroids
        if titer == lastiter:
            stats = compute_stats(xa-numpy.round(xa), ya-numpy.round(ya),
                                  stamps[0], stamps[2],
                                  stamps[3], stamps[1],
                                  flux)
            if dq is not None:
                stats['flags'] = extract_im(xa, ya, dq).astype('i4')
            stats['sky'] = extract_im(xa, ya, sky+msky).astype('f4')
            break
        guessflux = flux[:len(xa)*repeat:repeat]
        if refit_psf and len(xa) > 0:
            psf, xa, ya = refit_psf_from_stamps(
                psf, xa, ya, xcen, ycen, stamps, name=(titer, ccd), plot=plot)
        # enforce maximum step
        if derivcentroids:
            maxstep = 1
        else:
            maxstep = 3
        dcen = numpy.sqrt(xcen**2 + ycen**2)
        m = dcen > maxstep
        xcen[m] /= dcen[m]
        ycen[m] /= dcen[m]
        xa, ya = (numpy.clip(c, -0.499, s-0.501)
                  for c, s in zip((xa+xcen, ya+ycen), im.shape))
        fluxunc = numpy.sum(stamps[2]**2.*stamps[3]**2., axis=(1, 2))
        fluxunc = fluxunc + (fluxunc == 0)*1e-20
        fluxunc = (fluxunc**(-0.5)).astype('f4')
        # for very bright stars, fluxunc is unreliable because the entire
        # (small) stamp is saturated.
        # these stars all have very bright inferred fluxes
        # i.e., 50k saturates, so we can cut there.
        brightenough = (guessflux/fluxunc > threshold*3/5.) | (guessflux > 1e5)
        isolatedenough = cull_near(xa, ya, guessflux)

        keep = brightenough & isolatedenough
        xa, ya = (c[keep] for c in (xa, ya))
        passno = passno[keep]
        guessflux = guessflux[keep]
        if verbose:
            print('Extension %s, iteration %2d, found %6d sources; %4d close and '
                  '%4d faint sources removed.' %
                  (ccd, titer+1, len(xn),
                   numpy.sum(~isolatedenough),
                   numpy.sum(~brightenough & isolatedenough)))

        # should probably also subtract these stars from the model image
        # which is used for peak finding.  But the faint stars should
        # make little difference?

    # This is the end of the internal iteration loops
    # Prepares found sources for export
    stars = OrderedDict([('x', xa), ('y', ya), ('flux', flux),
                         ('passno', passno)] +
                        [(f, stats[f]) for f in stats])
    dtypenames = list(stars.keys())
    dtypeformats = [stars[n].dtype for n in dtypenames]
    dtype = dict(names=dtypenames, formats=dtypeformats)
    stars = numpy.fromiter(zip(*stars.values()),
                           dtype=dtype, count=len(stars['x']))
    res = (stars, model+sky, sky+msky, psf)
    return res


def compute_stats(xs, ys, impsfstack, psfstack, weightstack, imstack, flux):
    residstack = impsfstack - psfstack
    norm = numpy.sum(psfstack, axis=(1, 2))
    psfstack = psfstack / (norm + (norm == 0)).reshape(-1, 1, 1)
    qf = numpy.sum(psfstack*(weightstack > 0), axis=(1, 2))
    fluxunc = numpy.sum(psfstack**2.*weightstack**2., axis=(1, 2))
    fluxunc = fluxunc + (fluxunc == 0)*1e-20
    fluxunc = (fluxunc**(-0.5)).astype('f4')
    posunc = [numpy.zeros(len(qf), dtype='f4'),
              numpy.zeros(len(qf), dtype='f4')]
    psfderiv = numpy.gradient(-psfstack, axis=(1, 2))
    for i, p in enumerate(psfderiv):
        dp = numpy.sum((p*weightstack*flux[:, None, None])**2., axis=(1, 2))
        dp = dp + (dp == 0)*1e-40
        dp = dp**(-0.5)
        posunc[i][:] = dp
    rchi2 = numpy.sum(residstack**2.*weightstack**2.*psfstack,
                      axis=(1, 2)) / (qf + (qf == 0.)*1e-20).astype('f4')
    fracfluxn = numpy.sum(impsfstack*(weightstack > 0)*psfstack,
                          axis=(1, 2))
    fracfluxd = numpy.sum(imstack*(weightstack > 0)*psfstack,
                          axis=(1, 2))
    fracfluxd = fracfluxd + (fracfluxd == 0)*1e-20
    fracflux = (fracfluxn / fracfluxd).astype('f4')
    fluxlbs, dfluxlbs = compute_lbs_flux(impsfstack, psfstack, weightstack,
                                         flux/norm)
    fluxiso, xiso, yiso = compute_iso_fit(impsfstack, psfstack, weightstack,
                                          flux/norm, psfderiv)
    fluxlbs = fluxlbs.astype('f4')
    dfluxlbs = dfluxlbs.astype('f4')
    fwhm = psfmod.neff_fwhm(psfstack).astype('f4')
    spread, dspread = spread_model(impsfstack, psfstack, weightstack)
    return OrderedDict([('dx', posunc[0]), ('dy', posunc[1]),
                        ('dflux', fluxunc),
                        ('qf', qf), ('rchi2', rchi2), ('fracflux', fracflux),
                        ('fluxlbs', fluxlbs), ('dfluxlbs', dfluxlbs),
                        ('fwhm', fwhm), ('spread_model', spread),
                        ('dspread_model', dspread),
                        ('fluxiso', fluxiso), ('xiso', xiso), ('yiso', yiso)])


def spread_model(impsfstack, psfstack, weightstack):
    # need to convolve psfs with 1/16 FWHM exponential
    # can get FWHM from n_eff
    # better way?  n_eff can be a bit annoying; not necessarily what one
    # expects if there's a sharp peak on a broad background.
    # spread_model is on the whole a bit goofy: one sixteenth of a FWHM is very
    # little.  So this is really more like the significance of the derivative
    # of the PSF with radius, which I would compute a bit differently.
    # still, other people compute spread_model, and it's well defined, so...
    import crowdsource.galconv as galconv
    fwhm = psfmod.neff_fwhm(psfstack)
    sigma = fwhm/16.
    re = sigma * 1.67834699
    expgalstack = galconv.gal_psfstack_conv(re, 0, 0, galconv.ExpGalaxy,
                                            numpy.eye(2), 0, 0, psfstack)
    GWp = numpy.sum(expgalstack*weightstack**2*impsfstack, axis=(1, 2))
    PWp = numpy.sum(psfstack*weightstack**2*impsfstack, axis=(1, 2))
    GWP = numpy.sum(expgalstack*weightstack**2*psfstack, axis=(1, 2))
    PWP = numpy.sum(psfstack**2*weightstack**2, axis=(1, 2))
    GWG = numpy.sum(expgalstack**2*weightstack**2, axis=(1, 2))
    spread = (GWp/(PWp+(PWp == 0)) - GWP/(PWP+(PWP == 0)))
    dspread = numpy.sqrt(numpy.clip(
        PWp**2*GWG + GWp**2*PWP - 2*GWp*PWp*GWP, 0, numpy.inf)
                         /(PWp + (PWp == 0))**4)
    return spread, dspread


def extract_im(xa, ya, im, sentinel=999):
    m = numpy.ones(len(xa), dtype='bool')
    for c, sz in zip((xa, ya), im.shape):
        m = m & (c > -0.5) & (c < sz - 0.5)
    res = numpy.zeros(len(xa), dtype=im.dtype)
    res[~m] = sentinel
    xp, yp = (numpy.round(c[m]).astype('i4') for c in (xa, ya))
    res[m] = im[xp, yp]
    return res


def compute_lbs_flux(stamp, psf, isig, apcor):
    sumisig2 = numpy.sum(isig**2, axis=(1, 2))
    sumpsf2isig2 = numpy.sum(psf*psf*isig**2, axis=(1, 2))
    sumpsfisig2 = numpy.sum(psf*isig**2, axis=(1, 2))
    det = numpy.clip(sumisig2*sumpsf2isig2 - sumpsfisig2**2, 0, numpy.inf)
    det = det + (det == 0)
    unc = numpy.sqrt(sumisig2/det)
    flux = (sumisig2*numpy.sum(psf*stamp*isig**2, axis=(1, 2)) -
            sumpsfisig2*numpy.sum(stamp*isig**2, axis=(1, 2)))/det
    flux *= apcor
    unc *= apcor
    return flux, unc


def compute_iso_fit(impsfstack, psfstack, weightstack, apcor, psfderiv):
    nstar = len(impsfstack)
    par = numpy.zeros((nstar, 3), dtype='f4')
    for i in range(len(impsfstack)):
        aa = numpy.array([psfstack[i]*weightstack[i],
                          psfderiv[0][i]*weightstack[i],
                          psfderiv[1][i]*weightstack[i]])
        aa = aa.reshape(3, -1).T
        par[i, :] = numpy.linalg.lstsq(
            aa, (impsfstack[i]*weightstack[i]).reshape(-1), rcond=None)[0]
    zeroflux = par[:, 0] == 0
    return (par[:, 0],
            (1-zeroflux)*par[:, 1]/(par[:, 0]+zeroflux),
            (1-zeroflux)*par[:, 2]/(par[:, 0]+zeroflux))


def sky_model_basis(i, j, nskyx, nskyy, nx, ny):
    import basisspline
    if (nskyx < 3) or (nskyy < 3):
        raise ValueError('Invalid sky model.')
    expandx = (nskyx-1.)/(3-1)
    expandy = (nskyy-1.)/(3-1)
    xg = -expandx/3. + i*2/3.*expandx/(nskyx-1.)
    yg = -expandy/3. + j*2/3.*expandy/(nskyy-1.)
    x = numpy.linspace(-expandx/3.+1/6., expandx/3.-1/6., nx).reshape(-1, 1)
    y = numpy.linspace(-expandy/3.+1/6., expandy/3.-1/6., ny).reshape(1, -1)
    return basisspline.basis2dq(x-xg, y-yg)


def sky_model(coeff, nx, ny):
    # minimum sky model: if we want to use the quadratic basis functions we
    # implemented, and we want to allow a constant sky over the frame, then we
    # need at least 9 basis polynomials: [-0.5, 0.5, 1.5] x [-0.5, 0.5, 1.5].
    nskyx, nskyy = coeff.shape
    if (coeff.shape[0] == 1) & (coeff.shape[1]) == 1:
        return coeff[0, 0]*numpy.ones((nx, ny), dtype='f4')
    if (coeff.shape[0] < 3) or (coeff.shape[1]) < 3:
        raise ValueError('Not obvious what to do for <3')
    im = numpy.zeros((nx, ny), dtype='f4')
    for i in range(coeff.shape[0]):
        for j in range(coeff.shape[1]):
            # missing here: speed up available from knowing that
            # the basisspline is zero over a large area.
            im += coeff[i, j] * sky_model_basis(i, j, nskyx, nskyy, nx, ny)
    return im


def sky_parameters(nx, ny, nskyx, nskyy, weight):
    # yloc: just add rows to the end according to the current largest row
    # in there
    nskypar = nskyx * nskyy
    xloc = [numpy.arange(nx*ny, dtype='i4')]*nskypar
    # for the moment, don't take advantage of the bounded support.
    yloc = [i*numpy.ones((nx, ny), dtype='i4').ravel()
            for i in range(nskypar)]
    if (nskyx == 1) & (nskyy == 1):
        values = [(numpy.ones((nx, ny), dtype='f4')*weight).ravel()
                  for yl in yloc]
    else:
        values = [(sky_model_basis(i, j, nskyx, nskyy, nx, ny)*weight).ravel()
                  for i in range(nskyx) for j in range(nskyy)]
    return xloc, yloc, values


def cull_near(x, y, flux):
    """Delete faint sources within 1 pixel of a brighter source.

    Args:
        x (ndarray, int[N]): x coordinates for N sources
        y (ndarray, int[N]): y coordinates
        flux (ndarray, int[N]): fluxes

    Returns:
        ndarray (bool[N]): mask array indicating sources to keep
    """
    if len(x) == 0:
        return numpy.ones(len(x), dtype='bool')
    m1, m2, dist = match_xy(x, y, x, y, neighbors=6)
    m = (dist < 1) & (flux[m1] < flux[m2]) & (m1 != m2)
    keep = numpy.ones(len(x), dtype='bool')
    keep[m1[m]] = 0
    return keep


def neighbor_dist(x1, y1, x2, y2):
    """Return distance of nearest neighbor to x1, y1 in x2, y2"""
    m1, m2, d12 = match_xy(x2, y2, x1, y1, neighbors=1)
    return d12


def match_xy(x1, y1, x2, y2, neighbors=1):
    """Match x1 & y1 to x2 & y2, neighbors nearest neighbors.

    Finds the neighbors nearest neighbors to each point in x2, y2 among
    all x1, y1."""
    from scipy.spatial import cKDTree
    vec1 = numpy.array([x1, y1]).T
    vec2 = numpy.array([x2, y2]).T
    kdt = cKDTree(vec1)
    dist, idx = kdt.query(vec2, neighbors)
    m1 = idx.ravel()
    m2 = numpy.repeat(numpy.arange(len(vec2), dtype='i4'), neighbors)
    dist = dist.ravel()
    dist = dist
    m = m1 < len(x1)  # possible if fewer than neighbors elements in x1.
    return m1[m], m2[m], dist[m]


def add_bright_stars(xa, ya, blist, im):
    xout = []
    yout = []
    for x, y, mag in zip(*blist):
        if ((x < -0.499) or (x > im.shape[0]-0.501) or
            (y < -0.499) or (y > im.shape[1]-0.501)):
            continue
        if len(xa) > 0:
            mindist2 = numpy.min((x-xa)**2 + (y-ya)**2)
        else:
            mindist2 = 9999
        if mindist2 > 5**2:
            xout.append(x)
            yout.append(y)
    return (numpy.array(xout, dtype='f4'), numpy.array(yout, dtype='f4'))


# This is almost entirely deprecated for the psf.py module... go look there.
def find_psf(xcen, shiftx, ycen, shifty, psfstack, weightstack,
             imstack, stampsz=59, nkeep=100):
    """Find PSF from stamps."""
    # let's just go ahead and correlate the noise
    xr = numpy.round(shiftx)
    yr = numpy.round(shifty)
    psfqf = (numpy.sum(psfstack*(weightstack > 0), axis=(1, 2)) /
             numpy.sum(psfstack, axis=(1, 2)))
    totalflux = numpy.sum(psfstack, axis=(1, 2))
    timflux = numpy.sum(imstack, axis=(1, 2))
    toneflux = numpy.sum(psfstack, axis=(1, 2))
    tmedflux = numpy.median(psfstack, axis=(1, 2))
    tfracflux = toneflux / numpy.clip(timflux, 100, numpy.inf)
    tfracflux2 = ((toneflux-tmedflux*psfstack.shape[1]*psfstack.shape[2]) /
                  numpy.clip(timflux, 100, numpy.inf))
    okpsf = ((numpy.abs(psfqf - 1) < 0.03) &
             (tfracflux > 0.5) & (tfracflux2 > 0.2))
    if numpy.sum(okpsf) > 0:
        shiftxm = numpy.median(shiftx[okpsf])
        shiftym = numpy.median(shifty[okpsf])
        okpsf = (okpsf &
                 (numpy.abs(shiftx-shiftxm) < 1.) &
                 (numpy.abs(shifty-shiftym) < 1.))
    if numpy.sum(okpsf) <= 5:
        print('Fewer than 5 stars accepted in image, keeping original PSF')
        return None
    if numpy.sum(okpsf) > nkeep:
        okpsf = okpsf & (totalflux > -numpy.sort(-totalflux[okpsf])[nkeep-1])
    psfstack = psfstack[okpsf, :, :]
    weightstack = weightstack[okpsf, :, :]
    totalflux = totalflux[okpsf]
    xcen = xcen[okpsf]
    ycen = ycen[okpsf]
    shiftx = shiftx[okpsf]
    shifty = shifty[okpsf]
    for i in range(psfstack.shape[0]):
        psfstack[i, :, :] = shift(psfstack[i, :, :], [-shiftx[i], -shifty[i]])
        if (numpy.abs(xr[i]) > 0) or (numpy.abs(yr[i]) > 0):
            weightstack[i, :, :] = shift(weightstack[i, :, :],
                                         [-xr[i], -yr[i]],
                                         mode='constant', cval=0.)
        # our best guess as to the PSFs & their weights
    # select some reasonable sample of the PSFs
    totalflux = numpy.sum(psfstack, axis=(1, 2))
    psfstack /= totalflux.reshape(-1, 1, 1)
    weightstack *= totalflux.reshape(-1, 1, 1)
    tpsf = numpy.median(psfstack, axis=0)
    tpsf = psfmod.center_psf(tpsf)
    if tpsf.shape == stampsz:
        return tpsf
    xc = numpy.arange(tpsf.shape[0]).reshape(-1, 1)-tpsf.shape[0]//2
    yc = xc.reshape(1, -1)
    rc = numpy.sqrt(xc**2.+yc**2.)
    stampszo2 = psfstack[0].shape[0] // 2
    wt = numpy.clip((stampszo2+1-rc)/4., 0., 1.)
    overlap = (wt != 1) & (wt != 0)

    def objective(par):
        mod = psfmod.moffat_psf(par[0], beta=2.5, xy=par[2], yy=par[3],
                                deriv=False, stampsz=tpsf.shape[0])
        mod /= numpy.sum(mod)
        return ((tpsf-mod)[overlap]).reshape(-1)
    from scipy.optimize import leastsq
    par = leastsq(objective, [4., 3., 0., 1.])[0]
    modpsf = psfmod.moffat_psf(par[0], beta=2.5, xy=par[2], yy=par[3],
                               deriv=False, stampsz=stampsz)
    modpsf /= numpy.sum(psfmod.central_stamp(modpsf))
    npsf = modpsf.copy()
    npsfcen = psfmod.central_stamp(npsf, tpsf.shape[0])
    npsfcen[:, :] = tpsf*wt+(1-wt)*npsfcen[:, :]
    npsf /= numpy.sum(npsf)
    return psfmod.SimplePSF(npsf, normalize=-1)


def subregions(shape, nx, ny, overlap=149):
    # ugh.  I guess we want:
    # starts and ends of each _primary_ fit region
    # starts and ends of each _entire_ fit region
    # should be nothing else?
    # need this for both x and y: 8 things to return.
    nx = nx if nx > 0 else 1
    ny = ny if ny > 0 else 1
    bdx = numpy.round(numpy.linspace(0, shape[0], nx+1)).astype('i4')
    bdlx = numpy.clip(bdx - overlap, 0, shape[0])
    bdrx = numpy.clip(bdx + overlap, 0, shape[0])
    bdy = numpy.round(numpy.linspace(0, shape[1], ny+1)).astype('i4')
    bdly = numpy.clip(bdy - overlap, 0, shape[1])
    bdry = numpy.clip(bdy + overlap, 0, shape[1])
    xf = bdx[:nx]
    xl = bdx[1:]
    xaf = bdlx[:nx]
    xal = bdrx[1:]
    yf = bdy[:nx]
    yl = bdy[1:]
    yaf = bdly[:nx]
    yal = bdry[1:]
    for i in range(nx):
        for j in range(ny):
            yield (xf[i], xl[i], xaf[i], xal[i], yf[j], yl[j], yaf[j], yal[j])


def in_bounds(x, y, xbound, ybound):
    return ((x > xbound[0]) & (x <= xbound[1]) &
            (y > ybound[0]) & (y <= ybound[1]))
