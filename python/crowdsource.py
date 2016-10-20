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


import numpy
import scipy
import pdb
import scipy.ndimage.filters as filters


def shift(im, offset, **kw):
    """Wrapper for scipy.ndimage.interpolation.shift"""
    from scipy.ndimage.interpolation import shift
    if 'order' not in kw:
        kw['order'] = 4  # order 3: ~0.6 mmag; order=4 ~0.2 mmag
    if 'mode' not in kw:
        kw['mode'] = 'nearest'
    if 'output' not in kw:
        kw['output'] = im.dtype
    return shift(im, offset, **kw)


def sim_image(nx, ny, nstar, fwhm, noise, return_psf=False, nskyx=3, nskyy=3):
    im = numpy.random.randn(nx, ny).astype('f4')*noise
    stampsz = (6*numpy.ceil(fwhm)+1).astype('i4')
    stampszo2 = (3*numpy.ceil(fwhm)).astype('i4')
    im = numpy.pad(im, [stampszo2, stampszo2], constant_values=-1e6,
                   mode='constant')
    x = numpy.random.rand(nstar).astype('f4')*(nx-1)
    y = numpy.random.rand(nstar).astype('f4')*(ny-1)
    xp = numpy.round(x).astype('i4')
    yp = numpy.round(y).astype('i4')
    xc = numpy.arange(stampsz, dtype='f4')-stampszo2
    yc = xc.copy()
    sigma = fwhm / numpy.sqrt(8*numpy.log(2))
    psf = numpy.exp(
        -(xc.reshape(-1, 1)**2. + yc.reshape(1, -1)**2.) /
        2./sigma**2.).astype('f4')
    flux = 1./numpy.random.power(1.2, nstar)
    # pdb.set_trace()
    for i in range(nstar):
        fracshiftx = x[i]-numpy.round(x[i])
        fracshifty = y[i]-numpy.round(y[i])
        psf2 = shift(psf, [fracshiftx, fracshifty], output=numpy.dtype('f4'))
        im[xp[i]:xp[i]+stampsz, yp[i]:yp[i]+stampsz] += psf2*flux[i]
    if (nskyx != 0) or (nskyy != 0):
        im += sky_model(100*numpy.random.rand(nskyx, nskyy).astype('f4'),
                        im.shape[0], im.shape[1])
    ret = im[stampszo2:-stampszo2, stampszo2:-stampszo2], x, y, flux
    if return_psf:
        ret = (ret, psf)
    return ret


def significance_image(im, isig, psf):
    # assume, for the moment, the image has already been sky-subtracted
    # stampszo2 = (psf.shape[0]-1)/2
    from scipy.ndimage.filters import convolve
    f = numpy.max([0, psf.shape[0]//2 - 9])
    l = psf.shape[0] - f
    psfstamp = psf[f:l, f:l].copy()
    sigim = convolve(im*isig**2., psfstamp, mode='constant', origin=0)
    varim = convolve(isig**2., psfstamp**2., mode='constant', origin=0)
    ivarim = 1./(varim + (varim == 0) * 1e12)
    return sigim * numpy.sqrt(ivarim)


def peakfind(im, sigim, isig, keepsat=False, threshhold=5):
    fac = 0.3
    data_max = filters.maximum_filter(im, 3)
    sig_max = filters.maximum_filter(sigim, 3)
    isig_min = filters.minimum_filter(isig, 3)
    # don't accept a source where isig != 0, but isig_min = 0
    # this excludes sources on the edges of masked regions, but not
    # sources in the centers of masked regions (saturated stars)
    exclude = (isig_min == 0)
    if keepsat:
        exclude = exclude & (isig > 0)
    brightpeak = numpy.nonzero((data_max == im) &
                               (im*isig > threshhold/fac) & ~exclude)
    faintpeak = numpy.nonzero((sig_max == sigim) & (sigim > threshhold) &
                              (im*isig <= threshhold/fac) & ~exclude)
    return [numpy.concatenate([b, f]) for b, f in zip(brightpeak, faintpeak)]
    # return numpy.nonzero((data_max == im) & (im > 0) & (sigim > threshhold))


def peaksig(x, y, im, modelim, psf):
    sigstat1 = (im/numpy.clip(modelim, 0.3, numpy.inf))[x, y]
    from scipy.ndimage.filters import convolve
    f = numpy.max([0, psf.shape[0]//2 - 9])
    l = psf.shape[0] - f
    psfstamp = psf[f:l, f:l].copy()
    sigstat2 = (im/numpy.clip(convolve(modelim, psfstamp, mode='constant',
                                       origin=0), 0.3, numpy.inf))[x, y]
    return sigstat1, sigstat2


def build_model(x, y, flux, nx, ny, psf=None, psflist=None, psfderiv=None):
    if psf is None and psflist is None:
        raise ValueError('One of psf and psflist must be set')
    if psf is not None and psflist is not None:
        raise ValueError('Only one of psf and psflist must be set')
    if psf is not None:
        psflist = {'stamp': [psf], 'ind': numpy.zeros(len(x), dtype='i4')}
        if psfderiv is not None:
            psflist['psfderiv'] = [psfderiv]
    stampsz = psflist['stamp'][0].shape[0]
    stampszo2 = int(numpy.ceil(stampsz/2.)-1)
    im = numpy.zeros((nx, ny), dtype='f4')
    im = numpy.pad(im, [stampszo2, stampszo2], constant_values=0.,
                   mode='constant')
    xp = numpy.round(x).astype('i4')
    yp = numpy.round(y).astype('i4')
    xf = x - xp
    yf = y - yp
    # _subtract_ stampszo2 to move from the center of the PSF to the edge
    # of the stamp.
    # _add_ it back to move from the original image to the padded image.
    xe = xp - stampszo2 + stampszo2
    ye = yp - stampszo2 + stampszo2
    repeat = 3 if 'psfderiv' in psflist else 1
    for i in range(len(x)):
        psf = psflist['stamp'][psflist['ind'][i]]
        im[xe[i]:xe[i]+stampsz, ye[i]:ye[i]+stampsz] += (
            shift(psf, [xf[i], yf[i]], output=numpy.dtype('f4')) *
            flux[i*repeat])
        if 'psfderiv' in psflist:
            psfderiv = psflist['psfderiv'][psflist['ind'][i]]
            for j, p in enumerate(psfderiv):
                im[xe[i]:xe[i]+stampsz, ye[i]:ye[i]+stampsz] += (
                    p*flux[i*repeat+j+1])
    im = im[stampszo2:-stampszo2, stampszo2:-stampszo2]
    # ignoring varying PSF sizes.  We only use maximum size here.
    return im


def in_padded_region(flatcoord, imshape, pad):
    coord = numpy.unravel_index(flatcoord, imshape)
    m = numpy.zeros(len(flatcoord), dtype='bool')
    for c, length in zip(coord, imshape):
        m |= (c < pad) | (c >= length - pad)
    return m


def fit_once(im, x, y, psf, weight=None, psfderiv=None, nskyx=0, nskyy=0,
             guess=None, sz=None):
    """Fit fluxes for psfs at x & y in image im.

    Args:
        im (ndarray[NX, NY] float): image to fit
        x (ndarray[NS] float): x coord
        y (ndarray[NS] float): y coord
        psf (ndarray[sz, sz] float): psf stamp
        weight (ndarray[NX, NY] float): weight for image
        psfderiv (tuple(ndarray[sz, sz] float)): x, y derivatives of psf image
        nskyx (int): number of sky pixels in x direction (0 or >= 3)
        nskyy (int): numpy. of sky pixels in y direction (0 or >= 3)

    Returns:
        tuple(flux, model, sky)
        flux: output of optimization routine; needs to be refined
        model (ndarray[NX, NY]): best fit model image
        sky (ndarray(NX, NY]): best fit model sky
    """
    # sparse matrix, with rows at first equal to the fluxes at each peak
    # later add in the derivatives at each peak
    if sz is None:
        sz = numpy.ones(len(x), dtype='i4')*psf.shape[0]
    stampsz = numpy.max(sz)
    stampszo2 = stampsz // 2
    szo2 = sz // 2
    nx, ny = im.shape
    im = numpy.pad(im, [stampszo2, stampszo2], constant_values=0.,
                   mode='constant')
    if weight is None:
        weight = numpy.ones_like(im)
    weight = numpy.pad(weight, [stampszo2, stampszo2], constant_values=0.,
                       mode='constant')
    weight[weight == 0.] = 1.e-20
    pix = numpy.arange(stampsz*stampsz, dtype='i4').reshape(stampsz, stampsz)
    # convention: x is the first index, y is the second
    # sorry.
    xpix = pix // stampsz
    ypix = pix % stampsz
    xp = numpy.round(x).astype('i4')
    yp = numpy.round(y).astype('i4')
    xf = x - xp
    yf = y - yp
    # _subtract_ stampszo2 to move from the center of the PSF to the edge
    # of the stamp.
    # _add_ it back to move from the original image to the padded image.
    xe = xp - stampszo2 + stampszo2
    ye = yp - stampszo2 + stampszo2
    repeat = 1 if not psfderiv else 3
    nskypar = nskyx * nskyy
    npixim = im.shape[0]*im.shape[1]
    xloc = numpy.zeros(repeat*numpy.sum(sz*sz) + nskypar*npixim, dtype='i4')
    yloc = numpy.zeros(len(xloc), dtype='i4')
    values = numpy.zeros(len(yloc), dtype='f4')
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
            yloc[first:first+sz[i]**2] = i*repeat+j
            if j == 0:
                values[first:first+sz[i]**2] = (
                    shift(psf[f:l, f:l], [xf[i], yf[i]],
                          output=numpy.dtype('f4'))*wt).reshape(-1)
            else:
                values[first:first+sz[i]**2] = (
                    (psfderiv[j-1][f:l, f:l]*wt).reshape(-1))
            colnorm[i*repeat+j] = numpy.sqrt(
                numpy.sum(values[first:first+sz[i]**2]**2.))
            colnorm[i*repeat+j] += (colnorm[i*repeat+j] == 0)
            values[first:first+sz[i]**2] /= colnorm[i*repeat+j]
            first += sz[i]**2
    #npixpsf = stampsz**2.
    #tvalues = values[:first].reshape(-1, npixpsf)  # view
    #colnorm2 = colnorm.copy()
    #colnorm2[:repeat*len(xe)] = numpy.sqrt(numpy.sum(tvalues**2., axis=1))
    #colnorm2[:repeat*len(xe)] += colnorm[:repeat*len(xe)] == 0.
    #pdb.set_trace()
    #tvalues /= colnorm[:repeat*len(xe)].reshape(-1, 1)

    if nskypar != 0:
        sxloc, syloc, svalues = sky_parameters(nx+stampszo2*2, ny+stampszo2*2,
                                               nskyx, nskyy, weight)
        startidx = len(x)*repeat
        nskypix = len(sxloc[0])
        for i in range(len(sxloc)):
            xloc[first:first+nskypix] = sxloc[i]
            yloc[first:first+nskypix] = startidx+syloc[i]
            colnorm[startidx+i] = numpy.sqrt(numpy.sum(svalues[i]**2.))
            colnorm[startidx+i] += (colnorm[startidx+i] == 0.)
            values[first:first+nskypix] = svalues[i] / colnorm[startidx+i]
            first += nskypix
    shape = (im.shape[0]*im.shape[1], len(x)*repeat+nskypar)

    from scipy import sparse
    # mat = sparse.csc_matrix((values, (xloc, yloc)), shape=shape,
    #                         dtype='f4')
    # csc_indptr = numpy.array([i*sz[i]**2 for i in range(len(xe)*repeat+1)])
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
        guessvec[len(xe)*repeat:] = guess[len(xe):]
        guessvec *= colnorm
    else:
        guessvec = None
    flux = lsqr_cp(mat, (im*weight).ravel(), atol=1.e-4, btol=1.e-4,
                   guess=guessvec)
    model = mat.dot(flux[0]).reshape(*im.shape)
    flux[0][:] = flux[0][:] / colnorm
    im = im[stampszo2:-stampszo2, stampszo2:-stampszo2]
    model = model[stampszo2:-stampszo2, stampszo2:-stampszo2]
    weight = weight[stampszo2:-stampszo2, stampszo2:-stampszo2]
    if nskypar != 0:
        sky = sky_model(flux[0][-nskypar:].reshape(nskyx, nskyy),
                        nx+stampszo2*2, ny+stampszo2*2)
        sky = sky[stampszo2:-stampszo2, stampszo2:-stampszo2]
    else:
        sky = model * 0
    model = model / (weight + (weight == 0))
    # sky = sky / (weight + (weight == 0))
    res = (flux, model, sky)
    return res


def unpack_fitpar(guess, nsource, psfderiv):
    """Extract fluxes and sky parameters from fit parameter vector."""
    repeat = 3 if psfderiv else 1
    return guess[0:nsource*repeat:repeat], guess[nsource*repeat:]


def lsqr_cp(aa, bb, guess=None, **kw):
    # implement two possible speed-ups:
    # 1. "column preconditioning": make sure each column of aa has the same
    #    norm
    # 2. allow guesses

    # column preconditioning means we need to rescale the columns of aa
    # by their norms.  I guess leave zero columns alone (should only happen
    # if a star gets placed in a fully weight=0 location, and even that case
    # I'm considering eliminating; could give these places tiny weights).
    # Then rescale the fit parameters to take the column norms back out.

    # allow guesses: solving Ax = b is the same as solving A(x-x*) = b-Ax*.
    # => A(dx) = b-Ax*.  So we can solve for dx instead, then return dx+x*.
    # this might improve speed?
    from scipy.sparse import linalg

    if guess is not None:
        bb2 = bb - aa.dot(guess)
        if 'btol' in kw:
            fac = numpy.sum(bb**2.)**(0.5)/numpy.sum(bb2**2.)**0.5
            kw['btol'] = kw['btol']*numpy.clip(fac, 0.1, 10.)
    else:
        bb2 = bb.copy()

    # column preconditioning:
    # no longer necessary: built into fit_once
    # aacsc = aa.tocsc().copy()
    # norm = numpy.array(aacsc.multiply(aacsc).sum(axis=0)).squeeze()
    # norm = norm + (norm == 0)
    # from scipy.sparse import diags
    # aacsc = aacsc.dot(diags(norm**(-0.5)))
    # aacsc has columns of constant norm.

    normbb = numpy.sum(bb2**2.)
    bb2 /= normbb**(0.5)
    par = linalg.lsqr(aa, bb2, **kw)
    # or lsmr; lsqr seems to be better
    # par[0][:] *= norm**(-0.5)*normbb**(0.5)
    par[0][:] *= normbb**0.5
    if guess is not None:
        par[0][:] += guess
    return par


def central_stamp(stamp, censize=19):
    # placeholder
    stampsz = stamp.shape[0]
    if stampsz <= censize:
        return stamp
    else:
        if (stamp.shape[0] % 2) == 0:
            pdb.set_trace()
        trim = (stamp.shape[0] - censize)/2
        f = trim
        l = stampsz - trim
        return stamp[f:l, f:l]


def compute_centroids(x, y, psf, flux, resid, weight, psfderiv=None,
                      return_psfstack=False):
    # way more complicated than I'd like.
    # we just want the weighted centroids.
    # these are sum(x * I * W) / sum(I * W)
    # if we use the PSF model for W, it turns out these are always?
    # biased too small by exactly? one half.
    # So that's straightforward.
    # It turns out that if the PSF is asymmetric, these are further biased
    # by a constant; we subtract that with x3 and y3 below.  In principle
    # those don't need to be in the loop (though if the PSF varies that's
    # another story).
    # Then we additionally want to downweight pixels with high noise.
    # I'm sure there's a correct prescription here, but instead we just
    # weight by inverse variance, which feels right.
    # this can kick out random fractions of flux, and easily lead to, e.g.,
    # only using pixels from the right side of the PSF in the weight
    # computation.  This may introduce both multiplicative and additive
    # biases.  The additive part we subtract by explicitly calculating
    # how biased our model star would be (imm), when we compute the
    # centroid with weights (im1) and without weights (im2).
    # I don't know how to think about the multiplicative bias, and am
    # okay for the moment acknowledging that the centroids of stars on
    # masked regions will be problematic.

    # centroid is sum(x * weight) / sum(weight)
    # we'll make the weights = psf * image
    # we want to compute the centroids on the image after the other sources
    # have been subtracted off.
    # we construct this image by taking the residual image, and then
    # star-by-star adding the model back.
    psf = central_stamp(psf).copy()
    if psfderiv is not None:
        psfderiv = [central_stamp(p).copy() for p in psfderiv]
    stampsz = psf.shape[0]
    stampszo2 = (psf.shape[1]-1)/2
    dx = numpy.arange(stampsz, dtype='i4')-stampszo2
    dx = dx.reshape(-1, 1)
    dy = dx.copy().reshape(1, -1)
    # there's a much faster way to do this:
    # we only really need to calculate x*psf*res and psf*res everywhere
    # the numerator of the centroid is sum(x * psf * (res + mod))
    # the denominator is sum(psf * mod)
    # since the model is a psf, the denominator is basically fixed to
    # flux * sum(psf**2) for everything, modulo shifting, possibly
    # with a sum(psf * shift * psfderiv) term.
    # if psfderiv:
    #     psfderivmomx = [numpy.sum(psf * p * dx)/numpy.sum(psf * p)
    #                     for p in psfderiv]
    #     psfderivmomy = [numpy.sum(psf * p * dy)/numpy.sum(psf * p)
    #                     for p in psfderiv]
    xp = numpy.round(x).astype('i4')
    yp = numpy.round(y).astype('i4')
    # subtracting to get to the edge of the stamp, adding back to deal with
    # the padded image.
    xe = xp - stampszo2 + stampszo2
    ye = yp - stampszo2 + stampszo2
    xcen = numpy.zeros(len(x), dtype='f4')
    ycen = xcen.copy()
    resid = numpy.pad(resid, [stampszo2, stampszo2], constant_values=0.,
                      mode='constant')
    weight = numpy.pad(weight, [stampszo2, stampszo2], constant_values=0.,
                       mode='constant')
    repeat = 3 if psfderiv else 1
    xf = x - xp
    yf = y - yp
    residst = numpy.array([resid[xe0:xe0+stampsz, ye0:ye0+stampsz]
                           for (xe0, ye0) in zip(xe, ye)])
    weightst = numpy.array([weight[xe0:xe0+stampsz, ye0:ye0+stampsz]
                            for (xe0, ye0) in zip(xe, ye)])
    psfst = numpy.array([shift(psf, [xf0, yf0], output=numpy.dtype('f4'))
                         for (xf0, yf0) in zip(xf, yf)])
    modelst = psfst * flux[:len(x)*repeat:repeat].reshape(-1, 1, 1)
    if psfderiv is not None:
        for i, psfderiv0 in enumerate(psfderiv):
            modelst += (psfderiv0[None, :, :] *
                        flux[i+1:len(x)*repeat:repeat].reshape(-1, 1, 1))
    cen = []
    denom0 = numpy.sum((modelst+residst)*psfst*weightst, axis=(1, 2))
    denom1 = numpy.sum(modelst*psfst*weightst, axis=(1, 2))
    denom2 = numpy.sum(modelst*psfst, axis=(1, 2))
    denom3 = numpy.sum(psf*psf)
    for (dc, off) in [(dx, xf), (dy, yf)]:
        # the centroids
        numer0 = numpy.sum(
            dc[None, :, :]*(modelst+residst)*psfst*weightst, axis=(1, 2))
        # difference between 1 & 2 is bias due to weights
        numer1 = numpy.sum(dc[None, :, :]*modelst*psfst*weightst,
                           axis=(1, 2))
        numer2 = numpy.sum(dc[None, :, :]*modelst*psfst,
                           axis=(1, 2))
        # bias for asymmetric PSFs
        numer3 = numpy.sum(dc*psf*psf)
        c0 = numer0 / (denom0 + (denom0 == 0))
        c1 = numer1 / (denom1 + (denom1 == 0))
        c2 = numer2 / (denom2 + (denom2 == 0))
        c3 = numer3 / (denom3 + (denom3 == 0))
        cen.append(c0 - off - (c1 - c2) - c3)
    xcen, ycen = cen
    xcen *= 2
    ycen *= 2
    res = (xcen, ycen)
    if return_psfstack:
        res = res + ((modelst+residst, residst, weightst),)
    return res


def estimate_sky_background(im, sdev=None):
    """Find peak of count distribution; pretend this is the sky background."""
    # for some reason, I have found this hard to work robustly.  Replace with
    # median at the moment.

    return numpy.median(im)
    # from scipy.stats.mstats import mquantiles
    # from scipy.special import erf
    # q05, med, q7 = mquantiles(im, prob=[0.05, 0.5, 0.7])
    # if sdev is None:
    #     if med > 0:
    #         sdev = 1.5*numpy.sqrt(med/4.)
    #     else:
    #         q16, q84 = mquantiles(im, prob=[0.16, 0.84])
    #         sdev = (q84-q16)/2.
    #     # appropriate for DECam images; gain ~ 4
    # lb, ub = q05, q7

    # def objective(par, lb, ub):
    #     mask = (im > lb) & (im < ub)
    #     nobs = numpy.sum(mask)
    #     chi = (im[mask]-par)/sdev
    #     norm = 0.5*(erf((ub-par)/(numpy.sqrt(2)*sdev)) -
    #                 erf((lb-par)/(numpy.sqrt(2)*sdev)))
    #     if norm <= 1.e-6:
    #         normchi2 = 1e10
    #     else:
    #         normchi2 = 2*nobs*numpy.log(norm)
    #     return numpy.sum(chi**2.)+normchi2

    # from scipy.optimize import minimize_scalar
    # par = minimize_scalar(objective, bracket=[lb, ub], args=(lb, ub))
    # lb, ub = par['x']-sdev, par['x']+sdev
    # par = minimize_scalar(objective, bracket=[lb, ub], args=(lb, ub))
    # pdb.set_trace()
    # return par['x']


def sky_im(im, weight=None, npix=20):
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
            if usedpix[i, j] > 100:
                val[i, j] = estimate_sky_background(
                    im[xg[i]:xg[i+1], yg[j]:yg[j+1]][use].reshape(-1))
            else:
                usedpix[i, j] = 0
    from scipy.ndimage.filters import gaussian_filter
    count = 0
    while numpy.any(usedpix == 0):
        sig = 1.
        valc = gaussian_filter(val, sig, mode='constant')
        weightc = gaussian_filter((usedpix != 0).astype('f4'), sig,
                                  mode='constant')
        m = (usedpix == 0) & (weightc > 1.e-10)
        val[m] = valc[m]/weightc[m]
        usedpix[m] = 1
        count += 1
        if count > 10:
            m = usedpix == 0
            val[:, :] = numpy.median(im)
            print('Sky estimation failed badly.')
            break
    if numpy.any(val == 0):
        pdb.set_trace()
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
    bg = map_coordinates(val, coord, mode='nearest', order=1).reshape(im.shape)
    if numpy.any(bg == 0):
        pdb.set_trace()
    return bg


def get_sizes(x, y, imbs, psf, weight=None):
    x = numpy.round(x).astype('i4')
    y = numpy.round(y).astype('i4')
    peakbright = imbs[x, y]
    sz = numpy.zeros(len(x), dtype='i4')
    cutoff = 1500
    sz[peakbright > cutoff] = psf.shape[0]
    sz[peakbright <= cutoff] = 19  # for the moment...
    if weight is not None:
        sz[weight[x, y] == 0] = psf.shape[0]  # saturated sources get big PSFs
    return sz


def subtract_satstars(model, weight, x, y, flux, psf, psfderiv=None):
    xp, yp = (numpy.round(c).astype('i4') for c in (x, y))
    m = weight[xp, yp] == 0
    if psfderiv is None:
        bflux = flux[:len(x)][m]
    else:
        repeat = 3
        ind = numpy.flatnonzero(m)
        bflux = numpy.concatenate([flux[repeat*i:repeat*(i+1)] for i in ind])
    bmodel = build_model(x[m], y[m], bflux, model.shape[0], model.shape[1],
                         psf=psf, psfderiv=psfderiv)
    return model-bmodel


def fit_im(im, psf, threshhold=0.3, weight=None, psfderiv=None,
           nskyx=0, nskyy=0, refit_psf=False, fixedstars=None,
           verbose=False):
    if fixedstars is not None and len(fixedstars['x']) > 0:
        psflist = {'stamp': fixedstars['stamp'], 'ind': fixedstars['psf']}
        fixedmodel = build_model(fixedstars['x'], fixedstars['y'],
                                 fixedstars['flux'], im.shape[0], im.shape[1],
                                 psflist=psflist)
        im = im.copy()-fixedmodel
    else:
        fixedmodel = numpy.zeros_like(im)
    sky = sky_im(im, weight=weight)
    # if (nskyx != 0) or (nskyy != 0):
    #
    # else:
    #     imbs = im
    if isinstance(weight, int):
        weight = numpy.ones_like(im)*weight
    sigim = significance_image(im-sky, weight, psf)
    # because of inaccuracy of initial sky subtraction, this is guaranteed to
    # go less deep than five sigma.
    x, y = peakfind(im-sky, sigim, weight, keepsat=True)

    if verbose:
        print('Found %d initial sources.' % len(x))

    # first fit; brightest sources
    sz = get_sizes(x, y, im-sky, psf, weight=weight)
    flux, model, sky = fit_once(im-sky, x, y, psf, psfderiv=psfderiv, sz=sz,
                                weight=weight, nskyx=nskyx, nskyy=nskyy)
    sky = sky_im(im-model, weight=weight)
    model[:, :] = subtract_satstars(model, weight, x, y, flux[0],
                                    psf, psfderiv=psfderiv)
    sigim_res = significance_image(im - model - sky, weight, psf)
    x2, y2 = peakfind(im - model - sky, sigim_res, weight)
    # should reach 5 sigma.

    # accept only peaks where im / numpy.clip(model, threshhold) > 0.3
    sigstat1, sigstat2 = peaksig(x2, y2, im-model-sky, model+fixedmodel, psf)

    # m = (sigstat1 > threshhold) & (sigstat2 > threshhold)
    m = (sigstat1 > 1) & (sigstat2 > 1)
    # in first pass, where the PSF might be bad, be really aggressive
    # about not splitting bright stars into multiple components.
    x2, y2 = x2[m], y2[m]
    if verbose:
        print('Found %d additional sources.' % len(x2))
    x = numpy.concatenate([x, x2])
    y = numpy.concatenate([y, y2])

    # now fit sources that may have been too blended to detect initially
    guessflux, guesssky = unpack_fitpar(flux[0], len(x)-len(x2),
                                        psfderiv is not None)
    guess = numpy.concatenate([guessflux, numpy.zeros(len(x2)), guesssky])
    sz = get_sizes(x, y, im-sky, psf, weight=weight)
    flux, model, sky = fit_once(im-sky, x, y, psf, psfderiv=psfderiv,
                                weight=weight, nskyx=nskyx, nskyy=nskyy,
                                guess=guess, sz=sz)
    sky = sky_im(im-model, weight=weight)
    centroids = compute_centroids(x, y, psf, flux[0], im-model-sky,
                                  weight, psfderiv=psfderiv,
                                  return_psfstack=refit_psf)
    xcen, ycen = centroids[0], centroids[1]
    if refit_psf:
        stamps = centroids[2]
        shiftx = xcen + x - numpy.round(x)
        shifty = ycen + y - numpy.round(y)
        npsf = find_psf(x, shiftx, y, shifty, stamps[0], stamps[1], stamps[2],
                        psf)
        if npsf is not None:
            psf = npsf
            if psfderiv is not None:
                psfderiv = numpy.gradient(-psf)
    model[:, :] = subtract_satstars(model, weight, x, y, flux[0],
                                    psf, psfderiv=psfderiv)
    m = (numpy.abs(xcen) < 3) & (numpy.abs(ycen) < 3)
    x = x.astype('f4')
    y = y.astype('f4')
    x[m] += xcen[m]
    y[m] += ycen[m]
    x[m] = numpy.clip(x[m], 0, im.shape[0]-1)
    y[m] = numpy.clip(y[m], 0, im.shape[1]-1)
    guessflux, guesssky = unpack_fitpar(flux[0], len(x), psfderiv is not None)
    keep = cull_near(x, y, guessflux) & (guessflux > 0) & m
    x = x[keep]
    y = y[keep]

    sigim_res = significance_image(im - model - sky, weight, psf)
    x3, y3 = peakfind(im - model - sky, sigim_res, weight)

    sigstat1, sigstat2 = peaksig(x3, y3, im-model-sky, model+fixedmodel, psf)
    m = (sigstat1 > threshhold) | (sigstat2 > threshhold)
    x3, y3 = x3[m], y3[m]
    if verbose:
        print('Found %d additional sources.' % len(x3))
    x = numpy.concatenate([x, x3])
    y = numpy.concatenate([y, y3])

    # now fit with improved locations
    guessflux, guesssky = unpack_fitpar(flux[0], len(keep),
                                        psfderiv is not None)
    guess = numpy.concatenate([guessflux[keep], numpy.zeros(len(x3)),
                               guesssky])
    sz = get_sizes(x, y, im-sky, psf, weight=weight)
    flux, model, sky = fit_once(im-sky, x, y, psf, psfderiv=psfderiv,
                                weight=weight, nskyx=nskyx, nskyy=nskyy,
                                guess=guess, sz=sz)
    sky = sky_im(im-model, weight=weight)
    xcen, ycen = compute_centroids(x, y, psf, flux[0], im-model-sky, weight,
                                   psfderiv=psfderiv)
    m = (numpy.abs(xcen) < 3) & (numpy.abs(ycen) < 3)
    x = x.astype('f4')
    y = y.astype('f4')
    x[m] += xcen[m]
    y[m] += ycen[m]
    x[m] = numpy.clip(x[m], 0, im.shape[0]-1)
    y[m] = numpy.clip(y[m], 0, im.shape[1]-1)
    guessflux, guesssky = unpack_fitpar(flux[0], len(x), psfderiv is not None)
    keep = cull_near(x, y, guessflux) & (guessflux > 0) & m
    x = x[keep]
    y = y[keep]

    # final fit with final locations; no psfderiv allowed, positions are fixed.
    guess = numpy.concatenate([guessflux[keep], guesssky])
    flux, model, _ = fit_once(im-sky, x, y, psf, psfderiv=None,
                              weight=weight, nskyx=nskyx, nskyy=nskyy,
                              guess=guess)
    if fixedmodel is not None:
        model += fixedmodel
    flux, skypar = unpack_fitpar(flux[0], len(x), False)
    res = (x, y, flux, skypar, model+sky, sky, psf)
    return res


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
    m1, m2, dist = match_xy(x, y, x, y, neighbors=4)
    m = (dist < 1.5) & (flux[m1] < flux[m2]) & (m1 != m2)
    keep = numpy.ones(len(x), dtype='bool')
    keep[m1[m]] = 0
    keep[flux < 0] = 0
    return keep


def match_xy(x1, y1, x2, y2, neighbors=1):
    """Match x1 & y1 to x2 & y2, neighbors nearest neighbors."""
    from scipy.spatial import cKDTree
    vec1 = numpy.array([x1, y1]).T
    vec2 = numpy.array([x2, y2]).T
    swap = len(vec1) > len(vec2)
    if swap:
        vec1, vec2 = vec2, vec1
    kdt = cKDTree(vec1)
    dist, idx = kdt.query(vec2, neighbors)
    m1 = numpy.repeat(numpy.arange(len(vec1), dtype='i4'), neighbors)
    m2 = idx.ravel()
    m = (m2 >= 0) & (m2 < len(vec1))
    m1 = m1[m]
    m2 = m2[m]
    dist = dist.ravel()
    dist = dist[m]
    if swap:
        m1, m2 = m2, m1
    return m1, m2, dist


def gaussian_psf(fwhm, stampsz=19):
    """Create Gaussian psf & derivatives for a given fwhm and stamp size.

    Args:
        fwhm (float): the full width at half maximum
        stampsz (int): the return psf stamps are [stampsz, stampsz] in size

    Returns:
        (psf, dpsfdx, dpsfdy)
        psf (ndarray[stampsz, stampsz]): the psf stamp
        dpsfdx (ndarray[stampsz, stampsz]): the x-derivative of the PSF
        dpsfdy (ndarray[stampsz, stampsz]): the y-derivative of the PSF
    """
    sigma = fwhm / numpy.sqrt(8*numpy.log(2))
    stampszo2 = stampsz // 2
    xc = numpy.arange(stampsz, dtype='f4')-stampszo2
    yc = xc.copy()
    psf = numpy.exp(-(xc.reshape(-1, 1)**2. + yc.reshape(1, -1)**2.) /
                    2./sigma**2.).astype('f4')
    psf /= numpy.sum(psf)
    dpsfdx = xc.reshape(-1, 1)/sigma**2.*psf
    dpsfdy = yc.reshape(1, -1)/sigma**2.*psf
    return psf, dpsfdx, dpsfdy


def moffat_psf(fwhm, beta=3., xy=0., yy=1., stampsz=19, deriv=True):
    """Create Moffat psf & derivatives for a given fwhm and stamp size.

    Args:
        fwhm (float): the full width at half maximum
        stampsz (int): the returned psf stamps are [stampsz, stampsz] in size
        beta (float): beta parameter for Moffat distribution

    Returns:
        (psf, dpsfdx, dpsfdy)
        psf (ndarray[stampsz, stampsz]): the psf stamp
        dpsfdx (ndarray[stampsz, stampsz]): the x-derivative of the PSF
        dpsfdy (ndarray[stampsz, stampsz]): the y-derivative of the PSF
    """
    alpha = fwhm/(2*numpy.sqrt(2**(1./beta)-1))
    stampszo2 = stampsz // 2
    xc = numpy.arange(stampsz, dtype='f4')-stampszo2
    xc = xc.reshape(-1, 1)
    yc = xc.copy().reshape(1, -1)
    rc = numpy.sqrt(xc**2. + xy*xc*yc + yy*yc**2.)
    psf = (beta - 1)/(numpy.pi * alpha**2.)*(1.+(rc**2./alpha**2.))**(-beta)
    ret = psf
    if deriv:
        dpsffac = (beta-1)/(numpy.pi*alpha**2.)*(beta)*(
            (1+(rc**2./alpha**2.))**(-beta-1))
        dpsfdx = dpsffac*2*xc/alpha
        dpsfdy = dpsffac*2*yc/alpha
        ret = (psf, dpsfdx, dpsfdy)
    return ret


def center_psf(psf):
    """Center and normalize a psf; centroid is placed at center."""
    cpsf = central_stamp(psf)
    stampsz = cpsf.shape[0]
    stampszo2 = stampsz // 2
    xc = numpy.arange(stampsz, dtype='f4')-stampszo2
    xc *= numpy.abs(xc) <= 9  # only use the inner 19 pixels for centroiding
    xc = xc.reshape(-1, 1)
    yc = xc.copy().reshape(1, -1)
    for _ in range(3):
        xcen = numpy.sum(xc*cpsf)/numpy.sum(cpsf)
        ycen = numpy.sum(yc*cpsf)/numpy.sum(cpsf)
        psf[:, :] = shift(psf, [-xcen, -ycen], output=numpy.dtype('f4'))
    psf /= numpy.sum(psf)
    psf = psf.astype('f4')
    return psf


def find_psf(xcen, shiftx, ycen, shifty, psfstack, residstack, weightstack,
             psf):
    """Find PSF from stamps."""
    # let's just go ahead and correlate the noise
    xr = numpy.round(shiftx)
    yr = numpy.round(shifty)
    psfqf = (numpy.sum(psfstack*(weightstack > 0), axis=(1, 2)) /
             numpy.sum(psfstack, axis=(1, 2)))
    totalflux = numpy.sum(psfstack, axis=(1, 2))
    chi2 = numpy.sum(residstack**2.*((weightstack+1.e-10)**-2. +
                                     (0.2*psfstack)**2.)**-1.,
                     axis=(1, 2))
    okpsf = ((numpy.abs(psfqf - 1) < 0.03) & (totalflux > 10000) &
             (numpy.abs(shiftx) < 2) & (numpy.abs(shifty) < 2))  # &
             # (chi2 < 1.5*len(psfstack[0].reshape(-1))))
    if numpy.sum(okpsf) <= 5:
        print('Fewer than 5 stars accepted in image, keeping original PSF')
        return None
    pdb.set_trace()
    psfstack = psfstack[okpsf, :, :]
    weightstack = weightstack[okpsf, :, :]
    residstack = residstack[okpsf, :, :]
    totalflux = totalflux[okpsf]
    xcen = xcen[okpsf]
    ycen = ycen[okpsf]
    shiftx = shiftx[okpsf]
    shifty = shifty[okpsf]
    for i in range(psfstack.shape[0]):
        psfstack[i, :, :] = shift(psfstack[i, :, :], [-shiftx[i], -shifty[i]])
        if (numpy.abs(xr[i]) > 0) or (numpy.abs(yr[i]) > 0):
            residstack[i, :, :] = shift(residstack[i, :, :], [-xr[i], -yr[i]])
            weightstack[i, :, :] = shift(residstack[i, :, :], [-xr[i], -yr[i]],
                                         mode='constant', cval=0.)
        # our best guess as to the PSFs & their weights
    # select some reasonable sample of the PSFs
    totalflux = numpy.sum(psfstack, axis=(1, 2))
    psfstack /= totalflux.reshape(-1, 1, 1)
    weightstack *= totalflux.reshape(-1, 1, 1)
    tpsf = numpy.median(psfstack, axis=0)
    tpsf = center_psf(tpsf)
    if tpsf.shape == psf.shape:
        return tpsf
    xc = numpy.arange(tpsf.shape[0]).reshape(-1, 1)-tpsf.shape[0]//2
    yc = xc.reshape(1, -1)
    rc = numpy.sqrt(xc**2.+yc**2.)
    stampszo2 = psfstack[0].shape[0] // 2
    wt = numpy.clip((stampszo2+1-rc)/4., 0., 1.)
    overlap = (wt != 1) & (wt != 0)

    def objective(par):
        mod = moffat_psf(par[0], beta=2.5, xy=par[2], yy=par[3],
                         deriv=False, stampsz=tpsf.shape[0])
        mod /= numpy.sum(mod)
        return ((tpsf-mod)[overlap]).reshape(-1)
    from scipy.optimize import leastsq
    par = leastsq(objective, [4., 3., 0., 1.])[0]
    modpsf = moffat_psf(par[0], beta=2.5, xy=par[2], yy=par[3], deriv=False,
                        stampsz=psf.shape[0])
    modpsf /= numpy.sum(modpsf)
    npsf = modpsf.copy()
    npsfcen = central_stamp(npsf, tpsf.shape[0])
    npsfcen[:, :] = tpsf*wt+(1-wt)*npsfcen[:, :]
    return npsf


sample = """
sample code:

psf, dpsfdx, dpsfdy = crowdsource.gaussian_psf(fwhm, 19)
reload(crowdsource) ; im, xt, yt, fluxt = crowdsource.sim_image(1000, 1000, 30000, 5., 3.) ; clf() ; util_efs.imshow(im, arange(1000), arange(1000), vmax=20, aspect='equal') ; xlim(0, 300) ; ylim(0, 300)
reload(crowdsource) ; sigim = crowdsource.significance_image(im, im*0+3.**2., psf)
reload(crowdsource) ; xydat = crowdsource.peakfind(im, sigim, 3.)
reload(crowdsource) ;  x2, y2, flux2, model2 = crowdsource.fit_im(im, psf, 3., 0.3, psfderiv=[dpsfdx, dpsfdy])
"""
