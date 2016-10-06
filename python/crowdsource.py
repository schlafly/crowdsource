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
        kw['order'] = 3 # ~0.6 mmag; order=4 ~0.2 mmag
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


def significance_image(im, ivar, psf):
    # assume, for the moment, the image has already been sky-subtracted
    # stampszo2 = (psf.shape[0]-1)/2
    from scipy.ndimage.filters import convolve
    sigim = convolve(im*ivar, psf, mode='reflect', origin=0)
    varim = convolve(ivar, psf**2., mode='reflect', origin=0)
    ivarim = 1./(varim + (varim == 0) * 1e12)
    return sigim * numpy.sqrt(ivarim)


def peakfind(im, sigim, weight, threshhold=5):
    fac = 0.4
    data_max = filters.maximum_filter(im*numpy.sqrt(weight), 3)
    sig_max = filters.maximum_filter(sigim, 3)
    brightpeak = numpy.nonzero((data_max == im*numpy.sqrt(weight)) &
                               (im*numpy.sqrt(weight) > threshhold/fac))
    faintpeak = numpy.nonzero((sig_max == sigim) & (sigim > threshhold) &
                              (im*numpy.sqrt(weight) <= threshhold/fac))
    return [numpy.concatenate([b, f]) for b, f in zip(brightpeak, faintpeak)]
    # return numpy.nonzero((data_max == im) & (im > 0) & (sigim > threshhold))


def build_model(x, y, flux, nx, ny, psf=None, psflist=None):
    if psf is None and psflist is None:
        raise ValueError('One of psf and psflist must be set')
    if psf is not None and psflist is not None:
        raise ValueError('Only one of psf and psflist must be set')
    if psf is not None:
        psflist = {'stamp': [psf], 'ind': [numpy.zeros(len(x), dtype='i4')]}
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
    for i in range(len(x)):
        psf = psflist['stamp'][psflist['ind'][i]]
        im[xe[i]:xe[i]+stampsz, ye[i]:ye[i]+stampsz] += (
            shift(psf, [xf[i], yf[i]], output=numpy.dtype('f4'))*flux[i])
    im = im[stampszo2:-stampszo2, stampszo2:-stampszo2]
    return im


def in_padded_region(flatcoord, imshape, pad):
    coord = numpy.unravel_index(flatcoord, imshape)
    m = numpy.zeros(len(flatcoord), dtype='bool')
    for c, length in zip(coord, imshape):
        m |= (c < pad) | (c >= length - pad)
    return m


def fit_once(im, x, y, psf, weight=None, psfderiv=None, nskyx=0, nskyy=0,
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
        nskyy (int): numpy. of sky pixels in y direction (0 or >= 3)

    Returns:
        tuple(flux, model, sky)
        flux: output of optimization routine; needs to be refined
        model (ndarray[NX, NY]): best fit model image
        sky (ndarray(NX, NY]): best fit model sky
    """
    guess = None
    # sparse matrix, with rows at first equal to the fluxes at each peak
    # later add in the derivatives at each peak
    stampsz = psf.shape[0]
    stampszo2 = int(numpy.ceil(stampsz/2.)-1)
    nx, ny = im.shape
    im = numpy.pad(im, [stampszo2, stampszo2], constant_values=0.,
                   mode='constant')
    if weight is None:
        weight = numpy.ones_like(im)
    weight = numpy.pad(weight, [stampszo2, stampszo2], constant_values=0.,
                       mode='constant')
    pix = numpy.arange(stampsz*stampsz, dtype='i4')
    # convention: x is the first index, y is the second
    # sorry.
    xpix = pix // stampsz
    ypix = pix % stampsz
    yloc, xloc, values = [], [], []
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
    for i in range(len(xe)):
        for _ in range(repeat):
            xloc.append(numpy.ravel_multi_index(((xe[i]+xpix), (ye[i]+ypix)),
                                                im.shape))
    xloc = numpy.concatenate(xloc)
    yloc = (numpy.arange(len(x)*repeat, dtype='i4').reshape(-1, 1) *
            numpy.ones(len(xpix), dtype='i4').reshape(1, -1)).ravel()
    valarr = []
    for i in xrange(len(xe)):
        wt = weight[xe[i]:xe[i]+stampsz, ye[i]:ye[i]+stampsz]
        valarr += [(shift(psf, [xf[i], yf[i]],
                          output=numpy.dtype('f4'))*wt).ravel()]
        if psfderiv:
            valarr += [(p*wt).ravel() for p in psfderiv]
    values = numpy.concatenate(valarr)
    nskypar = nskyx * nskyy
    if nskypar != 0:
        xloc, yloc, values = add_sky_parameters(nx+stampszo2*2, ny+stampszo2*2,
                                                nskyx, nskyy,
                                                xloc, yloc, values, weight)
    shape = (im.shape[0]*im.shape[1], len(x)*repeat+nskypar)
    mdelete = in_padded_region(xloc, im.shape, stampszo2)
    xloc = xloc[~mdelete]
    yloc = yloc[~mdelete]
    values = values[~mdelete]
    from scipy import sparse
    mat = sparse.coo_matrix((values, (xloc, yloc)), shape=shape,
                            dtype='f4')
    if guess is not None:
        # guess is a guess for the fluxes and sky; no derivatives.
        guessvec = numpy.zeros(len(xe)*repeat+nskypar, dtype='f4')
        guessvec[0:len(xe)*repeat:repeat] = guess[0:len(xe)]
        guessvec[len(xe)*repeat:] = guess[len(xe):]
    else:
        guessvec = None
    flux = lsqr_cp(mat, (im*weight).ravel(), atol=1.e-3, btol=1.e-3,
                   guess=guessvec)
    model = mat.dot(flux[0]).reshape(*im.shape)
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
    from scipy.sparse import diags, linalg

    # column preconditioning:
    aacsc = aa.tocsc().copy()
    norm = numpy.array(aacsc.multiply(aacsc).sum(axis=0)).squeeze()
    norm = norm + (norm == 0)
    aacsc = aacsc.dot(diags(norm**(-0.5)))
    # aacsc has columns of constant norm.

    bb2 = bb.copy()
    normbb = numpy.sum(bb2**2.)
    bb2 /= normbb**(0.5)
    # allow guesses:
    if guess is not None:
        bb2 = (bb2 - aa.dot(guess)/normbb**(0.5)).astype('f4')
        if 'btol' in kw:
            kw['btol'] = kw['btol']*normbb**(0.5)/numpy.sum(bb2**2.)**(0.5)
            print kw['btol']
            kw['btol'] = numpy.min([kw['btol'], 0.1])
    par = linalg.lsqr(aacsc, bb2, **kw)
    # or lsmr; lsqr seems to be better
    par[0][:] *= norm**(-0.5)*normbb**(0.5)
    if guess is not None:
        par[0][:] += guess
    return par


def compute_centroids(x, y, psf, flux, resid, weight, psfderiv=None,
                      return_psfstack=False):
    # centroid is sum(x * weight) / sum(weight)
    # we'll make the weights = psf * image
    # we want to compute the centroids on the image after the other sources
    # have been subtracted off.
    # we construct this image by taking the residual image, and then
    # star-by-star adding the model back.
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
    if return_psfstack:
        psfstack = []
        residstack = []
        weightstack = []
    for i in range(len(x)):
        # way more complicated than I'd like.
        # we just want the weighted centroids.
        # these are sum(x * I * W) / sum(I * W)
        # if we use the PSF model for W, it turns out these are always?
        # biased too small by exactly? one half.
        # So that's straightforward.
        # It turns out that if the PSF is asymmetric, these are further biased
        # by a constant; we subtract that with x3 and y3 below.  In principle
        # those don't need to be in the loop (though if the PSF varies that's
        # another story.
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

        if (xp[i] != x[i]) or (yp[i] != y[i]):
            offset = [x[i]-xp[i], y[i]-yp[i]]
            psf0 = shift(psf, offset, output=numpy.dtype('f4'))
            if psfderiv is None:
                psfderiv0 = None
            else:
                # psfderiv0 = [shift(p, offset, output=numpy.dtype('f4'))
                #             for p in psfderiv]
                psfderiv0 = psfderiv
        else:
            psf0 = psf
            psfderiv0 = psfderiv
            offset = [0., 0.]
        wstamp = weight[xe[i]:xe[i]+stampsz, ye[i]:ye[i]+stampsz]
        imr = resid[xe[i]:xe[i]+stampsz, ye[i]:ye[i]+stampsz]
        imm = psf0*flux[repeat*i]
        if psfderiv:
            imm += psfderiv0[0]*flux[repeat*i+1]+psfderiv0[1]*flux[repeat*i+2]
        im0 = (imm + imr) * psf0 * wstamp
        im1 = imm * psf0 * wstamp
        im2 = imm * psf0
        im3 = psf0 * psf0
        sim0 = numpy.sum(im0)
        sim1 = numpy.sum(im1)
        sim2 = numpy.sum(im2)
        sim3 = numpy.sum(im3)
        sim0 = sim0 + (sim0 == 0)*1e-9
        sim1 = sim1 + (sim1 == 0)*1e-9
        sim2 = sim2 + (sim2 == 0)*1e-9
        sim3 = sim3 + (sim3 == 0)*1e-9
        x1 = numpy.sum(dx*im1)/sim1
        y1 = numpy.sum(dy*im1)/sim1
        x2 = numpy.sum(dx*im2)/sim2
        y2 = numpy.sum(dy*im2)/sim2
        x3 = numpy.sum(dx*im3)/sim3
        y3 = numpy.sum(dy*im3)/sim3
        xcen[i] = numpy.sum(dx*im0)/sim0-offset[0]-(x1-x2)-(x3-offset[0])
        ycen[i] = numpy.sum(dy*im0)/sim0-offset[1]-(y1-y2)-(y3-offset[1])
        if return_psfstack:
            psfstack.append((imm+imr).copy())
            residstack.append(imr.copy())
            weightstack.append(wstamp.copy())
        # if (xe[i] > 100) and (ye[i] > 100) and (flux[repeat*i] > 10000):
        #     pdb.set_trace()
    xcen *= 2
    ycen *= 2
    res = (xcen, ycen)
    if return_psfstack:
        res = res + ((numpy.array(psfstack), numpy.array(residstack),
                      numpy.array(weightstack)),)
    return res


def estimate_sky_background(im, sdev=None):
    """Find peak of count distribution; pretend this is the sky background."""
    from scipy.stats.mstats import mquantiles
    from scipy.special import erf
    q05, med, q7 = mquantiles(im, prob=[0.05, 0.5, 0.7])
    if sdev is None:
        sdev = 1.5*numpy.sqrt(med/4.)
        # appropriate for DECam images; gain ~ 4
    lb, ub = q05, q7

    def objective(par, lb, ub):
        mask = (im > lb) & (im < ub)
        nobs = numpy.sum(mask)
        chi = (im[mask]-par)/sdev
        norm = 0.5*(erf((ub-par)/(numpy.sqrt(2)*sdev)) -
                    erf((lb-par)/(numpy.sqrt(2)*sdev)))
        if norm <= 1.e-6:
            normchi2 = 1e10
        else:
            normchi2 = 2*nobs*numpy.log(norm)
        return numpy.sum(chi**2.)+normchi2

    from scipy.optimize import minimize_scalar
    par = minimize_scalar(objective, bracket=[lb, ub], args=(lb, ub))
    lb, ub = par['x']-sdev/2., par['x']+sdev/2.
    par = minimize_scalar(objective, bracket=[lb, ub], args=(lb, ub))
    return par['x']


def initial_sky_filter(im, sz=100):
    """Remove sky from image."""
    nx, ny = im.shape[0]/sz, im.shape[1]/sz
    xg = -sz/2. + sz*numpy.arange(nx+2)
    yg = -sz/2. + sz*numpy.arange(ny+2)
    val = numpy.zeros((nx+2, ny+2), dtype='f4')
    # annoying!
    for i in range(nx):
        for j in range(ny):
            sx, sy = i*sz, j*sz
            ex = numpy.min([sx+sz, im.shape[0]])
            ey = numpy.min([sy+sz, im.shape[1]])
            val[i+1, j+1] = (
                estimate_sky_background(im[sx:ex, sy:ey].reshape(-1)))
    # I can't figure out how to ask for any kind of reasonable extrapolation
    # off the grid.  I'm going to extend xg, yg, val to compensate.
    val[0, :] = val[1, :]-(val[2, :]-val[1, :])
    val[:, 0] = val[:, 1]-(val[:, 2]-val[:, 1])
    val[-1, :] = val[-2, :]+(val[-2, :]-val[-3, :])
    val[:, -1] = val[:, -2]+(val[:, -2]-val[:, -3])
    # this gets the corners since they're done both times
    x = numpy.arange(im.shape[0])
    y = numpy.arange(im.shape[1])
    from scipy.ndimage import map_coordinates
    xp = numpy.interp(x, xg, numpy.arange(len(xg), dtype='f4'))
    yp = numpy.interp(y, yg, numpy.arange(len(yg), dtype='f4'))
    xpa = xp.reshape(-1, 1)*numpy.ones(len(yp)).reshape(1, -1)
    ypa = yp.reshape(1, -1)*numpy.ones(len(xp)).reshape(-1, 1)
    coord = [xpa.ravel(), ypa.ravel()]
    bg = map_coordinates(val, coord, mode='nearest', order=1).reshape(im.shape)
    return im-bg


def fit_im(im, psf, threshhold=0.3, weight=None, psfderiv=None,
           nskyx=3, nskyy=3, refit_psf=False, fixedstars=None,
           verbose=False):
    if fixedstars is not None and len(fixedstars['x']) > 0:
        psflist = {'stamp': fixedstars['stamp'], 'ind': fixedstars['psf']}
        fixedmodel = build_model(fixedstars['x'], fixedstars['y'],
                                 fixedstars['flux'], im.shape[0], im.shape[1],
                                 psflist=psflist)
        im = im.copy()-fixedmodel
    else:
        fixedmodel = None
    if (nskyx != 0) or (nskyy != 0):
        imbs = initial_sky_filter(im)
    else:
        imbs = im
    if isinstance(weight, int):
        weight = numpy.ones_like(im)*weight
    sigim = significance_image(imbs, weight, psf)
    x, y = peakfind(imbs, sigim, weight)
    if verbose:
        print('Found %d initial sources.' % len(x))
    # pdb.set_trace()

    # first fit; brightest sources
    flux, model, sky = fit_once(im, x, y, psf, psfderiv=psfderiv,
                                weight=weight, nskyx=nskyx, nskyy=nskyy)
    sigim_res = significance_image(im - model, weight, psf)
    x2, y2 = peakfind(im - model, sigim_res, weight)

    # accept only peaks where im / numpy.clip(model, threshhold) > 0.3
    sigstat = ((im-model)/numpy.clip(model-sky, 0.3, numpy.inf))[x2, y2]
    m = sigstat > threshhold
    x2, y2 = x2[m], y2[m]
    if verbose:
        print('Found %d additional sources.' % len(x2))
    x = numpy.concatenate([x, x2])
    y = numpy.concatenate([y, y2])

    # now fit sources that may have been too blended to detect initially
    # pdb.set_trace()
    guessflux, guesssky = unpack_fitpar(flux[0], len(x)-len(x2),
                                        psfderiv is not None)
    guess = numpy.concatenate([guessflux, numpy.zeros(len(x2)), guesssky])
    flux, model, sky = fit_once(im, x, y, psf, psfderiv=psfderiv,
                                weight=weight, nskyx=nskyx, nskyy=nskyy,
                                guess=guess)
    xcen, ycen, stamps = compute_centroids(x, y, psf, flux[0], im-model,
                                           weight, psfderiv=psfderiv,
                                           return_psfstack=True)
    if refit_psf:
        shiftx = xcen + x - numpy.round(x)
        shifty = ycen + y - numpy.round(y)
        psf = find_psf(x, shiftx, y, shifty, stamps[0], stamps[1], stamps[2])
        if psfderiv is not None:
            psfderiv = numpy.gradient(-psf)
    # pdb.set_trace()
    m = (numpy.abs(xcen) < 3) & (numpy.abs(ycen) < 3)
    x = x.astype('f4')
    y = y.astype('f4')
    x[m] += xcen[m]
    y[m] += ycen[m]
    x[m] = numpy.clip(x[m], 0, im.shape[0]-1)
    y[m] = numpy.clip(y[m], 0, im.shape[1]-1)
    guessflux, guesssky = unpack_fitpar(flux[0], len(x), psfderiv is not None)
    keep = cull_near(x, y, guessflux)
    x = x[keep]
    y = y[keep]

    sigim_res = significance_image(im - model, weight, psf)
    x3, y3 = peakfind(im - model, sigim_res, weight)

    # accept only peaks where im / numpy.clip(model, threshhold) > 0.3
    sigstat = ((im-model)/numpy.clip(model-sky, 0.3, numpy.inf))[x3, y3]
    m = sigstat > threshhold
    x3, y3 = x3[m], y3[m]
    if verbose:
        print('Found %d additional sources.' % len(x3))
    x = numpy.concatenate([x, x3])
    y = numpy.concatenate([y, y3])

    # now fit with improved locations
    # pdb.set_trace()
    guessflux, guesssky = unpack_fitpar(flux[0], len(keep),
                                        psfderiv is not None)
    guess = numpy.concatenate([guessflux[keep], numpy.zeros(len(x3)),
                               guesssky])
    flux, model, sky = fit_once(im, x, y, psf, psfderiv=psfderiv,
                                weight=weight, nskyx=nskyx, nskyy=nskyy,
                                guess=guess)
    xcen, ycen = compute_centroids(x, y, psf, flux[0], im-model, weight,
                                   psfderiv=psfderiv)
    # pdb.set_trace()
    m = (numpy.abs(xcen) < 3) & (numpy.abs(ycen) < 3)
    x = x.astype('f4')
    y = y.astype('f4')
    x[m] += xcen[m]
    y[m] += ycen[m]
    x[m] = numpy.clip(x[m], 0, im.shape[0]-1)
    y[m] = numpy.clip(y[m], 0, im.shape[1]-1)
    guessflux, guesssky = unpack_fitpar(flux[0], len(x), psfderiv is not None)
    keep = cull_near(x, y, guessflux)
    x = x[keep]
    y = y[keep]

    # final fit with final locations; no psfderiv allowed, positions are fixed.
    # pdb.set_trace()
    guessflux, guesssky = unpack_fitpar(flux[0], len(keep),
                                        psfderiv is not None)
    guess = numpy.concatenate([guessflux[keep], guesssky])
    flux, model, sky = fit_once(im, x, y, psf, psfderiv=None,
                                weight=weight, nskyx=nskyx, nskyy=nskyy,
                                guess=guess)
    if fixedmodel is not None:
        model += fixedmodel
    flux, skypar = unpack_fitpar(flux[0], len(x), False)
    res = (x, y, flux, skypar, model, sky)
    if refit_psf:
        res = res + (psf,)
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
    if (coeff.shape[0] < 3) or (coeff.shape[1]) < 3:
        raise ValueError('Not obvious what to do for <3')
    im = numpy.zeros((nx, ny), dtype='f4')
    for i in range(coeff.shape[0]):
        for j in range(coeff.shape[1]):
            # missing here: speed up available from knowing that
            # the basisspline is zero over a large area.
            im += coeff[i, j] * sky_model_basis(i, j, nskyx, nskyy, nx, ny)
    return im


def add_sky_parameters(nx, ny, nskyx, nskyy, xloc, yloc, values, weight):
    # yloc: just add rows to the end according to the current largest row
    # in there
    nskypar = nskyx * nskyy
    newxloc = [numpy.arange(nx*ny, dtype='i4')]*nskypar
    # for the moment, don't take advantage of the bounded support.
    newyloc = [numpy.max(yloc) + 1 + i*numpy.ones((nx, ny), dtype='i4').ravel()
               for i in range(nskypar)]
    newvalues = [(sky_model_basis(i, j, nskyx, nskyy, nx, ny)*weight).ravel()
                 for i in range(nskyx) for j in range(nskyy)]
    for i in range(nskypar):
        m = newvalues[i] != 0
        newxloc[i] = newxloc[i][m]
        newyloc[i] = newyloc[i][m]
        newvalues[i] = newvalues[i][m]
    # print('Think about what is going on with the padding?
    # We're now passing nx & ny for the padded image, so the sky
    # extends through the padded region.  Probably fine.
    xloc = numpy.concatenate([xloc] + newxloc)
    yloc = numpy.concatenate([yloc] + newyloc)
    values = numpy.concatenate([values] + newvalues)
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
    dist = dist.ravel()
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


def moffat_psf(fwhm, beta=3., stampsz=19):
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
    rc = numpy.sqrt(xc**2. + yc**2.)
    psf = (beta - 1)/(numpy.pi * alpha**2.)*(1.+(rc**2./alpha**2.))**(-beta)
    dpsffac = (beta-1)/(numpy.pi*alpha**2.)*(beta)*(
        (1+(rc**2./alpha**2.))**(-beta-1))
    dpsfdx = dpsffac*2*xc/alpha
    dpsfdy = dpsffac*2*yc/alpha
    return psf, dpsfdx, dpsfdy


def center_psf(psf):
    """Center and normalize a psf; centroid is placed at center."""
    stampsz = psf.shape[0]
    stampszo2 = stampsz // 2
    xc = numpy.arange(stampsz, dtype='f4')-stampszo2
    xc = xc.reshape(-1, 1)
    yc = xc.copy().reshape(1, -1)
    for _ in range(3):
        xcen = numpy.sum(xc*psf)/numpy.sum(psf)
        ycen = numpy.sum(yc*psf)/numpy.sum(psf)
        psf = shift(psf, [-xcen, -ycen], output=numpy.dtype('f4'))
    psf /= numpy.sum(psf)
    psf = psf.astype('f4')
    return psf


def find_psf(xcen, shiftx, ycen, shifty, psfstack, residstack, weightstack):
    """Find PSF from stamps."""
    # let's just go ahead and correlate the noise
    xr = numpy.round(shiftx)
    yr = numpy.round(shifty)
    psfqf = (numpy.sum(psfstack*(weightstack > 0), axis=(1, 2)) /
             numpy.sum(psfstack, axis=(1, 2)))
    totalflux = numpy.sum(psfstack, axis=(1, 2))
    chi2 = numpy.sum(residstack**2.*((weightstack+1.e-10)**-1. +
                                     (0.3*psfstack)**2.)**-1.,
                     axis=(1, 2))
    okpsf = ((numpy.abs(psfqf - 1) < 0.03) & (totalflux > 5000) &
             (chi2 < 1.5*len(psfstack[0].reshape(-1))))
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
    tpsf = numpy.median(psfstack, axis=0)
    tpsf = center_psf(tpsf)
    return tpsf


sample = """
sample code:

psf, dpsfdx, dpsfdy = crowdsource.gaussian_psf(fwhm, 19)
reload(crowdsource) ; im, xt, yt, fluxt = crowdsource.sim_image(1000, 1000, 30000, 5., 3.) ; clf() ; util_efs.imshow(im, arange(1000), arange(1000), vmax=20, aspect='equal') ; xlim(0, 300) ; ylim(0, 300)
reload(crowdsource) ; sigim = crowdsource.significance_image(im, im*0+3.**2., psf)
reload(crowdsource) ; xydat = crowdsource.peakfind(im, sigim, 3.)
reload(crowdsource) ;  x2, y2, flux2, model2 = crowdsource.fit_im(im, psf, 3., 0.3, psfderiv=[dpsfdx, dpsfdy])
"""
