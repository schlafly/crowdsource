import numpy
import scipy
import pdb
import scipy.ndimage.filters as filters
from scipy.ndimage.interpolation import shift


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
        psf2 = shift(psf, [fracshiftx, fracshifty], numpy.dtype('f4'), order=4)
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


def in_padded_region(flatcoord, imshape, pad):
    coord = numpy.unravel_index(flatcoord, imshape)
    m = numpy.zeros(len(flatcoord), dtype='bool')
    for c, length in zip(coord, imshape):
        m |= (c < pad) | (c >= length - pad)
    return m


def fit_once(im, x, y, psf, weight=None, psfderiv=None, nskyx=0, nskyy=0):
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
        for j in range(repeat):
            xloc.append(numpy.ravel_multi_index(((xe[i]+xpix), (ye[i]+ypix)),
                                                im.shape))
    xloc = numpy.concatenate(xloc)
    yloc = (numpy.arange(len(x)*repeat, dtype='i4').reshape(-1, 1) *
            numpy.ones(len(xpix), dtype='i4').reshape(1, -1)).ravel()
    valarr = []
    for i in xrange(len(xe)):
        wt = weight[xe[i]:xe[i]+stampsz, ye[i]:ye[i]+stampsz]
        valarr += [(shift(psf, [xf[i], yf[i]],
                          numpy.dtype('f4'), order=4)*wt).ravel()]
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
    # below only handles square case.  I can, of course, convert, but...
    # flux = scipy.sparse.linalg.cg(mat, im.ravel())
    flux = lsqr_cp(mat, (im*weight).ravel(), atol=1.e-3, btol=1.e-3)
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
    # the numerator of the centroid is sum(x * psf * (res + mod)
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
            psf0 = shift(psf, offset, numpy.dtype('f4'), order=4)
            if psfderiv is None:
                psfderiv0 = None
            else:
                psfderiv0 = [shift(p, offset, numpy.dtype('f4'), order=4)
                             for p in psfderiv]
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
        # if (xe[i] > 100) and (ye[i] > 100) and (flux[repeat*i] > 10000):
        #     pdb.set_trace()
    xcen *= 2
    ycen *= 2
    res = (xcen, ycen)
    if return_psfstack:
        res = res + (psfstack, residstack)
    return res


def initial_sky_filter(im, sz=100):
    # too slow!
    # return filters.median_filter(im, sz, mode='reflect')
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
            val[i+1, j+1] = numpy.median(im[sx:ex, sy:ey])
    # I can't figure out how to ask for any kind of reasonable extrapolation
    # off the grid.  I'm going to extend xg, yg, val to compensate.
    val[0, :] = val[1, :]-(val[2, :]-val[1, :])
    val[:, 0] = val[:, 1]-(val[:, 2]-val[:, 1])
    val[-1, :] = val[-2, :]+(val[-2, :]-val[-3, :])
    val[:, -1] = val[:, -2]+(val[:, -2]-val[:, -3])
    # this gets the corners since they're done both times
    print('Manually adjusting sky to account for the median being biased high '
          'due to real sources...')
    val -= 9.7
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


def fit_im(im, psf, threshhold, weight=None, psfderiv=None,
           nskyx=3, nskyy=3):
    if (nskyx != 0) or (nskyy != 0):
        imbs = initial_sky_filter(im)
    else:
        imbs = im
    if isinstance(weight, int):
        weight = numpy.ones_like(im)*weight
    sigim = significance_image(imbs, weight, psf)
    x, y = peakfind(imbs, sigim, weight)
    print('Found %d initial sources.' % len(x))
    # pdb.set_trace()

    # first fit; brightest sources
    flux, model, sky = fit_once(im, x, y, psf, psfderiv=psfderiv,
                                weight=weight, nskyx=nskyx, nskyy=nskyy)
    sigim_res = significance_image(im - model, weight, psf)
    x2, y2 = peakfind(im - model, sigim_res, weight)

    # accept only peaks where im / numpy.clip(model, threshhold) > 0.3
    sigstat = ((im-model)/numpy.clip(model-sky, threshhold, numpy.inf))[x2, y2]
    m = sigstat > 0.3  # we haven't computed centroids yet.
    x2, y2 = x2[m], y2[m]
    print('Found %d additional sources.' % len(x2))
    x = numpy.concatenate([x, x2])
    y = numpy.concatenate([y, y2])

    # now fit sources that may have been too blended to detect initially
    # pdb.set_trace()
    flux, model, sky = fit_once(im, x, y, psf, psfderiv=psfderiv,
                                weight=weight, nskyx=nskyx, nskyy=nskyy)
    xcen, ycen = compute_centroids(x, y, psf, flux[0], im-model, weight,
                                   psfderiv=psfderiv)
    m = (numpy.abs(xcen) < 3) & (numpy.abs(ycen) < 3)
    x = x.astype('f4')
    y = y.astype('f4')
    x[m] += xcen[m]
    y[m] += ycen[m]
    x[m] = numpy.clip(x[m], 0, im.shape[0]-1)
    y[m] = numpy.clip(y[m], 0, im.shape[1]-1)
    repeat = 3 if psfderiv else 1
    keep = cull_near(x, y, flux[0][0:repeat*len(x):repeat])
    x = x[keep]
    y = y[keep]

    sigim_res = significance_image(im - model, weight, psf)
    x3, y3 = peakfind(im - model, sigim_res, weight)

    # accept only peaks where im / numpy.clip(model, threshhold) > 0.3
    sigstat = ((im-model)/numpy.clip(model-sky, threshhold, numpy.inf))[x3, y3]
    m = sigstat > 0.3
    x3, y3 = x3[m], y3[m]
    print('Found %d additional sources.' % len(x3))
    x = numpy.concatenate([x, x3])
    y = numpy.concatenate([y, y3])

    # now fit with improved locations
    # pdb.set_trace()
    flux, model, sky = fit_once(im, x, y, psf, psfderiv=psfderiv,
                                weight=weight, nskyx=nskyx, nskyy=nskyy)
    xcen, ycen = compute_centroids(x, y, psf, flux[0], im-model, weight,
                                   psfderiv=psfderiv)
    m = (numpy.abs(xcen) < 3) & (numpy.abs(ycen) < 3)
    x = x.astype('f4')
    y = y.astype('f4')
    x[m] += xcen[m]
    y[m] += ycen[m]
    x[m] = numpy.clip(x[m], 0, im.shape[0]-1)
    y[m] = numpy.clip(y[m], 0, im.shape[1]-1)
    keep = cull_near(x, y, flux[0][0:repeat*len(x):repeat])
    x = x[keep]
    y = y[keep]

    # final fit with final locations; no psfderiv allowed, positions are fixed.
    # pdb.set_trace()
    flux, model, sky = fit_once(im, x, y, psf, psfderiv=None,
                                weight=weight, nskyx=nskyx, nskyy=nskyy)
    return x, y, flux, model


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
    stampszo2 = (stampsz // 2) + 1
    xc = numpy.arange(stampsz, dtype='f4')-stampszo2
    yc = xc.copy()
    psf = numpy.exp(-(xc.reshape(-1, 1)**2. + yc.reshape(1, -1)**2.) /
                    2./sigma**2.).astype('f4')
    dpsfdx = xc.reshape(-1, 1)/sigma**2.*psf
    dpsfdy = yc.reshape(1, -1)/sigma**2.*psf
    return psf, dpsfdx, dpsfdy


def find_psf(psfstack, xcen, ycen):
    posflux = numpy.sum(numpy.sum(psfstack * (psfstack > 0), axis=1), axis=1)
    negflux = -numpy.sum(numpy.sum(psfstack * (psfstack < 0), axis=1), axis=1)
    m = posflux > negflux * 3
    psfstack = psfstack[m, :, :]
    xcen = xcen[m]
    ycen = ycen[m]
    print numpy.sum(m)
    for i in range(psfstack.shape[0]):
        psfstack[i, :, :] = shift(psfstack[i, :, :], [-xcen[i], -ycen[i]],
                                  numpy.dtype('f4'), order=4)
    sums = numpy.sum(numpy.sum(psfstack, axis=1), axis=1)
    psfstack /= sums.reshape(-1, 1, 1)
    psfstack = numpy.median(psfstack, axis=0)
    psfstack /= numpy.sum(psfstack)
    return psfstack


sample = """
sample code:

psf, dpsfdx, dpsfdy = gaussian_psf(fwhm, 19)
reload(crowdsource) ; im, xt, yt, fluxt = crowdsource.sim_image(1000, 1000, 30000, 5., 3.) ; clf() ; util_efs.imshow(im, arange(1000), arange(1000), vmax=20, aspect='equal') ; xlim(0, 300) ; ylim(0, 300)
reload(crowdsource) ; sigim = crowdsource.significance_image(im, im*0+3.**2., psf)
reload(crowdsource) ; xydat = crowdsource.peakfind(im, sigim, 3.)
reload(crowdsource) ;  x2, y2, flux2, model2 = crowdsource.fit_im(im, psf, 3., 0.3, psfderiv=[dpsfdx, dpsfdy])
"""
