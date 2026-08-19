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

import time
import sys
import os
import numpy as np
from scipy import sparse
import pdb
import crowdsource.psf as psfmod
import scipy.ndimage.filters as filters
from collections import OrderedDict
from scipy import sparse
from typing import List, Tuple, Optional
from astropy.stats import mad_std

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
    im = np.random.randn(nx, ny).astype('f4')*noise
    stampszo2 = stampsz // 2
    im = np.pad(im, [stampszo2, stampszo2], constant_values=-1e6,
                   mode='constant')
    x = np.random.rand(nstar).astype('f4')*(nx-1)
    y = np.random.rand(nstar).astype('f4')*(ny-1)
    flux = 1./np.random.power(1.0, nstar)
    for i in range(nstar):
        stamp = psf(x[i], y[i], stampsz=stampsz)
        xl = np.round(x[i]).astype('i4')
        yl = np.round(y[i]).astype('i4')
        im[xl:xl+stampsz, yl:yl+stampsz] += stamp*flux[i]
    if (nskyx != 0) or (nskyy != 0):
        im += sky_model(100*np.random.rand(nskyx, nskyy).astype('f4'),
                        im.shape[0], im.shape[1])
    ret = im[stampszo2:-stampszo2, stampszo2:-stampszo2], x, y, flux
    return ret


def significance_image(ims, models, isigs, psfs, band_weights=None, sz=19):
    """
    Significance of a PSF at each point, without local background fit, for single or multi-band data.

    Parameters
    ----------
    ims : 2darray or list of 2darray
        Sky-subtracted data image(s). 
    models : 2darray or list of 2darray
        Current model image(s), same shape(s) as ims.
    isigs : 2darray or list of 2darray
        Inverse-sigma map(s) for each band.
    psfs : PSF object or list of PSF objects
        PSF(s) corresponding to each band.
    band_weights : list(float), optional
        Relative weights for each band in the matched filter. If None, use equal weights (=1/nband).
    sz : int, optional
        PSF stamp size for convolution.

    Returns
    -------
    sigim   : 2darray[Nx,Ny]
        Joint significance image (data x PSF / noise).
    modim : 2darray[Nx,Ny]
        Combined model significance image (model x PSF / noise).
    """
    from scipy.signal import fftconvolve
    
    # assume, for the moment, the image has already been sky-subtracted
    def convolve(im, kernel):
        return fftconvolve(im, kernel[::-1, ::-1], mode='same')

    if np.array(ims).ndim == 2:
        ims     = [ims]
        models  = [models]
        isigs   = [isigs]
        psfs    = [psfs]
        band_weights = [1.0]
    else:
        nband = len(ims)
        if band_weights is None:
            band_weights = [1.0 / nband] * nband

    sigim, modim, varim = 0.0, 0.0, 0.0
    for w, im, mod, isig, psf in zip(band_weights, ims, models, isigs, psfs):
        psfstamp = psfmod.central_stamp(psf, sz).copy()
        sigim += w * convolve(im   * isig**2, psfstamp)
        modim += w * convolve(mod  * isig**2, psfstamp)
        varim += w**2 * convolve(isig**2, psfstamp**2)

    varim[varim <= 1e-14] = 0.0   # numerical noise starts to set in around here.
    ivarim = 1. / (varim + (varim == 0) * 1e14) 

    return sigim * np.sqrt(ivarim), modim * np.sqrt(ivarim)




def significance_image_lbs(im, model, isig, psf, sz=19):
    """Give significance of PSF at each point, with local background fits."""

    def convolve(im, kernel):
        from scipy.signal import fftconvolve
        return fftconvolve(im, kernel[::-1, ::-1], mode='same')

    def convolve_flat(im, sz):
        from scipy.ndimage.filters import convolve
        filt = np.ones(sz, dtype='f4')
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
    fluxisig = np.sqrt(atcinvadet/ivarf)
    fluxsig = fluxest*fluxisig
    modim = convolve(model*ivar, psfstamp)
    return fluxsig, modim*np.sqrt(ivarimsimple)

def peakfind(ims, models, isigs, dq, psfs, *,
                  band_weights=None, keepsat=False, threshold=5,
                  blendthreshold=0.3, psfvalsharpcutfac=0.7,
                  psfsharpsat=0.7, maxfiltershape=3, psfsz=59):
    """
    Identify significant, non-blended peaks in a singleband or *joint* multiband fit.

    Parameters
    ----------
    ims, models, isigs : 2darray or list of 2darray
        Sky-subtracted data, current model, and inverse-sigma/weight maps.
    dq : ndarray or None
        Data-quality mask.
    psfs : PSF or list of PSF functions
        PSF(s) corresponding to each band.
    band_weights : list(float), optional
        Relative weights across bands (defaults to equal weights).
    keepsat, threshold, blendthreshold, psfvalsharpcutfac, psfsharpsat, maxfiltershape, psfsz : float or int
        Detection and filtering parameters.

    Returns
    -------
    x, y : 1-D int arrays
        Coordinates of detected peaks.
    sigim : 2D ndarray
        Detection significance image (e.g., S/N-like).
    modelsigim : 2D ndarray
        Model significance image. Useful for QA/debugging.
    """
    from scipy.ndimage import filters

    # Detect single-band or multi-band
    if np.array(ims).ndim == 2:
        ims     = [ims]
        models  = [models]
        isigs   = [isigs]
        psfs    = [psfs]
        band_weights = [1.0]
    else:
        nband = len(ims)
        if band_weights is None:
            band_weights = [1.0 / nband] * nband

    # Build PSF stamps
    psfstamps = [psf(int(im.shape[0] / 2.), int(im.shape[1] / 2.), deriv=False, stampsz=psfsz) for psf, im in zip(psfs, ims)]
    
    sigim, modelsigim = significance_image(ims, models, isigs, psfstamps, band_weights=band_weights, sz=psfsz)

    # weighted combined images
    w_invvar = [w * (isig**2) for w, isig in zip(band_weights, isigs)]
    den = np.clip(np.sum(w_invvar, axis=0), 1e-30, np.inf)  
    im_tot    = np.sum([wiv * im  for wiv, im  in zip(w_invvar, ims)],    axis=0) / den
    model_tot = np.sum([wiv * mod for wiv, mod in zip(w_invvar, models)], axis=0) / den
    
    den_var = np.clip(np.sum([(w**2) * (isig**2) for w, isig in zip(band_weights, isigs)], axis=0),1e-30, np.inf) 
    isig_tot = den / np.sqrt(den_var)
    
    # Weighted PSFstamp st each band’s PSF contributes proportional to that band's contribution to the combined image
    wavg = np.array([np.median(w * isig**2) for w, isig in zip(band_weights, isigs)], dtype=float)
    wavg /= wavg.sum()
    psf_tot = np.sum([wi * p for wi, p in zip(wavg, psfstamps)], axis=0)


    # Find local maxima above threshold
    sig_max = filters.maximum_filter(sigim, maxfiltershape)
    x, y = np.nonzero((sig_max == sigim) & (sigim > threshold) & (keepsat | (isig_tot > 0))) 

    fluxratio = im_tot[x, y] / np.clip(model_tot[x, y], 0.01, np.inf)
    sigratio  = (im_tot[x, y] * isig_tot[x, y]) / np.clip(modelsigim[x, y], 0.01, np.inf)
    sigratio2 = sigim[x, y] / np.clip(modelsigim[x, y], 0.01, np.inf)

    # Mask for valid/saturated points
    m = ((isig_tot[x, y] > 0) | (keepsat & (isig_tot[x, y] == 0)))    # ~saturated, or saturated & keep

    if dq is not None and np.any(dq[x, y] & nodeblend_maskbit):   # HyperLeda - big galaxies list
        nodeblend = (dq[x, y] & nodeblend_maskbit) != 0
        bt = np.ones_like(x) * blendthreshold
        bt[nodeblend] = 100          
        blendthreshold = bt           
        
    if dq is not None and np.any(dq[x, y] & sharp_maskbit):   # NN nebulosity or very bright stars - sharp them
        sharp = (dq[x, y] & sharp_maskbit) != 0
        msharp = ~sharp | psfvalsharpcut(
            x, y, sigim, isig_tot, psf_tot,
            psfvalsharpcutfac=psfvalsharpcutfac,
            psfsharpsat=psfsharpsat
        )
        m &= msharp

    # Blend / significance conditions 
    m &= ((sigratio2 > blendthreshold * 2) |
          ((fluxratio > blendthreshold) &
           (sigratio >  blendthreshold / 4.) &
           (sigratio2 > blendthreshold)))

    return x[m], y[m], sigim, modelsigim
    

def psfvalsharpcut(x, y, sigim, isig, psf, psfvalsharpcutfac=0.7,
                   psfsharpsat=0.7):
    xl = np.clip(x-1, 0, sigim.shape[0]-1)
    xr = np.clip(x+1, 0, sigim.shape[0]-1)
    yl = np.clip(y-1, 0, sigim.shape[1]-1)
    yr = np.clip(y+1, 0, sigim.shape[1]-1)
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


def build_model(x, y, flux, nx, ny, psf=None, psflist=None, psfderiv=False,
                stampsz=59):
    if psf is None and psflist is None:
        raise ValueError('One of psf and psflist must be set')
    if psf is not None and psflist is not None:
        raise ValueError('Only one of psf and psflist must be set')
    if psflist is None:
        psflist = build_psf_list(x, y, psf, stampsz, psfderiv=psfderiv)
        sz = np.ones(len(x), dtype='i4')*stampsz
    else:
        sz = np.array([tpsf[0].shape[-1] for tpsf in psflist[0]])
        if len(sz) > 0:
            stampsz = np.max(sz)

    stampszo2 = stampsz//2
    im = np.zeros((nx, ny), dtype='f4')
    im = np.pad(im, [stampszo2, stampszo2], constant_values=0.,
                   mode='constant')
    xp = np.round(x).astype('i4')
    yp = np.round(y).astype('i4')
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
    sz = np.broadcast_to(sz, x.shape)
    psflist = {}
    for tsz in np.unique(sz):
        m = sz == tsz
        res = psf(x[m], y[m], stampsz=tsz, deriv=psfderiv)
        if not psfderiv:
            res = [res]
        psflist[tsz] = res
    counts = {tsz: 0 for tsz in np.unique(sz)}
    out = [[] for i in range(3 if psfderiv else 1)]
    for i in range(len(x)):
        for j in range(len(out)):
            out[j].append(psflist[sz[i]][j][counts[sz[i]]])
        counts[sz[i]] += 1
    return out


def in_padded_region(flatcoord, imshape, pad):
    coord = np.unravel_index(flatcoord, imshape)
    m = np.zeros(len(flatcoord), dtype='bool')
    for c, length in zip(coord, imshape):
        m |= (c < pad) | (c >= length - pad)
    return m

def pad_image_and_weight(im, weight, pad):
    """Pad image and weight"""
    im_pad = np.pad(im, [pad, pad], constant_values=0.)
    if weight is None:
        weight = np.ones_like(im)
    wt_pad = np.pad(weight, [pad, pad], constant_values=0.)
    wt_pad[wt_pad == 0] = 1e-20
    return im_pad, wt_pad


def build_sparse_matrix(images_pad, weights_pad, psfs, x, y,
                        psfderiv=False, nskyx=0, nskyy=0, guess=None):
    """
    Create the sparse design matrix (A) for single band or simultaneous fitting of multiple bands. Rows at first equal to the fluxes at each peak,
    later add in the derivatives at each peak.

    Parameters
    ----------
    images_pad : list of ndarray
        Padded images (one per band).
    weights_pad : list of ndarray
        Corresponding padded weight maps.
    psfs : list of list of ndarray
        psfs[b][j][i] gives PSF stamp for source i, derivative j (0=flux, 1=dx, 2=dy) in band b.
    x, y : ndarray
        Shared PADDED source positions (length N).
    psfderiv : bool
        If True, include dx, dy PSF derivatives as shared parameters.
    nskyx, nskyy : int
        Number of sky model parameters in x, y direction (0 or >=3).
    guess : ndarray, optional
        Initial flux guess for weighted derivative normalization.

    Returns
    -------
    mat : scipy.sparse.csc_matrix
        Sparse design matrix A (weighted PSFs + optional sky).
    colnorm : ndarray
        Column normalization factors used for LSQR stability.
    npixim : int
        Number of pixels per image (for later slicing).
    nskypar : int
        Number of sky parameters per band.
    """

    B = len(images_pad)
    N = len(x)
    nskypar = nskyx*nskyy
    npixim = images_pad[0].shape[0] * images_pad[0].shape[1]

    # PSF stamp sizes
    sz_b = [np.array([psfs[b][0][i].shape[-1] for i in range(N)]) for b in range(B)]
    szo2_b = [sz // 2 for sz in sz_b]
    stampsz = max(max(sz) for sz in sz_b) if N > 0 else 19      # common stampsz across both bands to keep images aligned

    # Coordinates on padded images (x,y are already in padded coords)
    xp = np.round(x).astype('i4')
    yp = np.round(y).astype('i4')
    
    # move from PSF center to stamp edge (top-left corner of stamp box)
    xe = xp - stampsz // 2
    ye = yp - stampsz // 2

    # Pixel grid for stamps to track where each pixel lands on the image
    # convention: x is the first index, y is the second
    pix = np.arange(stampsz * stampsz).reshape(stampsz, stampsz)
    xpix = pix // stampsz
    ypix = pix % stampsz

    # parameter count
    ncol = B * N                       # fluxes
    if psfderiv:
        ncol += 2 * N                  # shared dx, dy per source
    ncol += B * nskypar                # sky terms

    #Total number of non-zero entries in the sparse matrix 'A'. Sky parameters take nskypar non-zero entries to be precise, but this way is just easier. 
    zsz = (sum(np.sum(sz_b[b] ** 2) for b in range(B)) * (1 + (2 if psfderiv else 0)) + B * nskypar * npixim).astype('i8')
    if zsz >= 2**32:
        raise ValueError("Too many non-zero entries in sparse matrix (>2^32)")

    # Allocate
    xloc = np.zeros(zsz, dtype='i4')   #row number (i.e., pixel index) of the non-zero entries
    values = np.zeros(zsz, dtype='f4')   #values of the non-zero entries
    colnorm = np.zeros(ncol, dtype='f4')   #no. of columns in 'A' matrix. Compute the norm(magnitude) of the column vector for stability.
    first = 0   #pointer that tracks where in the xloc and values arrays we will write data.

    # Flux columns (per band, per source) - base PSF (no scaling)
    for b in range(B):
        for i in range(N):
            # f and l crops the stamp if it's greater than the psf. f stands for 'first index' and l stands for 'last index' to pick from the psf stamp.
            f = stampsz // 2 - szo2_b[b][i]
            l = stampsz - f
            wt = weights_pad[b][xe[i]:xe[i]+stampsz, ye[i]:ye[i]+stampsz][f:l, f:l]

            xloc[first:first+sz_b[b][i]**2] = (
                np.ravel_multi_index(
                    ((xe[i] + xpix[f:l, f:l]), (ye[i] + ypix[f:l, f:l])),   #gives the actual x,y pixel coordinates of each PSF pixel on the padded image.
                    images_pad[b].shape
                ).reshape(-1) + b * npixim
            )
            values[first:first+sz_b[b][i]**2] = (psfs[b][0][i] * wt).reshape(-1)  # store weighted PSF in values
            col_idx = b * N + i

            colnorm[col_idx] = np.sqrt(np.sum(values[first:first+sz_b[b][i]**2] ** 2))   # normalize this column for numerical stability
            if colnorm[col_idx] == 0:
                colnorm[col_idx] = 1.0   #Column-wise L2 norm (for scaling and LSQR stability).Otherwise, bright sources (large flux) would dominate the solution.
            values[first:first+sz_b[b][i]**2] /= colnorm[col_idx]
            first += sz_b[b][i]**2
            
    # Shared derivative columns
    if psfderiv:
        for i in range(N):
            for d, j in enumerate([1, 2]):  # dx, dy
                col_idx = B * N + 2*i + d
                vals = []
                idxs = []
                for b in range(B):
                    f = stampsz // 2 - szo2_b[b][i]
                    l = stampsz - f
                    wt = weights_pad[b][xe[i]:xe[i]+stampsz, ye[i]:ye[i]+stampsz][f:l, f:l]
                    pix_idx = (
                        np.ravel_multi_index(
                            ((xe[i] + xpix[f:l, f:l]), (ye[i] + ypix[f:l, f:l])),
                            images_pad[b].shape
                        ).reshape(-1) + b * npixim
                    )
                    if guess is not None:
                        vals.append((guess[b*N + i] * psfs[b][j][i] * wt).reshape(-1))    # scale by guessflux if available
                    else:
                        vals.append((psfs[b][j][i] * wt).reshape(-1))
                    idxs.append(pix_idx)
                idxs = np.concatenate(idxs)
                vals = np.concatenate(vals)

                colnorm[col_idx] = np.sqrt(np.sum(vals ** 2))
                if colnorm[col_idx] == 0:
                    colnorm[col_idx] = 1.0
                values[first:first+len(vals)] = vals / colnorm[col_idx]
                xloc[first:first+len(vals)] = idxs
                first += len(vals)

    # Sky parameter columns
    if nskypar > 0:
        for b in range(B):
            sxloc, syloc, svalues = sky_parameters(
                images_pad[b].shape[0], images_pad[b].shape[1],
                nskyx, nskyy, weights_pad[b]
            )
            startidx = (B * N + (2 * N if psfderiv else 0)) + b * nskypar  #where sky columns begin in matrix (after source columns)
            for i in range(len(sxloc)):
                xloc[first:first+len(sxloc[i])] = sxloc[i] + b * npixim   #len(sxloc[0]) = total pixels(rows in A) affected by each sky parameter
                colnorm[startidx + syloc[i]] = np.sqrt(np.sum(svalues[i] ** 2))
                if colnorm[startidx + syloc[i]] == 0:
                    colnorm[startidx + syloc[i]] = 1.0
                values[first:first+len(sxloc[i])] = svalues[i] / colnorm[startidx + syloc[i]]
                first += len(sxloc[i])         

    # CSC index pointers (column offsets)
    #csc_indptr defines the start and end of each column in the sparse matrix. It grows by sz[i]**2 per source column, and by nskypix per sky column
    csc_indptr = [0]

    # Flux columns (per band)
    for b in range(B):                  # loop over bands
        for i in range(N):              # loop over sources
            nnz = sz_b[b][i]**2         # each flux column has PSF stamp pixels
            csc_indptr.append(csc_indptr[-1] + nnz)
    
    # Shared derivative columns 
    if psfderiv:
        for i in range(N):              # loop over sources
            nnz_dx = sum(sz_b[b][i] ** 2 for b in range(B))
            csc_indptr.append(csc_indptr[-1] + nnz_dx)  # dx
            csc_indptr.append(csc_indptr[-1] + nnz_dx)  # dy
    
    #  Sky columns (per band) - each sky basis spans the whole image, so nnz = npixim
    if nskypar > 0:
        for b in range(B):
            for _ in range(nskypar):
                csc_indptr.append(csc_indptr[-1] + npixim)

    shape = (B * npixim, ncol)
    mat = sparse.csc_matrix((values, xloc, csc_indptr), shape=shape, dtype='f4')

    return mat, colnorm, npixim, nskypar


def solve_and_reconstruct(mat, rhs, colnorm, pad,
                          images_pad, weights_pad, nskyx, nskyy,
                          psfderiv, nskypar, B, N, guessvec):
    """
    Solve LSQR and reconstruct per-band model/sky images. 
    """ 
    flux = lsqr_cp(mat, rhs.ravel(), atol=1.e-4, btol=1.e-4, guess=guessvec)
    flux_scaled = flux[0].copy()          # still column-normalised to be used for model
    flux[0] /= colnorm            # Undo column normalization. This will be the returned flux and will be used in sky model

    model, sky = [], []
    sky_offset = B*N + (2*N if psfderiv else 0)       # index where sky parameters start

    y = mat.dot(flux_scaled)  
    for b in range(B):
        npixim = images_pad[b].size
        band_model = y[b*npixim : (b+1)*npixim].reshape(images_pad[b].shape)
        band_model = band_model[pad:-pad, pad:-pad]
        wt = weights_pad[b][pad:-pad, pad:-pad]
        band_model /= (wt + (wt == 0))    #Undo the weight multiplication applied earlier to the image
        model.append(band_model)

        if nskypar > 0:
            sky_params = flux[0][sky_offset + b*nskypar : sky_offset + (b+1)*nskypar]
            skymap = sky_model(sky_params.reshape(nskyx, nskyy), images_pad[b].shape[0], images_pad[b].shape[1])
            sky.append(skymap[pad:-pad, pad:-pad])
        else:
            sky.append(np.zeros_like(model[-1]))

    return flux, np.array(model), np.array(sky)


def fit_once(ims, x, y, psfs, weights=None, psfderiv=False, nskyx=0, nskyy=0, guess=None):

    """
    Fit fluxes for psfs at x & y in image im.
    Single band and/or Multiband fitting of fluxes for shared (x, y) positions across bands.

    Parameters
    ----------
    images : list of ndarray
        List of sky-subtracted images, one per band.
    x, y : ndarray
        Shared source coordinates (length N).
    psfs : list of list of ndarray
        psfs[b][j][i] gives PSF stamp for source i and derivatives j (j<=3) in band b.
    weights : list of ndarray, optional
        One weight map per band. Defaults to uniform if None.
    psfderiv : bool
        Whether PSF derivatives are included (dx, dy).
    nskyx, nskyy : int
        Sky model parameterization per band (0 or >=3).
    guess : ndarray, optional
        Initial guess vector (flux only, no derivatives).

    Returns
    -------
    flux : ndarray
        Solution vector where flux[0] is of shape (B*N + (2*N if psfderiv) + B*nskypar,), ordered as:
        [ per-band fluxes (B*N), shared dx (N), shared dy (N), sky params (B*nskypar) ].
    model : list of ndarray
        Modeled image for each band.
    sky : list of ndarray
        Modeled sky image for each band.
    """

    # Detect single-band or multiband case
    if np.array(ims).ndim == 2: 
        ims = [ims]
        psfs = [psfs]
        weights = [weights]
    B = len(ims)

    # common stamp size & padding
    N = len(x)
    sz_b = [np.array([psfs[b][0][i].shape[-1] for i in range(N)]) for b in range(B)]
    stampsz = max(max(sz) for sz in sz_b) if N > 0 else 19
    pad = stampsz // 2 + 1

    # Pad all bands
    images_pad, weights_pad = [], []
    for b in range(B):
        im_pad, wt_pad = pad_image_and_weight(ims[b], weights[b], pad)
        images_pad.append(im_pad)
        weights_pad.append(wt_pad)
        
    # Images are now padded but x, y were estimated on unpadded image. So we pad them here.
    xp = np.round(x).astype('i4') + pad
    yp = np.round(y).astype('i4') + pad

    # Build sparse matrix
    mat, colnorm, npixim, nskypar = build_sparse_matrix( images_pad, weights_pad, psfs, xp, yp, psfderiv=psfderiv, nskyx=nskyx, nskyy=nskyy, guess=guess)

    # RHS= (stacked image*weight) 
    rhs = np.concatenate([(images_pad[b]*weights_pad[b]).ravel() for b in range(B)])

    if guess is not None:
        # total parameter count 
        ncol = B * N + (2 * N if psfderiv else 0) + B * nskypar
        guessvec = np.zeros(ncol, dtype='f4')
        for b in range(B):
            start = b * N
            end = (b + 1) * N
            guessvec[start:end] = guess[start:end]  # Fill only the flux slots
    
        # sky guesses
        if nskypar > 0:
            # take last B*nskypar entries of guess for sky params
            guessvec[-B * nskypar:] = guess[-B * nskypar:]
        guessvec *= colnorm
    else: guessvec = None

    # Solve and reconstruct
    flux, model, sky = solve_and_reconstruct(mat, rhs, colnorm, pad, images_pad, weights_pad,nskyx, nskyy, psfderiv, nskypar, B, N, guessvec = guessvec)

    return flux, model, sky


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
            fac = np.sum(bb**2.)**(0.5)/np.sum(bb2**2.)**0.5
            kw['btol'] = kw['btol']*np.clip(fac, 0.1, 10.)
    else:
        bb2 = bb.copy()

    normbb = np.sum(bb2**2.)
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


def build_source_stamps(x, y, psflists, flux, images, resids, weights, centroidsize=19):
    """
    define c = integral(x * I * P * W) / integral(I * P * W)
    x = x/y coordinate, I = isolated stamp, P = PSF model, W = weight
    Assuming I ~ P(x-y) for some small offset y and expanding, integrating by parts gives:
    y = 2 / integral(P*P*W) * integral(x*(I-P)*W). That is the offset we want.

    we want to compute the centroids on the image after the other sources have been subtracted off.
    we construct this image by taking the residual image, and then star-by-star adding the model back.
    
    Parameters
    ----------
    images : ndarray or list of ndarray
        Sky-subtracted images. One per band if list, else single image.
    x, y : ndarray
        Shared source coordinates (length N).
    psflists : list (of list)
        psflists[b][j][i] gives PSF stamp for source i, derivative j, in band b.
        For single-band, psflists[j][i].
    weights : ndarray or list of ndarray
        Weight maps, same structure as images.
    flux : ndarray
        Flattened flux vector from fit_once().
        Layout is:
          [ f^(1)_1...f^(1)_N, f^(2)_1...f^(2)_N, ..., f^(B)_1...f^(B)_N,
            (optional) dx_1, dy_1, dx_2, dy_2, ..., dx_N, dy_N,
            (optional) sky params ... ]
    resids : ndarray or list of ndarray
        Residual images (= data - model - sky). Same structure as images.
    derivcentroids : bool
        If True, include derivative PSFs (not used anymore; centroids forced to 0).
    centroidsize : int
        Side length of cutout stamps.

    Returns
    -------
    stamps : tuple of ndarrays
        Each element has shape (B, N, S, S):
        (model+resid, image, model, weight, psf*flux)
    """

    # Normalize psflists to always have psflists[b][j] form
    if callable(psflists):                  # single-band callable
        psflists = [[psflists]]
    elif callable(psflists[0]):             # multiband callables
        psflists = [[p] for p in psflists]

    # Detect single-band vs multiband case
    if np.array(images).ndim == 2:
        images, resids, weights = [images], [resids], [weights]
        B = 1
    else:
        B = len(images)

    N = len(x)
    deriv_offset = B * N
    has_deriv = len(flux) >= deriv_offset + 2 * N

    # Initialize per-band stamp containers
    stamps_b = []

    for b in range(B):
        # Slice flux block for this band
        start, stop = b * N, (b + 1) * N
        flux_band = flux[start:stop]

        # PSF components (usually 1 or 3: PSF, dPSF/dx, dPSF/dy)
        Kb = len(psflists[b])
        if Kb == 3:
            # flux layout: (f, dx, dy)
            flux_b = np.zeros(3 * N, dtype='f4')
            flux_b[0:3 * N:3] = flux_band

            # fill dx,dy if they exist in the flat flux vector
            if has_deriv:
                dx_block = flux[deriv_offset : deriv_offset + 2*N : 2]
                dy_block = flux[deriv_offset+1 : deriv_offset + 2*N : 2]
                flux_b[1:3 * N:3] = dx_block
                flux_b[2:3 * N:3] = dy_block
        else:
            flux_b = flux_band.copy()  

        # build PSF stamps centered to 'centroidsize'
        psfs = [np.zeros((N, centroidsize, centroidsize), dtype='f4') for _ in range(len(psflists[b]))]
        for j in range(Kb):
            for i in range(N):
                psf_stamp = psfmod.central_stamp(psflists[b][j][i], censize=centroidsize)
                if np.all(psf_stamp == 0):
                    # Fallback: rebuild from canonical PSF at (x[i], y[i])
                    psf_stamp = psfmod.central_stamp(psflists[b][j][i], censize=centroidsize, force_center=True)
                    if psf_stamp is None or np.all(psf_stamp == 0):
                        print(f"[repair-final] source {i}, comp {j}: could not restore PSF stamp")
                psfs[j][i, :, :] = psf_stamp

        stampsz = psfs[0].shape[-1]
        stampszo2 = (stampsz - 1) // 2

        # source centers & padding for stamp extraction
        xp = np.round(x).astype('i4')
        yp = np.round(y).astype('i4')
        # subtracting to get to the edge of the stamp, adding back to deal with the padded image. 
        # The point we want the edge from the middle (subtract stampszo2) but the image is padded by szo2 (add stampszo2).
        xe = xp - stampszo2 + stampszo2   
        ye = yp - stampszo2 + stampszo2

        resid = np.pad(resids[b], [stampszo2, stampszo2], constant_values=0., mode='constant')
        weight = np.pad(weights[b], [stampszo2, stampszo2], constant_values=0., mode='constant')
        im = np.pad(images[b], [stampszo2, stampszo2], constant_values=0., mode='constant')

        # extract residual/data/weight stamps
        residst = np.array([resid[xe0:xe0+stampsz, ye0:ye0+stampsz] for (xe0, ye0) in zip(xe, ye)])
        weightst = np.array([weight[xe0:xe0+stampsz, ye0:ye0+stampsz] for (xe0, ye0) in zip(xe, ye)])
        psfst = psfs[0] * flux_b[:len(x) * len(psflists[b]):len(psflists[b])].reshape(-1, 1, 1)
        imst = np.array([im[xe0:xe0+stampsz, ye0:ye0+stampsz] for (xe0, ye0) in zip(xe, ye)])

        if len(x) == 0:
            weightst = psfs[0].copy()
            residst = psfs[0].copy()
            imst = psfs[0].copy()

        # reconstruct model stamp
        modelst = psfst.copy()
        if len(psflists[b]) > 1:
            modelst += psfs[1] * flux_b[1:len(x) * len(psflists[b]):len(psflists[b])].reshape(-1, 1, 1)
            modelst += psfs[2] * flux_b[2:len(x) * len(psflists[b]):len(psflists[b])].reshape(-1, 1, 1)

        if np.any([np.all(w == 0) for w in weightst]):
            print(f"[band {b}] Warning: empty weight stamp for some sources")
            
        # stamps tuple:
        #   0: neighbor-subtracted stamp (model + residual)
        #   1: image stamp
        #   2: model stamp (base PSF plus shifts, i.e., dx/dy PSF-deriv terms. flux scaling applied)
        #   3: weight stamp
        #   4: base PSF stamp (flux component only; no dx/dy terms)
        res = (modelst + residst, imst, modelst, weightst, psfst)
        stamps_b.append(res)  

    # Stack per element across bands = (B, N, S, S)
    stacked = tuple(np.stack([st[idx] for st in stamps_b], axis=0) for idx in range(len(stamps_b[0])))

    return stacked


def estimate_sky_background(im):
    """Find peak of count distribution; pretend this is the sky background."""
    # for some reason, I have found this hard to work robustly.  Replace with
    # median at the moment.

    return np.median(im)


def sky_im(im, weight=None, npix=20, order=1):
    """Remove sky from image."""
    nbinx, nbiny = (np.ceil(sh/1./npix).astype('i4') for sh in im.shape)
    xg = np.linspace(0, im.shape[0], nbinx+1).astype('i4')
    yg = np.linspace(0, im.shape[1], nbiny+1).astype('i4')
    val = np.zeros((nbinx, nbiny), dtype='f4')
    usedpix = np.zeros((nbinx, nbiny), dtype='f4')
    if weight is None:
        weight = np.ones_like(im, dtype='f4')
    if np.all(weight == 0):
        return im*0
    # annoying!
    for i in range(nbinx):
        for j in range(nbiny):
            use = weight[xg[i]:xg[i+1], yg[j]:yg[j+1]] > 0
            usedpix[i, j] = np.sum(use)
            if usedpix[i, j] > 0:
                val[i, j] = estimate_sky_background(
                    im[xg[i]:xg[i+1], yg[j]:yg[j+1]][use])
    val[usedpix < 20] = 0.
    usedpix[usedpix < 20] = 0.
    from scipy.ndimage.filters import gaussian_filter
    count = 0
    while np.any(usedpix == 0):
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
            val[m] = np.median(im)
            print('Sky estimation failed badly.')
            break
    x = np.arange(im.shape[0])
    y = np.arange(im.shape[1])
    xc = (xg[:-1]+xg[1:])/2.
    yc = (yg[:-1]+yg[1:])/2.
    from scipy.ndimage import map_coordinates
    xp = np.interp(x, xc, np.arange(len(xc), dtype='f4'))
    yp = np.interp(y, yc, np.arange(len(yc), dtype='f4'))
    xpa = xp.reshape(-1, 1)*np.ones(len(yp)).reshape(1, -1)
    ypa = yp.reshape(1, -1)*np.ones(len(xp)).reshape(-1, 1)
    coord = [xpa.ravel(), ypa.ravel()]
    bg = map_coordinates(val, coord, mode='nearest', order=order)
    bg = bg.reshape(im.shape)
    return bg


def get_sizes(x, y, imbs, weight=None, blist=None, blistsz=299,
              cutofflist=None):
    if cutofflist is None:
        cutofflist = [
            (-np.inf, 19), (1000, 59), (20000, 149)]
    x = np.round(x).astype('i4')
    y = np.round(y).astype('i4')
    peakbright = imbs[x, y]

    if weight is not None:
        # treat saturated / off edge sources as very bright.
        peakbright[weight[x, y] == 0] = cutofflist[-1][0] + 1

    sz = np.zeros(len(x), dtype='i4')
    nbright = list()
    for cutoff, tsz in cutofflist:
        m = peakbright > cutoff
        sz[m] = tsz
        nbright.append(np.sum(m))

    if ((len(nbright) > 2) and (nbright[-1] > 100) and
            (nbright[-1] > nbright[-2] / 2)):
        print('Too many bright sources, using smaller PSF stamp size...')
        sz[peakbright > cutofflist[-2][0]] = cutofflist[-2][1]

    # sources near listed sources get very big PSF
    if blist is not None and len(x) > 0:
        for xb, yb in zip(blist[0], blist[1]):
            dist2 = (x-xb)**2 + (y-yb)**2
            indclose = np.argmin(dist2)
            if dist2[indclose] < 5**2:
                sz[indclose] = blistsz
    return sz


def refit_psf_from_stamps(psf, x, y, xcen, ycen, stamps, name=None,
                          plot=False):
    # how far the centroids of the model PSFs would
    # be from (0, 0) if instantiated there
    # this initial definition includes the known offset (since
    # we instantiated off a pixel center), and the model offset
    xe, ye = psfmod.simple_centroid(
        psfmod.central_stamp(stamps[4], censize=stamps[0].shape[-1]))
    # now we subtract the known offset
    xe -= x-np.round(x)
    ye -= y-np.round(y)
    if hasattr(psf, 'fitfun'):
        psffitfun = psf.fitfun
        npsf = psffitfun(x, y, xcen+xe, ycen+ye, stamps[0],
                         stamps[1], stamps[2], stamps[3], nkeep=200,
                         name=name, plot=plot)
        if npsf is not None:
            npsf.fitfun = psffitfun
    else:
        shiftx = xcen + xe + x - np.round(x)
        shifty = ycen + ye + y - np.round(y)
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


def fit_im(images, psfs, weights=None, dq=None, band_weights=None,
                psfderiv=True, nskyx=0, nskyy=0, refit_psf=False,
                verbose=False, miniter=4, maxiter=10, blist=None,
                maxstars=40000, ntilex=1, ntiley=1, fewstars=100, threshold=5,
                ccd=None, plot=False, titer_thresh=2, blendthreshu=2,
                psfvalsharpcutfac=0.7, psfsharpsat=0.7):
    """
    Handles both:
      - Single-band inputs (im, psf, weight)
      - Multiband inputs ([im_b1, im_b2, ...], [psf_b1, psf_b2, ...], [weight_b1, ...])

    This function performs iterative source detection and photometric fitting:
      1. Background (sky) estimation
      2. Peak finding
      3. PSF building
      4. Iterative fitting via fit_once_auto()
      5. Centroid and flux refinement
      6. Final stats via compute_stats_auto()

    Parameters
    ----------
    images : ndarray or list of ndarray
        Sky-subtracted image(s).
    psfs : callable or list of callable
        PSF model(s) corresponding to each image band.
    weights : ndarray or list of ndarray
        Weight (inverse noise) maps.
    dq : ndarray, optional
        Data quality mask.
    band_weights : list, optional
        Relative per-band weights for multiband peak finding.
    psfderiv : bool
        Whether to include derivative PSFs (enables centroid fitting).
    nskyx, nskyy : int
        Sky model grid dimensions.
    refit_psf : bool
        Whether to refit the PSF after each iteration using source stamps.
    verbose : bool
        Print timing and iteration information.
    """

    from collections import OrderedDict
    import gc, time

    t0_all = time.time()

    # Detect single-band case and normalize inputs
    if isinstance(images, np.ndarray):
        images  = [images]
        psfs    = [psfs]
        weights = [np.ones_like(images[0]) if weights is None else weights]
        B = 1
    else:
        B = len(images)

    shape = images[0].shape
    models = [np.zeros_like(im) for im in images]
    mskys  = [np.zeros_like(im) for im in images]
    xa = np.zeros(0, dtype='f4')
    ya = np.zeros(0, dtype='f4')
    passno = np.zeros(0, dtype='i4')
    guessflux, guesssky = None, None
    titer, lastiter = -1, -1
    skypar = {}  # per-tile sky parameters
    nskypar = nskyx * nskyy
    subregs = list(subregions(shape, ntilex, ntiley))

    # Rough PSF FWHM from first band
    roughfwhm = psfmod.neff_fwhm(psfs[0](shape[0]//2, shape[1]//2))
    roughfwhm = np.max([roughfwhm, 3.])

    iter_history = []   #store xcen/ycen for each iteration

    while True:
        titer += 1
        # Sky background estimation (per band)
        hskys, lskys = [], []
        for b in range(B):
            hskys.append(sky_im(images[b] - models[b], weight=weights[b], npix=20))
            lskys.append(sky_im(images[b] - models[b], weight=weights[b], npix=50 * roughfwhm))

        # Peak finding (shared across all bands)
        if titer != lastiter:
            # in first passes, do not split sources!
            blendthresh = blendthreshu if titer < titer_thresh else 0.2
            # t_pk = time.time()
            xn, yn, sigim, modelsigim = peakfind([images[b]-models[b]-hskys[b] for b in range(B)],
                                            [models[b]-mskys[b] for b in range(B)],
                                            weights, dq, psfs, band_weights=band_weights,
                                            keepsat=(titer == 0),
                                            blendthreshold=blendthresh,
                                            threshold=threshold,
                                            psfvalsharpcutfac=psfvalsharpcutfac,
                                            psfsharpsat=psfsharpsat)

            iter_history.append({
                "titer": int(titer),
                "sigim": sigim.astype('f4').copy(),
                "modelsigim": modelsigim.astype('f4').copy(),
                "xn": xn.copy(),
                "yn": yn.copy(),
            })

            # print("peakfind:", time.time()-t_pk)

            # Remove duplicates
            if len(xa) > 0 and len(xn) > 0:
                keep = neighbor_dist(xn, yn, xa, ya) > 1.5
                xn, yn = (c[keep] for c in (xn, yn))

            # Add bright-star list if provided
            if (titer == 0) and (blist is not None):
                xnb, ynb = add_bright_stars(xn, yn, blist, images[0])
                xn = np.concatenate([xn, xnb]).astype('f4')
                yn = np.concatenate([yn, ynb]).astype('f4')

            # Append new detections
            xa = np.concatenate([xa, xn]).astype('f4')
            ya = np.concatenate([ya, yn]).astype('f4')
            passno = np.concatenate([passno, np.zeros(len(xn), dtype='i4') + titer])
        else:
            xn, yn = np.zeros(0, dtype='f4'), np.zeros(0, dtype='f4')

        # Stopping criterion
        if titer != lastiter:
            if (titer == maxiter-1) or (
                (titer >= miniter-1 and len(xn) < fewstars)) or (
                len(xa) > maxstars):
                lastiter = titer + 1

        # Prepare PSFs, fluxes, and skys
        minsz_b, sz_b = [], []
        for b in range(B):
            resid = images[b] - hskys[b] - mskys[b]
            # we probably don't want the sizes to change very much.  hsky certainly
            # will change a bit from iteration to iteration, though.
            sz = get_sizes(xa, ya, resid, weight=weights[b], blist=blist)
            sz_b.append(sz)
            minsz_b.append(np.min(sz) if len(sz) else 19)

        N = len(xa)
        N_old = len(xa) - len(xn)
        N_new = len(xn)
        
        if guessflux is not None:
            guessflux_b = guessflux.reshape(B, N_old)          # split old guesses per band
            zeros_new   = np.zeros(N_new, dtype=guessflux.dtype)
            guess = np.concatenate([
                np.concatenate([guessflux_b[b], zeros_new])    # pad each band with zeros for new sources
                for b in range(B)
            ])
        else:
            guess = None
        skys = hskys if titer >= 2 else lskys
        
        # in final iteration, no longer allow shifting locations; just fit centroids.
        tpsfderiv = psfderiv if lastiter != titer else False
        n_psfcomps = 3 if tpsfderiv else 1

        # PSF stamps per band
        psf_stamps = [[np.zeros((len(xa), minsz_b[b], minsz_b[b]), dtype='f4') for _ in range(n_psfcomps)] for b in range(B)]

        # Initialize flux array: [B*N] + [2*N shared derivs if enabled] + [B*nskypar sky]
        ncol_global = B*N + (2*N if tpsfderiv else 0) + B*nskypar
        flux = np.zeros(ncol_global, dtype='f4')

        # Subregion loop
        if verbose:
            subreg_iter = 0
            t0 = time.time()
            print("Starting subregion iterations")

        for (bdxf, bdxl, bdxaf, bdxal, bdyf, bdyl, bdyaf, bdyal) in subregs:
            if verbose:
                print(f"Subregion iteration {subreg_iter} starting; dt={time.time()-t0}", flush=True)
                subreg_iter += 1

            # Sources in this subregion (shared positions across all bands)
            mbda = in_bounds(xa, ya, [bdxaf-0.5, bdxal-0.5], [bdyaf-0.5, bdyal-0.5])
            if not np.any(mbda): 
                continue
            mbd = in_bounds(xa, ya, [bdxf-0.5, bdxl-0.5], [bdyf-0.5, bdyl-0.5])
        
            # Extract cutouts per band
            sall = np.s_[bdxaf:bdxal, bdyaf:bdyal]   # ALL region slice (big)
            spri = np.s_[bdxf:bdxl, bdyf:bdyl]       # PRIMARY region slice (small)
            dx, dy = bdxal-bdxaf, bdyal-bdyaf
            sfit = np.s_[bdxf-bdxaf:dx+bdxl-bdxal, bdyf-bdyaf:dy+bdyl-bdyal]   #cutout coordinates of the PRIMARY

            # Build PSFs, prepare inputs for fitting
            psfsbda = [build_psf_list(xa[mbda], ya[mbda], psfs[b], sz_b[b][mbda], psfderiv=tpsfderiv) for b in range(B)]
            imgs_tile    = [images[b][sall]  - skys[b][sall] for b in range(B)]
            weights_tile = [weights[b][sall] for b in range(B)]
            
            if (bdxf, bdyf) not in skypar:
                skypar[(bdxf, bdyf)] = np.zeros(B * nskypar, dtype='f4')

            # Prepare guesses if given
            if guess is not None:
                ind = np.flatnonzero(mbda)   # fitted sources (reuse same 'ind')
                guess_local_flux = guess[:B*N].reshape(B, N)[:, ind].ravel(order='C')
                guessmbda = np.concatenate([guess_local_flux, skypar[(bdxf, bdyf)]])
            else: guessmbda = None

            # t_fo = time.time()
            tflux, tmodels, tmskys = fit_once(
                imgs_tile, xa[mbda]-bdxaf, ya[mbda]-bdyaf, psfsbda,
                psfderiv=tpsfderiv, weights=weights_tile, guess=guessmbda,
                nskyx=nskyx, nskyy=nskyy)

            if np.all(np.isnan(tmodels)):
                raise ValueError("Model is all NaNs")
            
            # print("fit_once:", time.time()-t_fo)

            # Fit is done on ALL-region sources (primary + boundary) to avoid edge effects,
            # but we only commit parameters only for PRIMARY sources so each source is written once.
            # ind_all = np.flatnonzero(mbda)   # sources included in the fit (ALL region = PRIMARY + BOUNDARY)
            ind_pri_full = np.flatnonzero(mbd)        # sources owned by this tile (PRIMARY region)
            ind_pri_local = np.flatnonzero(mbd[mbda])  # Map primary sources into local ALL coordinates
            
            N_local = np.sum(mbda)  # number of sources fit in this tile (ALL region)
            if N_local>0:
                if tpsfderiv:  
                    # Store (dx,dy) (if present): block starts at B*N globally and B*N_local locally; interleaved per source
                    for d in range(2):
                        flux[B*N + 2*ind_pri_full + d] = tflux[0][B*N_local + 2*ind_pri_local + d]

                for b in range(B):
                    models[b][spri] = tmodels[b][sfit]
                    mskys[b][spri] = tmskys[b][sfit]
                    flux[b*N + ind_pri_full] = tflux[0][b*N_local + ind_pri_local]
                    for k in range(n_psfcomps):
                        psf_stamps[b][k][ind_pri_full] = np.stack([psfmod.central_stamp(psfsbda[b][k][tind], minsz_b[b]) for tind in ind_pri_local], axis=0).astype('f4')

            # Update per-tile sky
            if nskypar > 0:
                skypar[(bdxf, bdyf)] = tflux[0][-B*nskypar:].copy()

            # try to free memory!  Not sure where the circular reference
            # could be, but this makes a factor of a few difference
            # in peak memory usage on fields with lots of stars with
            # large models...

            del psfsbda
            gc.collect()
                
        # Build stamps 
        # t_cd = time.time()
        stamps = build_source_stamps(
            xa, ya, psf_stamps, flux,
            [images[b]-(skys[b]+mskys[b]) for b in range(B)],         # data- sky
            [images[b]-models[b]-skys[b] for b in range(B)],          # subtracted residual
            [weights[b] for b in range(B)])
        # print("centroids:", time.time()-t_cd)

        # Final iteration: compute stats
        if titer == lastiter:
            stats = compute_stats(xa-np.round(xa), ya-np.round(ya), stamps, flux)
            if dq is not None:
                stats['flags'] = extract_im(xa, ya, dq).astype('i4')
            for b in range(B):
                stats[f'sky_b{b}'] = extract_im(xa, ya, skys[b]+mskys[b]).astype('f4')
            break

        # Centroiding step (shared across bands)
        N = len(xa)
        if N > 0 and psfderiv:
            xcen = flux[B*N:B*N+2*N:2].astype('f4')
            ycen = flux[B*N+1:B*N+2*N:2].astype('f4')

            # ----- Convert the shared LSQR "derivative" parameters into actual pixel shifts. -----
            # In the design matrix, the dPSF/dx and dPSF/dy columns are scaled by a flux guess:
            #   - for old sources, guessflux ~ true flux, so the solved coefficients are already dx,dy pixel shifts
            #   - for newly detected sources, guessflux = 1, so the solved coefficients are flux*dx, flux*dy
            # Therefore, for new sources only, we divide by an estimated flux (best-SNR band) to recover dx,dy.

            flux_b = flux[:B*N].reshape(B, N)
        
            fluxunc_b = np.sum(stamps[2]**2 * stamps[3]**2, axis=(2, 3))
            fluxunc_b = (fluxunc_b + (fluxunc_b == 0)*1e-20)**(-0.5)
            snr_b = flux_b / fluxunc_b
            best_band = np.argmax(snr_b, axis=0)   # shape (N,)
        
            flux_snr = flux_b[best_band, np.arange(N)]
            flux_snr = flux_snr + (flux_snr == 0)*1e-20

            new = (passno == titer)

            # Only new sources need normalization by flux
            xcen[new] /= flux_snr[new]
            ycen[new] /= flux_snr[new]

            # # Only apply centroid update if SNR is decent?
            # snr_joint = np.sqrt(np.sum(np.clip(snr_b, 0, np.inf)**2, axis=0))
            # good = snr_joint > 5.0   # 3 gives same result
            
            # xcen[~good] = 0.0
            # ycen[~good] = 0.0


            print(f"[iter {titer}] var(xcen)={mad_std(xcen):.3e}, var(ycen)={mad_std(ycen):.3e}, "f"rms(dx)={np.sqrt(np.mean(xcen**2 + ycen**2)):.3e}")
        else:
            xcen = np.zeros(N, dtype='f4')
            ycen = np.zeros(N, dtype='f4')
        
        if refit_psf and N > 0:
            for b in range(B):
                impsf_b, im_b, model_b, weight_b, psfonly_b = (stamps[0][b], stamps[1][b], stamps[2][b], stamps[3][b], stamps[4][b])
                psfs[b], xa, ya = refit_psf_from_stamps(
                    psfs[b], xa, ya, xcen, ycen,
                    (impsf_b, im_b, model_b, weight_b, psfonly_b),
                    name=(titer, f"{ccd}_b{b}" if ccd else f"b{b}"),
                    plot=plot)

        # Enforce maximum centroid step and update (x, y)
        maxstep = 1.0 #if derivcentroids else 3.0
        dcen = np.sqrt(xcen**2 + ycen**2)
        m = dcen > maxstep
        if np.any(m): 
            xcen[m] /= (dcen[m] + 1e-20); ycen[m] /= (dcen[m] + 1e-20)
        xa = np.clip(xa + xcen, -0.499, shape[0]-0.501)
        ya = np.clip(ya + ycen, -0.499, shape[1]-0.501)

        # Flux uncertainty and pruning
        fluxunc_b = np.sum(stamps[2]**2 * stamps[3]**2, axis=(2, 3))
        fluxunc_b = (fluxunc_b + (fluxunc_b == 0)*1e-20)**(-0.5)
        # for very bright stars, fluxunc_b is unreliable because the entire
        # (small) stamp is saturated.
        # these stars all have very bright inferred fluxes
        # i.e., 50k saturates, so we can cut there.
        
        guessflux_b = np.vstack([flux[b*N:(b+1)*N] for b in range(B)])
        B, N = guessflux_b.shape

        # if B == 1:
        #     #single-band behavior (no absolute value, flux-based isolation)
        #     snr_legacy = guessflux_b[0] / (fluxunc_b[0] + 1e-20)   # signed SNR
        #     brightenough = (snr_legacy > threshold*3/5.) | (guessflux_b[0] > 1e5)
        #     isolatedenough = cull_near(xa, ya, guessflux_b[0])     # use flux, not SNR. Trying to mimic the original version here. Else use snr_legacy
        # else:
        #     # Multiband behavior (quadrature SNR)
        snr_b = guessflux_b / (fluxunc_b + 1e-20)
        snr_joint = np.sqrt(np.sum(np.clip(snr_b, 0, np.inf)**2, axis=0))          # |SNR| by construction
        bright_joint = snr_joint > threshold*3/5.
        bright_satur = np.any(guessflux_b > 1e5, axis=0)
        brightenough = bright_joint | bright_satur
        isolatedenough = cull_near(xa, ya, snr_joint)
        # print("removed close:", np.sum(~isolatedenough), "removed faint:", np.sum(~brightenough & isolatedenough))
            
        keep = brightenough & isolatedenough

        mad_x  = mad_std(xcen[keep])
        mad_y  = mad_std(ycen[keep])
        rms_xy = np.sqrt(np.mean(xcen[keep]**2 + ycen[keep]**2))

        # iter_history.append(
        #     {
        #         "titer": int(titer),
        #         "xcen": xcen[keep].copy(),
        #         "ycen": ycen[keep].copy(),
        #         "mad_x": float(mad_x),
        #         "mad_y": float(mad_y),
        #         "rms_xy": float(rms_xy),
        #     }
        # )
        
        xa, ya = xa[keep], ya[keep]
        passno = passno[keep]
        guessflux_b = guessflux_b[:, keep]
        guessflux = guessflux_b.ravel(order='C')

        # iter_history.append({
        #     "titer": int(titer),
        #     "xa": xa.copy(),
        #     "ya": ya.copy(),
        #     "model": [ (models[b] + skys[b]).copy() for b in range(B) ],
        #     "sky":   [ (skys[b] + mskys[b]).copy() for b in range(B) ],
        # })

        if verbose:
            print('Extension %s, iteration %2d, found %6d sources; %4d close and '
                  '%4d faint sources removed.' %
                  (ccd, titer+1, len(xn),
                   np.sum(~isolatedenough),
                   np.sum(~brightenough & isolatedenough)))

        # should probably also subtract these stars from the model image
        # which is used for peak finding.  But the faint stars should
        # make little difference?    

    flux_b = [flux[b*N:(b+1)*N] for b in range(B)]
    stars = OrderedDict(
        [('x', xa.astype('f8')), ('y', ya.astype('f8'))] +
        [(f'flux_b{b}', flux_b[b].astype('f4')) for b in range(B)] +
        [('passno', passno)] +
        [(f, stats[f]) for f in stats]
    )

    # compress single-band output to drop multiband-only fields (*_joint) and rename *_b0
    if B == 1:
        compressed = OrderedDict()
        for k, v in stars.items():
            if k.endswith('_joint'):
                continue          # drop joint fields in single-band case
            if k.endswith('_b0'):
                compressed[k[:-3]] = v   # rename flux_b0 -> flux, etc.
            else:
                compressed[k] = v
        stars = compressed
    
    dtype = np.dtype({'names': list(stars.keys()),
                      'formats': [stars[n].dtype for n in stars.keys()]})
    stars = np.fromiter(zip(*stars.values()), dtype=dtype, count=len(stars['x']))

    # print(f"[fit_im] total time: {time.time() - t0_all:.3f} s")
    result = {
    "stars": stars,
    "model": [models[b] + skys[b] for b in range(B)],
    "sky":   [skys[b] + mskys[b] for b in range(B)],
    "psfs": psfs,
    "iter_history": iter_history,
    }
    return result
    


def compute_stats(xs, ys, stamps, flux_flat):
    """
    Compute per-source diagnostics (errors, chi2, flux fractions, morphology)from postage stamp cutouts.
      - single-band inputs (stamps: tuple of 5 arrays, each (N,S,S))
      - multiband inputs (stamps: tuple of 5 arrays, each (B,N,S,S))

    Parameters
    ----------
    xs, ys : ndarray of length N
        Source centroid offsets (subpixel x,y positions).
    stamps : tuple
        (impsfstack, imstack, psfstack, weightstack, psf*flux) for 1 or many bands.
        For a single band:
            impsfstack : ndarray (N,S,S)
                Neighbor-subtracted data stamps (model + residual).
            psfstack : ndarray (N,S,S)
                Model PSF × fitted flux stamps for each source NOT the psf template stamp which stamp[4] element
            weightstack : ndarray (N,S,S)
                Weight (inverse noise) stamps.
            imstack : ndarray (N,S,S)
                Raw image data stamps.
    flux_flat : ndarray
        Flattened flux amplitude vector from fit_once().
        Layout:
          [f^(1)_1...f^(1)_N, f^(2)_1...f^(2)_N, ..., f^(B)_1...f^(B)_N,
           (optional) dx, dy block, (optional) sky params]

    Returns
    -------
    stats : OrderedDict
        Per-band and joint diagnostics. Dictionary of per-source measurements:
        - dx, dy : position uncertainties
        - dflux  : flux uncertainty
        - qf     : quality factor (fraction of valid PSF footprint)
        - rchi2  : reduced chi2 of fit
        - fracflux : fraction of flux modeled
        - fluxlbs/dfluxlbs : large-box summed flux and error
        - fluxiso,xiso,yiso : isophotal flux and centroid
        - fwhm   : effective PSF width
        - spread_model/dspread_model : star/galaxy separator
    """

    from collections import OrderedDict

    # Detect single vs multiband case
    impsfstack, imstack, psfstack, weightstack, psfflux = stamps
    if impsfstack.ndim == 3:
        # Single-band → wrap into multiband shape (1,B,N,S,S)
        impsfstack = impsfstack[None, ...]
        imstack    = imstack[None, ...]
        psfstack   = psfstack[None, ...]
        weightstack= weightstack[None, ...]
        psfflux    = psfflux[None, ...]
    B, N, S, _ = psfstack.shape

    # detect if derivatives present in flux vector
    base_flux_len = B * N
    has_deriv = len(flux_flat) >= base_flux_len + 2 * N

    # slice out only per-band fluxes
    flux_blocks = [flux_flat[b * N:(b + 1) * N] for b in range(B)]

    out = OrderedDict()
    dx_list, dy_list, qf_list, rchi2_list, dflux_list, flux_list = [], [], [], [], [], []

    for b in range(B):
        flux_b = flux_blocks[b]
        impsf_b, im_b, psf_b, w_b = impsfstack[b], imstack[b], psfstack[b], weightstack[b]

        # Residuals = (image data - model)
        residstack = impsf_b - psf_b

        # Normalize PSF stamps so they integrate to 1
        norm = np.sum(psf_b, axis=(1, 2))
        psf_b = psf_b / (norm + (norm == 0)).reshape(-1, 1, 1)

        # Quality factor how “complete” the measurement is = fraction of the PSF footprint that overlaps with good/unmasked pixels.
        qf = np.sum(psf_b * (w_b > 0), axis=(1, 2))

        # Flux uncertainty = inverse sqrt(sum(PSF^2 * weight^2))
        fluxunc = np.sum(psf_b**2 * w_b**2, axis=(1, 2))
        fluxunc = fluxunc + (fluxunc == 0) * 1e-20
        fluxunc = (fluxunc ** -0.5).astype('f4')

        if np.any(fluxunc > 1e8):
            print(f"[Band {b}] compute_stats: huge fluxunc", fluxunc.max())

        # mask_empty = np.sum(weightstack > 0, axis=(1,2)) == 0
        # if np.any(mask_empty):
        #     print("DEBUG: sources with empty weights:", np.nonzero(mask_empty)[0])

        # Derivatives of normalized PSF -> for position uncertainty
        psfderiv = np.gradient(-psf_b, axis=(1, 2))
        posunc = [np.zeros(len(qf), dtype='f4'), np.zeros(len(qf), dtype='f4')]
        for i, p in enumerate(psfderiv):
            # Position error sigma_x,sigma_y = 1/sqrt(sum((dPSF * flux * weight)^2))
            dp = np.sum((p * w_b * flux_b[:, None, None]) ** 2, axis=(1, 2))
            dp = dp + (dp == 0) * 1e-40
            posunc[i][:] = dp ** -0.5

        # Reduced chi2 of the fit, weighted by PSF footprint
        rchi2 = np.sum(residstack ** 2 * w_b ** 2 * psf_b, axis=(1, 2))
        rchi2 /= (qf + (qf == 0) * 1e-20)

        # Flux fraction: how much of the flux is explained by the model. Best for point sources, high S/N.
        fracfluxn = np.sum(impsf_b * (w_b > 0) * psf_b, axis=(1, 2))
        fracfluxd = np.sum(im_b * (w_b > 0) * psf_b, axis=(1, 2))
        fracflux = (fracfluxn / (fracfluxd + (fracfluxd == 0) * 1e-20)).astype('f4')

        # Alternative flux estimators
        # - LBS flux: large box summation - big box aperture flux. 
        # Robust sanity check (if PSF model is wrong, we notice a discrepancy).        
        fluxlbs, dfluxlbs = compute_lbs_flux(impsf_b, psf_b, w_b, flux_b / (norm + (norm == 0)))
        fluxlbs, dfluxlbs = fluxlbs.astype('f4'), dfluxlbs.astype('f4')

        # "isolated" flux; fit flux & centroid as if the star were a single star, 
        # based on imaged where neighbors have been subtracted from best simultaneous fit.
        fluxiso, xiso, yiso = compute_iso_fit(impsf_b, psf_b, w_b, flux_b / (norm + (norm == 0)), psfderiv)

        # Effective PSF FWHM for this source
        fwhm = psfmod.neff_fwhm(psf_b).astype('f4')

        # Spread model (star/galaxy separator)
        spread, dspread = spread_model(impsf_b, psf_b, w_b)

        out.update({
            f"dx_b{b}": posunc[0].astype('f4'),
            f"dy_b{b}": posunc[1].astype('f4'),
            f"dflux_b{b}": fluxunc,
            f"qf_b{b}": qf,
            f"rchi2_b{b}": rchi2.astype('f4'),
            f"fracflux_b{b}": fracflux,
            f"fluxlbs_b{b}": fluxlbs, f"dfluxlbs_b{b}": dfluxlbs,
            f"fwhm_b{b}": fwhm,
            f"spread_model_b{b}": spread, f"dspread_model_b{b}": dspread,
            f"fluxiso_b{b}": fluxiso.astype('f4'), 
            f"xiso_b{b}":    xiso.astype('f4'), 
            f"yiso_b{b}":    yiso.astype('f4')
        })

        # accumulate for joint metrics
        dx_list.append(posunc[0])
        dy_list.append(posunc[1])
        qf_list.append(qf)
        rchi2_list.append(rchi2)
        dflux_list.append(fluxunc)
        flux_list.append(flux_b)

    # === Joint stats across bands ==================================
    dx_arr, dy_arr = np.vstack(dx_list), np.vstack(dy_list)
    qf_arr, rchi2_arr = np.vstack(qf_list), np.vstack(rchi2_list)
    dflux_arr, flux_arr = np.vstack(dflux_list), np.vstack(flux_list)

    # 1) Joint reduced chi2 = Sum_b (rchi2_b * qf_b) / Sum_b qf_b
    num_joint = np.sum(rchi2_arr * qf_arr, axis=0)
    den_joint = np.sum(qf_arr, axis=0)
    rchi2_joint = num_joint / (den_joint + (den_joint == 0) * 1e-20)

    # 2) Joint positional uncertainties via inverse-variance sum
    inv2_dx = np.sum(1.0 / (dx_arr ** 2 + 1e-20), axis=0)
    inv2_dy = np.sum(1.0 / (dy_arr ** 2 + 1e-20), axis=0)
    dx_joint = (1.0 / (inv2_dx + (inv2_dx == 0) * 1e-20)) ** 0.5
    dy_joint = (1.0 / (inv2_dy + (inv2_dy == 0) * 1e-20)) ** 0.5

    # 3) Joint SNR = sqrt( SUM over (F_b / dF_b)^2 )
    snr_bands = flux_arr / (dflux_arr + 1e-20)
    snr_joint = np.sqrt(np.sum(snr_bands ** 2, axis=0))

    out["dx_joint"] = dx_joint.astype("f4")
    out["dy_joint"] = dy_joint.astype("f4")
    out["rchi2_joint"] = rchi2_joint.astype("f4")
    out["snr_joint"] = snr_joint.astype("f4")

    # Debug prints for extreme values
    bad = np.where(np.any(dflux_arr > 1e8, axis=0))[0]
    if len(bad) > 0:
        print("[compute_stats] Warning: very large dflux values for sources:", bad[:10])
        for b in range(B):
            print(f" Band {b}: flux={flux_blocks[b][bad][:5]}, qf={qf_list[b][bad][:5]}")

    return out

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
                                            np.eye(2), 0, 0, psfstack)
    GWp = np.sum(expgalstack*weightstack**2*impsfstack, axis=(1, 2))
    PWp = np.sum(psfstack*weightstack**2*impsfstack, axis=(1, 2))
    GWP = np.sum(expgalstack*weightstack**2*psfstack, axis=(1, 2))
    PWP = np.sum(psfstack**2*weightstack**2, axis=(1, 2))
    GWG = np.sum(expgalstack**2*weightstack**2, axis=(1, 2))
    spread = (GWp/(PWp+(PWp == 0)) - GWP/(PWP+(PWP == 0)))
    dspread = np.sqrt(np.clip(
        PWp**2*GWG + GWp**2*PWP - 2*GWp*PWp*GWP, 0, np.inf)
                         /(PWp + (PWp == 0))**4)
    return spread, dspread


def extract_im(xa, ya, im, sentinel=999):
    m = np.ones(len(xa), dtype='bool')
    for c, sz in zip((xa, ya), im.shape):
        m = m & (c > -0.5) & (c < sz - 0.5)
    res = np.zeros(len(xa), dtype=im.dtype)
    res[~m] = sentinel
    xp, yp = (np.round(c[m]).astype('i4') for c in (xa, ya))
    res[m] = im[xp, yp]
    return res


def compute_lbs_flux(stamp, psf, isig, apcor):
    sumisig2 = np.sum(isig**2, axis=(1, 2))
    sumpsf2isig2 = np.sum(psf*psf*isig**2, axis=(1, 2))
    sumpsfisig2 = np.sum(psf*isig**2, axis=(1, 2))
    det = np.clip(sumisig2*sumpsf2isig2 - sumpsfisig2**2, 0, np.inf)
    det = det + (det == 0)
    unc = np.sqrt(sumisig2/det)
    flux = (sumisig2*np.sum(psf*stamp*isig**2, axis=(1, 2)) -
            sumpsfisig2*np.sum(stamp*isig**2, axis=(1, 2)))/det
    flux *= apcor
    unc *= apcor
    return flux, unc


def compute_iso_fit(impsfstack, psfstack, weightstack, apcor, psfderiv):
    nstar = len(impsfstack)
    par = np.zeros((nstar, 3), dtype='f4')
    for i in range(len(impsfstack)):
        aa = np.array([psfstack[i]*weightstack[i],
                          psfderiv[0][i]*weightstack[i],
                          psfderiv[1][i]*weightstack[i]])
        aa = aa.reshape(3, -1).T
        par[i, :] = np.linalg.lstsq(
            aa, (impsfstack[i]*weightstack[i]).reshape(-1), rcond=None)[0]
    zeroflux = par[:, 0] == 0
    return (par[:, 0],
            (1-zeroflux)*par[:, 1]/(par[:, 0]+zeroflux),
            (1-zeroflux)*par[:, 2]/(par[:, 0]+zeroflux))


def sky_model_basis(i, j, nskyx, nskyy, nx, ny):
    from crowdsource import basisspline
    if (nskyx < 3) or (nskyy < 3):
        raise ValueError('Invalid sky model.')
    expandx = (nskyx-1.)/(3-1)
    expandy = (nskyy-1.)/(3-1)
    xg = -expandx/3. + i*2/3.*expandx/(nskyx-1.)
    yg = -expandy/3. + j*2/3.*expandy/(nskyy-1.)
    x = np.linspace(-expandx/3.+1/6., expandx/3.-1/6., nx).reshape(-1, 1)
    y = np.linspace(-expandy/3.+1/6., expandy/3.-1/6., ny).reshape(1, -1)
    return basisspline.basis2dq(x-xg, y-yg)


def sky_model(coeff, nx, ny):
    # minimum sky model: if we want to use the quadratic basis functions we
    # implemented, and we want to allow a constant sky over the frame, then we
    # need at least 9 basis polynomials: [-0.5, 0.5, 1.5] x [-0.5, 0.5, 1.5].
    nskyx, nskyy = coeff.shape
    if (coeff.shape[0] == 1) & (coeff.shape[1]) == 1:
        return coeff[0, 0]*np.ones((nx, ny), dtype='f4')
    if (coeff.shape[0] < 3) or (coeff.shape[1]) < 3:
        raise ValueError('Not obvious what to do for <3')
    im = np.zeros((nx, ny), dtype='f4')
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
    xloc = [np.arange(nx*ny, dtype='i4')]*nskypar
    # for the moment, don't take advantage of the bounded support.
    yloc = [i*np.ones((nx, ny), dtype='i4').ravel()
            for i in range(nskypar)]
    if (nskyx == 1) & (nskyy == 1):
        values = [(np.ones((nx, ny), dtype='f4')*weight).ravel()
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
        return np.ones(len(x), dtype='bool')
    m1, m2, dist = match_xy(x, y, x, y, neighbors=6)
    m = (dist < 1) & (flux[m1] < flux[m2]) & (m1 != m2)
    keep = np.ones(len(x), dtype='bool')
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
    vec1 = np.array([x1, y1]).T
    vec2 = np.array([x2, y2]).T
    kdt = cKDTree(vec1)
    dist, idx = kdt.query(vec2, neighbors)
    m1 = idx.ravel()
    m2 = np.repeat(np.arange(len(vec2), dtype='i4'), neighbors)
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
            mindist2 = np.min((x-xa)**2 + (y-ya)**2)
        else:
            mindist2 = 9999
        if mindist2 > 5**2:
            xout.append(x)
            yout.append(y)
    return (np.array(xout, dtype='f4'), np.array(yout, dtype='f4'))


# This is almost entirely deprecated for the psf.py module... go look there.
def find_psf(xcen, shiftx, ycen, shifty, psfstack, weightstack,
             imstack, stampsz=59, nkeep=100):
    """Find PSF from stamps."""
    # let's just go ahead and correlate the noise
    xr = np.round(shiftx)
    yr = np.round(shifty)
    psfqf = (np.sum(psfstack*(weightstack > 0), axis=(1, 2)) /
             np.sum(psfstack, axis=(1, 2)))
    totalflux = np.sum(psfstack, axis=(1, 2))
    timflux = np.sum(imstack, axis=(1, 2))
    toneflux = np.sum(psfstack, axis=(1, 2))
    tmedflux = np.median(psfstack, axis=(1, 2))
    tfracflux = toneflux / np.clip(timflux, 100, np.inf)
    tfracflux2 = ((toneflux-tmedflux*psfstack.shape[1]*psfstack.shape[2]) /
                  np.clip(timflux, 100, np.inf))
    okpsf = ((np.abs(psfqf - 1) < 0.03) &
             (tfracflux > 0.5) & (tfracflux2 > 0.2))
    if np.sum(okpsf) > 0:
        shiftxm = np.median(shiftx[okpsf])
        shiftym = np.median(shifty[okpsf])
        okpsf = (okpsf &
                 (np.abs(shiftx-shiftxm) < 1.) &
                 (np.abs(shifty-shiftym) < 1.))
    if np.sum(okpsf) <= 5:
        print('Fewer than 5 stars accepted in image, keeping original PSF')
        return None
    if np.sum(okpsf) > nkeep:
        okpsf = okpsf & (totalflux > -np.sort(-totalflux[okpsf])[nkeep-1])
    psfstack = psfstack[okpsf, :, :]
    weightstack = weightstack[okpsf, :, :]
    totalflux = totalflux[okpsf]
    xcen = xcen[okpsf]
    ycen = ycen[okpsf]
    shiftx = shiftx[okpsf]
    shifty = shifty[okpsf]
    for i in range(psfstack.shape[0]):
        psfstack[i, :, :] = shift(psfstack[i, :, :], [-shiftx[i], -shifty[i]])
        if (np.abs(xr[i]) > 0) or (np.abs(yr[i]) > 0):
            weightstack[i, :, :] = shift(weightstack[i, :, :],
                                         [-xr[i], -yr[i]],
                                         mode='constant', cval=0.)
        # our best guess as to the PSFs & their weights
    # select some reasonable sample of the PSFs
    totalflux = np.sum(psfstack, axis=(1, 2))
    psfstack /= totalflux.reshape(-1, 1, 1)
    weightstack *= totalflux.reshape(-1, 1, 1)
    tpsf = np.median(psfstack, axis=0)
    tpsf = psfmod.center_psf(tpsf)
    if tpsf.shape == stampsz:
        return tpsf
    xc = np.arange(tpsf.shape[0]).reshape(-1, 1)-tpsf.shape[0]//2
    yc = xc.reshape(1, -1)
    rc = np.sqrt(xc**2.+yc**2.)
    stampszo2 = psfstack[0].shape[0] // 2
    wt = np.clip((stampszo2+1-rc)/4., 0., 1.)
    overlap = (wt != 1) & (wt != 0)

    def objective(par):
        mod = psfmod.moffat_psf(par[0], beta=2.5, xy=par[2], yy=par[3],
                                deriv=False, stampsz=tpsf.shape[0])
        mod /= np.sum(mod)
        return ((tpsf-mod)[overlap]).reshape(-1)
    from scipy.optimize import leastsq
    par = leastsq(objective, [4., 3., 0., 1.])[0]
    modpsf = psfmod.moffat_psf(par[0], beta=2.5, xy=par[2], yy=par[3],
                               deriv=False, stampsz=stampsz)
    modpsf /= np.sum(psfmod.central_stamp(modpsf))
    npsf = modpsf.copy()
    npsfcen = psfmod.central_stamp(npsf, tpsf.shape[0])
    npsfcen[:, :] = tpsf*wt+(1-wt)*npsfcen[:, :]
    npsf /= np.sum(npsf)
    return psfmod.SimplePSF(npsf, normalize=-1)


def subregions(shape, nx, ny, overlap=149):
    # ugh.  I guess we want:
    # starts and ends of each _primary_ fit region
    # starts and ends of each _entire_ fit region
    # should be nothing else?
    # need this for both x and y: 8 things to return.
    nx = nx if nx > 0 else 1
    ny = ny if ny > 0 else 1
    bdx = np.round(np.linspace(0, shape[0], nx+1)).astype('i4')
    bdlx = np.clip(bdx - overlap, 0, shape[0])
    bdrx = np.clip(bdx + overlap, 0, shape[0])
    bdy = np.round(np.linspace(0, shape[1], ny+1)).astype('i4')
    bdly = np.clip(bdy - overlap, 0, shape[1])
    bdry = np.clip(bdy + overlap, 0, shape[1])
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
