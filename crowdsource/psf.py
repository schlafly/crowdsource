"""PSF module

This module provides a class implementing a spatially varying PSF.

Intended usage is:
>>> unknown ***
"""

import pdb
import numpy
import os

if 'DECAM_DIR' not in os.environ:
    decam_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"decam_dir")
    os.environ['DECAM_DIR'] = decam_dir


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


def central_stamp(stamp, censize=19):
    if censize is None:
        censize = 19
    stampsz = stamp.shape[-1]
    if ((stampsz % 2) == 0) | ((censize % 2) == 0):
        pdb.set_trace()
    if stampsz == censize:
        return stamp
    elif stampsz > censize:
        trim = (stamp.shape[-1] - censize)//2
        f = trim
        l = stampsz - trim
        return stamp[..., f:l, f:l]
    else:
        ret = numpy.zeros(stamp.shape[:-2]+(censize, censize), dtype='f4')
        central_stamp(ret, censize=stampsz)[..., :, :] = stamp
        return ret


def neff_fwhm(stamp):
    """FWHM-like quantity derived from N_eff = numpy.sum(PSF**2.)**-1"""
    norm = numpy.sum(stamp, axis=(-1, -2), keepdims=True)
    return 1.18 * (numpy.pi*numpy.sum((stamp/norm)**2., axis=(-1, -2)))**(-0.5)


def fwhm_neff(fwhm):
    return (fwhm/1.18)**2*numpy.pi


def gaussian_psf(fwhm, stampsz=19, deriv=True, shift=[0, 0]):
    """Create Gaussian psf & derivatives for a given fwhm and stamp size.

    Args:
        fwhm (float): the full width at half maximum
        stampsz (int): the return psf stamps are [stampsz, stampsz] in size
        deriv (bool): return derivatives?
        shift (float, float): shift centroid by this amount in x, y

    Returns:
        (psf, dpsfdx, dpsfdy)
        psf (ndarray[stampsz, stampsz]): the psf stamp
        dpsfdx (ndarray[stampsz, stampsz]): the x-derivative of the PSF
        dpsfdy (ndarray[stampsz, stampsz]): the y-derivative of the PSF
    """
    sigma = fwhm / numpy.sqrt(8*numpy.log(2))
    stampszo2 = stampsz // 2
    parshape = numpy.broadcast(fwhm, shift[0], shift[1]).shape
    if len(parshape) > 0:
        sigma, shift[0], shift[1] = (numpy.atleast_1d(q).reshape(-1, 1, 1)
                                     for q in (sigma, shift[0], shift[1]))
    xc = numpy.arange(stampsz, dtype='f4')-stampszo2
    yc = xc.copy()
    xc = xc.reshape(-1, 1)-shift[0]
    yc = yc.reshape(1, -1)-shift[1]
    psf = numpy.exp(-(xc**2. + yc**2.) /
                    2./sigma**2.).astype('f4')
    psf /= numpy.sum(psf[..., :, :])
    dpsfdx = xc/sigma**2.*psf
    dpsfdy = yc/sigma**2.*psf
    ret = psf
    if deriv:
        ret = (ret,) + (dpsfdx, dpsfdy)
    return ret


def moffat_psf(fwhm, beta=3., xy=0., yy=1., stampsz=19, deriv=True,
               shift=[0, 0]):
    """Create Moffat psf & derivatives for a given fwhm and stamp size.

    Args:
        fwhm (float): the full width at half maximum
        beta (float): beta parameter for Moffat distribution
        xy (float): xy coefficient (0 for uncorrelated)
        yy (float): yy coefficient (1 for FWHM_x == FWHM_y)
        stampsz (int): the returned psf stamps are [stampsz, stampsz] in size
        deriv (bool): return derivatives?
        shift (float, float): shift centroid by this amount in x, y

    Returns:
        (psf, dpsfdx, dpsfdy)
        psf (ndarray[stampsz, stampsz]): the psf stamp
        dpsfdx (ndarray[stampsz, stampsz]): the x-derivative of the PSF
        dpsfdy (ndarray[stampsz, stampsz]): the y-derivative of the PSF
    """
    if numpy.any(beta <= 1e-3):
        print('Warning: crazy values for beta in moffat_psf')
        beta = numpy.clip(beta, 1e-3, numpy.inf)
    alpha = fwhm/(2*numpy.sqrt(2**(1./beta)-1))
    stampszo2 = stampsz // 2
    xc = numpy.arange(stampsz, dtype='f4')-stampszo2
    parshape = numpy.broadcast(fwhm, beta, xy, yy, shift[0], shift[1]).shape
    xc = xc.reshape(-1, 1)
    yc = xc.copy().reshape(1, -1)
    if len(parshape) > 0:
        alpha, beta, xy, yy = (numpy.atleast_1d(q).reshape(-1, 1, 1)
                               for q in (alpha, beta, xy, yy))
        shift = list(shift)
        shift[0], shift[1] = (numpy.atleast_1d(q).reshape(-1, 1, 1)
                              for q in (shift[0], shift[1]))
    xc = xc - shift[0]
    yc = yc - shift[1]
    yy = numpy.abs(yy)
    rc2 = yy**(-0.5)*xc**2. + xy*xc*yc + yy**(0.5)*yc**2.
    # for bad xy, this can screw up and generate negative values.
    if numpy.any(rc2 < 0.):
        print('Warning: crazy xy and yy values to moffat_psf')
        rc2 = numpy.clip(rc2, 0., numpy.inf)
    rc = numpy.sqrt(rc2)
    psf = (beta - 1)/(numpy.pi * alpha**2.)*(1.+(rc**2./alpha**2.))**(-beta)
    ret = psf
    if deriv:
        dpsffac = (beta-1)/(numpy.pi*alpha**2.)*(beta)*(
            (1+(rc**2./alpha**2.))**(-beta-1))
        dpsfdx = dpsffac*2*xc/alpha
        dpsfdy = dpsffac*2*yc/alpha
        ret = (psf, dpsfdx, dpsfdy)
    return ret


def simple_centroid(psf, norm=True):
    stampsz = psf.shape[-1]
    stampszo2 = stampsz // 2
    xc = numpy.arange(stampsz, dtype='f4')-stampszo2
    xc = xc.reshape(-1, 1)
    yc = xc.copy().reshape(1, -1)
    denom = 1.
    if norm:
        denom = numpy.sum(psf, axis=(-1, -2))
    return (numpy.sum(xc*psf, axis=(-1, -2))/denom,
            numpy.sum(yc*psf, axis=(-1, -2))/denom)


def center_psf(psf, censize=None):
    """Center and normalize a psf; centroid is placed at center."""
    psf = psf.copy()
    cpsf = central_stamp(psf, censize=censize)
    for _ in range(3):
        xcen, ycen = simple_centroid(cpsf)
        psf[:, :] = shift(psf, [-xcen, -ycen],
                          output=numpy.dtype('f4'))
    psf /= numpy.sum(psf)
    psf = psf.astype('f4')
    return psf


class SimplePSF:
    def __init__(self, stamp, normalize=19):
        self.stamp = stamp
        if normalize > 0:
            norm = numpy.sum(central_stamp(stamp, censize=normalize))
            self.stamp /= norm
        self.deriv = numpy.gradient(-stamp)

    def render_model(self, x, y, stampsz=None):
        if stampsz is None:
            return self.stamp
        else:
            return central_stamp(self.stamp, censize=stampsz)

    def __call__(self, x, y, stampsz=None, deriv=False):
        parshape = numpy.broadcast(x, y).shape
        tparshape = parshape if len(parshape) > 0 else (1,)
        if stampsz is None:
            stampsz = self.stamp.shape[0]

        shiftx, shifty = (numpy.atleast_1d(q) - numpy.round(q) for q in (x, y))
        stamp = central_stamp(self.stamp, censize=stampsz)
        ret = numpy.zeros(tparshape+(stampsz, stampsz), dtype='f4')
        for i in range(ret.shape[0]):
            ret[i, :, :] = shift(stamp, (shiftx[i], shifty[i]))

        if deriv:
            dpsfdx = numpy.zeros_like(ret)
            dpsfdy = numpy.zeros_like(ret)
            dxstamp = central_stamp(self.deriv[0], censize=stampsz)
            dystamp = central_stamp(self.deriv[1], censize=stampsz)
            for i in range(ret.shape[0]):
                dpsfdx[i, :, :] = shift(dxstamp, (shiftx[i], shifty[i]))
                dpsfdy[i, :, :] = shift(dystamp, (shiftx[i], shifty[i]))

        if parshape != tparshape:
            ret = ret.reshape(stampsz, stampsz)
            if deriv:
                dpsfdx = dpsfdx.reshape(stampsz, stampsz)
                dpsfdy = dpsfdy.reshape(stampsz, stampsz)
        if deriv:
            ret = (ret, dpsfdx, dpsfdy)
        return ret

    def serialize(self, stampsz=None):
        stamp = self.stamp
        if stampsz is not None:
            stamp = central_stamp(self.stamp, stampsz)
        dtype = [('offset', '2f4'),
                 ('stamp', stamp.dtype, stamp.shape)]
        extrapar = getattr(self, 'extraparam', None)
        if extrapar is not None:
            dtype += extrapar.dtype.descr
        res = numpy.zeros(1, dtype=dtype)
        res['offset'][0, :] = getattr(self, 'offset', (0, 0))
        res['stamp'][0, ...] = stamp
        if getattr(self, 'extraparam', None) is not None:
            for name in extrapar.dtype.names:
                res[name][0, ...] = extrapar[name]
        return res



class MoffatPSF:
    def __init__(self, fwhm, beta, xy=0., yy=1., normalize=19):
        self.fwhm = fwhm
        self.beta = beta
        self.xy = xy
        self.yy = yy
        if normalize > 0:
            self.norm = numpy.sum(self.render_model(0, 0, stampsz=19))
        else:
            self.norm = 1

    def render_model(self, x, y, stampsz=59):
        res = moffat_psf(self.fwhm, beta=self.beta, xy=self.xy,
                         yy=self.yy, stampsz=stampsz)
        return res

    def __call__(self, x, y, stampsz=None, deriv=False):
        shiftx, shifty = (q - numpy.round(q) for q in (x, y))
        res = moffat_psf(self.fwhm, beta=self.beta, xy=self.xy,
                         yy=self.yy, stampsz=stampsz, deriv=deriv,
                         shift=(shiftx, shifty))
        if deriv:
            res = [r / self.norm for r in res]
        else:
            res = res / self.norm
        return res


class VariableMoffatPSF:
    def __init__(self, fwhm, beta, xy=0., yy=1., normalize=19):
        self.fwhm = numpy.atleast_2d(fwhm)
        self.beta = numpy.atleast_2d(beta)
        self.xy = numpy.atleast_2d(xy)
        self.yy = numpy.atleast_2d(yy)
        self.normalize = normalize

    def render_model(self, x, y, stampsz=59, deriv=False):
        from numpy.polynomial.polynomial import polyval2d
        x = x / 1000.
        y = y / 1000.
        fwhm = polyval2d(x, y, self.fwhm)
        beta = polyval2d(x, y, self.beta)
        xy = polyval2d(x, y, self.xy)
        yy = polyval2d(x, y, self.yy)
        return moffat_psf(fwhm, beta=beta, xy=xy,
                          yy=yy, stampsz=stampsz, deriv=deriv)

    def __call__(self, x, y, stampsz=59, deriv=False):
        from numpy.polynomial.polynomial import polyval2d
        shiftx, shifty = (q - numpy.round(q) for q in (x, y))
        x = x / 1000.
        y = y / 1000.
        fwhm = polyval2d(x, y, self.fwhm)
        beta = polyval2d(x, y, self.beta)
        xy = polyval2d(x, y, self.xy)
        yy = polyval2d(x, y, self.yy)
        tstampsz = max(stampsz, self.normalize)
        psf = moffat_psf(fwhm, beta=beta, xy=xy,
                         yy=yy, stampsz=tstampsz, deriv=deriv,
                         shift=(shiftx, shifty))
        if not deriv:
            psf = [psf]
        if self.normalize > 0:
            norms = numpy.sum(central_stamp(psf[0], censize=self.normalize),
                              axis=(-1, -2)).reshape(-1, 1, 1)
        else:
            norms = 1
        psf = [central_stamp(p, censize=stampsz) / norms
               for p in psf]
        if not deriv:
            psf = psf[0]
        return psf


class VariablePixelizedPSF:
    def __init__(self, stamp, normalize=19):
        stampsz = stamp.shape[-1]
        if (stampsz % 2) == 0:
            raise ValueError('problematic shape')
        self.stamp = stamp
        self.normalize = normalize
        self.deriv = numpy.gradient(-self.stamp, axis=(2, 3))
        if normalize > 0:
            cstamp = central_stamp(stamp, normalize)
        else:
            cstamp = stamp
        self.normstamp = numpy.sum(cstamp, axis=(2, 3))
        stampsz = cstamp.shape[-1]
        stampszo2 = stampsz // 2
        xc = numpy.arange(stampsz, dtype='f4')-stampszo2
        xc = xc.reshape(1, 1, -1, 1)
        yc = xc.copy().reshape(1, 1, 1, -1)
        self.xstamp = numpy.sum(xc*cstamp, axis=(2, 3))
        self.ystamp = numpy.sum(yc*cstamp, axis=(2, 3))

    def norm(self, x, y):
        from numpy.polynomial.polynomial import polyval2d
        x, y = (x/1000., y/1000.)
        return polyval2d(x, y, self.normstamp)

    def centroid(self, x, y):
        from numpy.polynomial.polynomial import polyval2d
        x, y = (x/1000., y/1000.)
        if self.normalize < 0:
            norm = 1
        else:
            norm = self.norm(x, y)
        xc = polyval2d(x, y, self.xstamp)
        yc = polyval2d(x, y, self.ystamp)
        return xc/norm, yc/norm

    def render_model(self, x, y, stampsz=59, deriv=False):
        from numpy.polynomial.polynomial import polyval2d
        x = x / 1000.
        y = y / 1000.
        tstamps = polyval2d(x, y, central_stamp(self.stamp, stampsz))
        if len(tstamps.shape) == 3:
            tstamps = tstamps.transpose(2, 0, 1)
        if deriv:
            dpsfdx = polyval2d(x, y, central_stamp(self.deriv[0], stampsz))
            dpsfdy = polyval2d(x, y, central_stamp(self.deriv[1], stampsz))
            if len(tstamps.shape) == 3:
                dpsfdx = dpsfdx.transpose(2, 0, 1)
                dpsfdy = dpsfdy.transpose(2, 0, 1)
            tstamps = (tstamps, dpsfdx, dpsfdy)
        return tstamps

    def serialize(self, stampsz=None):
        stamp = self.stamp
        if stampsz is not None:
            stamp = central_stamp(self.stamp, stampsz)
        dtype = [('offset', '2f4'),
                 ('stamp', stamp.dtype, stamp.shape)]
        extrapar = getattr(self, 'extraparam', None)
        if extrapar is not None:
            dtype += extrapar.dtype.descr
        res = numpy.zeros(1, dtype=dtype)
        res['offset'][0, :] = getattr(self, 'offset', (0, 0))
        res['stamp'][0, ...] = stamp
        if getattr(self, 'extraparam', None) is not None:
            for name in extrapar.dtype.names:
                res[name][0, ...] = extrapar[name]
        return res

    def __call__(self, x, y, stampsz=None, deriv=False):
        if stampsz is None:
            stampsz = self.stamp.shape[-1]
        parshape = numpy.broadcast(x, y).shape
        tparshape = parshape if len(parshape) > 0 else (1,)

        shiftx, shifty = (q - numpy.round(q) for q in (x, y))
        stamps = self.render_model(x, y, stampsz=stampsz, deriv=deriv)
        if deriv:
            stamps, dpsfdx, dpsfdy = stamps
            dpsfdx = dpsfdx.reshape(tparshape+(stampsz, stampsz))
            dpsfdy = dpsfdy.reshape(tparshape+(stampsz, stampsz))

        stamps = stamps.reshape(tparshape+(stampsz, stampsz))
        norm = numpy.atleast_1d(self.norm(x, y))
        shiftx = numpy.atleast_1d(shiftx)
        shifty = numpy.atleast_1d(shifty)

        for i in range(stamps.shape[0]):
            stamps[i, :, :] = shift(stamps[i, :, :], (shiftx[i], shifty[i]))
        stamps /= norm.reshape(-1, 1, 1)
        if tparshape != parshape:
            stamps = stamps.reshape(stamps.shape[1:])

        if deriv:
            for i in range(stamps.shape[0]):
                dpsfdx[i, :, :] = shift(dpsfdx[i, :, :],
                                        (shiftx[i], shifty[i]))
                dpsfdy[i, :, :] = shift(dpsfdy[i, :, :],
                                        (shiftx[i], shifty[i]))
            dpsfdx /= norm.reshape(-1, 1, 1)
            dpsfdy /= norm.reshape(-1, 1, 1)
            if tparshape != parshape:
                dpsfdx = dpsfdx.reshape(stamps.shape[1:])
                dpsfdy = dpsfdy.reshape(stamps.shape[1:])
            stamps = (stamps, dpsfdx, dpsfdy)

        return stamps


class VariableMoffatPixelizedPSF:
    def __init__(self, stamp, fwhm, beta, xy=0., yy=1., normalize=-1):
        self.moffat = VariableMoffatPSF(fwhm, beta, xy=xy, yy=yy, normalize=-1)
        self.resid = VariablePixelizedPSF(stamp, normalize=-1)
        self.normalize = normalize

    def render_model(self, x, y, stampsz=59, deriv=False):
        mof = self.moffat.render_model(x, y, stampsz=stampsz, deriv=deriv)
        res = self.resid.render_model(x, y, stampsz=stampsz, deriv=deriv)

        if not deriv:
            return mof + res
        else:
            return [a+b for (a, b) in zip(mof, res)]

    def __call__(self, x, y, stampsz=None, deriv=False):
        stampsz = (stampsz if stampsz is not None else
                   self.resid.stamp.shape[-1])
        tstampsz = max(stampsz, self.normalize)
        modstamp = self.render_model(x, y, stampsz=tstampsz, deriv=deriv)
        if not deriv:
            modstamp = [modstamp]
        shiftx, shifty = (q - numpy.round(q) for q in (x, y))
        if len(modstamp[0].shape) == 2:
            for modstamp0 in modstamp:
                modstamp0[:, :] = shift(modstamp0[:, :], (shiftx, shifty))
        else:
            for modstamp0 in modstamp:
                for i in range(modstamp0.shape[0]):
                    modstamp0[i, :, :] = shift(modstamp0[i, :, :],
                                               (shiftx[i], shifty[i]))
        if self.normalize > 0:
            norms = numpy.sum(central_stamp(modstamp[0],
                                            censize=self.normalize),
                              axis=(-1, -2))
            norms = numpy.array(norms)[..., None, None]
        else:
            norms = 1 + self.resid.norm(x, y)
        for modstamp0 in modstamp:
            if len(modstamp0.shape) == 2:
                modstamp0 /= norms
            else:
                modstamp0 /= norms.reshape(-1, 1, 1)
        if not deriv:
            modstamp = modstamp[0]
        return modstamp


class GridInterpPSF:
    def __init__(self, stamp, x, y, normalize=19):
        stampsz = stamp.shape[-1]
        if (stampsz % 2) == 0:
            raise ValueError('problematic shape')
        if (stamp.shape[0] != len(x)) or (stamp.shape[1] != len(y)):
            raise ValueError('mismatch between grid coordinates and stamp.')

        self.stamp = stamp
        self.normalize = normalize
        self.x = x
        self.y = y
        self.deriv = numpy.gradient(-self.stamp, axis=(2, 3))
        if normalize > 0:
            cstamp = central_stamp(stamp, normalize)
        else:
            cstamp = stamp
        self.normstamp = numpy.sum(cstamp, axis=(2, 3))
        stampsz = cstamp.shape[-1]
        stampszo2 = stampsz // 2
        xc = numpy.arange(stampsz, dtype='f4')-stampszo2
        xc = xc.reshape(1, 1, -1, 1)
        yc = xc.copy().reshape(1, 1, 1, -1)
        self.xstamp = numpy.sum(xc*cstamp, axis=(2, 3))
        self.ystamp = numpy.sum(yc*cstamp, axis=(2, 3))

    def interpolator(self, stamp, x, y):
        x0 = numpy.atleast_1d(x)
        y0 = numpy.atleast_1d(y)
        ind = [numpy.interp(z, zgrid, numpy.arange(len(zgrid), dtype='f4'),
                            left=0, right=len(zgrid)-1)
               for (z, zgrid) in ((x0, self.x), (y0, self.y))]
        w1 = [numpy.ceil(z) - z for z in ind]
        w2 = [1 - z for z in w1]
        left = [numpy.floor(z).astype('i4') for z in ind]
        right = [numpy.ceil(z).astype('i4') for z in ind]
        ret = numpy.zeros((len(x0),)+stamp.shape[2:], dtype=stamp.dtype)
        for i in range(len(x0)):
            ret[i, ...] = (
                w1[0][i]*w1[1][i]*stamp[left[0][i], left[1][i], ...] +
                w1[0][i]*w2[1][i]*stamp[left[0][i], right[1][i], ...] +
                w2[0][i]*w1[1][i]*stamp[right[0][i], left[1][i], ...] +
                w2[0][i]*w2[1][i]*stamp[right[0][i], right[1][i], ...])
        if x0 is not x:
            ret = ret[0]
        return ret

    def norm(self, x, y):
        return self.interpolator(self.normstamp, x, y)

    def centroid(self, x, y):
        if self.normalize < 0:
            norm = 1
        else:
            norm = self.norm(x, y)
        xc = self.interpolator(self.xstamp, x, y)
        yc = self.interpolator(self.ystamp, x, y)
        return xc/norm, yc/norm

    def render_model(self, x, y, stampsz=59, deriv=False):
        tstamps = self.interpolator(central_stamp(self.stamp, stampsz), x, y)
        if deriv:
            dpsfdx = self.interpolator(central_stamp(self.deriv[0], stampsz),
                                       x, y)
            dpsfdy = self.interpolator(central_stamp(self.deriv[1], stampsz),
                                       x, y)
            tstamps = (tstamps, dpsfdx, dpsfdy)
        return tstamps

    def serialize(self, stampsz=None):
        stamp = self.stamp
        if stampsz is not None:
            stamp = central_stamp(self.stamp, stampsz)
        dtype = [('stamp', stamp.dtype, stamp.shape),
                 ('x', len(self.x), 'f4'), ('y', len(self.y), 'f4')]
        extrapar = getattr(self, 'extraparam', None)
        if extrapar is not None:
            dtype += extrapar.dtype.descr
        res = numpy.zeros(1, dtype=dtype)
        res['stamp'][0, ...] = stamp
        res['x'][0, ...] = self.x
        res['y'][0, ...] = self.y
        if getattr(self, 'extraparam', None) is not None:
            for name in extrapar.dtype.names:
                res[name][0, ...] = extrapar[name]
        return res

    def __call__(self, x, y, stampsz=None, deriv=False):
        if stampsz is None:
            stampsz = self.stamp.shape[-1]
        parshape = numpy.broadcast(x, y).shape
        tparshape = parshape if len(parshape) > 0 else (1,)
        x = numpy.atleast_1d(x)
        y = numpy.atleast_1d(y)

        shiftx, shifty = (q - numpy.round(q) for q in (x, y))
        stamps = self.render_model(x, y, stampsz=stampsz, deriv=deriv)
        if deriv:
            stamps, dpsfdx, dpsfdy = stamps
            dpsfdx = dpsfdx.reshape(tparshape+(stampsz, stampsz))
            dpsfdy = dpsfdy.reshape(tparshape+(stampsz, stampsz))

        stamps = stamps.reshape(tparshape+(stampsz, stampsz))
        norm = numpy.atleast_1d(self.norm(x, y))
        shiftx = numpy.atleast_1d(shiftx)
        shifty = numpy.atleast_1d(shifty)

        for i in range(stamps.shape[0]):
            stamps[i, :, :] = shift(stamps[i, :, :], (shiftx[i], shifty[i]))
        stamps /= norm.reshape(-1, 1, 1)
        if tparshape != parshape:
            stamps = stamps.reshape(stamps.shape[1:])

        if deriv:
            for i in range(stamps.shape[0]):
                dpsfdx[i, :, :] = shift(dpsfdx[i, :, :],
                                        (shiftx[i], shifty[i]))
                dpsfdy[i, :, :] = shift(dpsfdy[i, :, :],
                                        (shiftx[i], shifty[i]))
            dpsfdx /= norm.reshape(-1, 1, 1)
            dpsfdy /= norm.reshape(-1, 1, 1)
            if tparshape != parshape:
                dpsfdx = dpsfdx.reshape(stamps.shape[1:])
                dpsfdy = dpsfdy.reshape(stamps.shape[1:])
            stamps = (stamps, dpsfdx, dpsfdy)
        return stamps


def select_stamps(psfstack, imstack, weightstack, shiftx, shifty):
    if psfstack.shape[0] == 0:
        return numpy.ones(0, dtype='bool')
    tflux = numpy.sum(psfstack, axis=(1, 2))
    timflux = numpy.sum(imstack, axis=(1, 2))
    tmedflux = numpy.median(psfstack, axis=(1, 2))
    npix = psfstack.shape[1]*psfstack.shape[2]
    tfracflux = tflux / numpy.clip(timflux, 100, numpy.inf)
    tfracflux2 = ((tflux-tmedflux*npix) /
                  numpy.clip(timflux, 100, numpy.inf))
    # tfracflux3 = ((tflux - tmedflux*npix)/
    #               numpy.clip(timflux-tmedflux*npix, 100, numpy.inf))
    cx, cy = (imstack.shape[-2] // 2, imstack.shape[-1] // 2)
    cenflux = imstack[:, cx, cy]
    psfqf = (numpy.sum(psfstack*(weightstack > 0), axis=(1, 2)) /
             (tflux + (tflux == 0)))
    okpsf = ((numpy.abs(psfqf - 1) < 0.03) &
             (tfracflux > 0.5) & (tfracflux2 > 0.2) &
             (weightstack[:, cx, cy] > 0) &
             (cenflux*weightstack[:, cx, cy] > 3))
    if numpy.sum(okpsf) > 0:
        shiftxm = numpy.median(shiftx[okpsf])
        shiftym = numpy.median(shifty[okpsf])
        okpsf = (okpsf &
                 (numpy.abs(shiftx-shiftxm) < 1.) &
                 (numpy.abs(shifty-shiftym) < 1.))
    return okpsf


def shift_and_normalize_stamps(psfstack, modstack, weightstack,
                               shiftx, shifty):
    xr = numpy.round(shiftx)
    yr = numpy.round(shifty)
    psfstack = psfstack.copy()
    weightstack = weightstack.copy()
    psfstack = (psfstack -
                numpy.median(psfstack-modstack, axis=(1, 2)).reshape(-1, 1, 1))
    norms = numpy.sum(psfstack, axis=(1, 2))
    psfstack /= norms.reshape(-1, 1, 1)
    weightstack *= norms.reshape(-1, 1, 1)
    for i in range(psfstack.shape[0]):
        psfstack[i, :, :] = shift(psfstack[i, :, :], [-shiftx[i], -shifty[i]])
        if (numpy.abs(xr[i]) > 0) or (numpy.abs(yr[i]) > 0):
            weightstack[i, :, :] = shift(weightstack[i, :, :],
                                         [-xr[i], -yr[i]],
                                         mode='constant', cval=0.)
    return psfstack, weightstack


def fill_param_matrix(param, order):
    ret = numpy.zeros((order+1, order+1)+param.shape[1:], dtype='f4')
    ret[numpy.tril_indices(order+1)] = param
    return ret[::-1, ...]


def extract_params(param, order, pixsz):
    nperpar = (order+1)*(order+2)/2
    if (pixsz**2.+3)*nperpar != len(param):
        raise ValueError('Bad parameter vector size?')
    return [fill_param_matrix(x, order) for x in
            (param[0:nperpar], param[nperpar:nperpar*2],
             param[nperpar*2:nperpar*3],
             param[nperpar*3:nperpar*(3+pixsz**2)].reshape(nperpar, pixsz,
                                                           pixsz))]


def extract_params_moffat(param, order):
    nperpar = (order+1)*(order+2)/2
    if 3*nperpar != len(param):
        raise ValueError('Bad parameter vector size?')
    return [fill_param_matrix(x, order) for x in
            (param[0:nperpar], param[nperpar:nperpar*2],
             param[nperpar*2:nperpar*3])]


def plot_psf_fits(stamp, x, y, model, isig, name=None, save=False):
    from matplotlib import pyplot as p

    datim = numpy.zeros((stamp.shape[1]*10, stamp.shape[1]*10), dtype='f4')
    modim = numpy.zeros((stamp.shape[1]*10, stamp.shape[1]*10), dtype='f4')
    xbd = numpy.linspace(numpy.min(x)-0.01, numpy.max(x)+0.01, 11)
    ybd = numpy.linspace(numpy.min(y)-0.01, numpy.max(y)+0.01, 11)
    medmodel = numpy.median(model, axis=0)
    sz = stamp.shape[-1]
    for i in range(10):
        for j in range(10):
            m = numpy.flatnonzero((x > xbd[i]) & (x <= xbd[i+1]) &
                                  (y > ybd[j]) & (y <= ybd[j+1]))
            if len(m) == 0:
                continue
            ind = m[numpy.argmax(numpy.median(isig[m, :, :], axis=(1, 2)))]
            datim0 = stamp[ind, :, :]
            modim0 = model[ind, :, :]
            datim[i*sz:(i+1)*sz, j*sz:(j+1)*sz] = datim0-medmodel
            modim[i*sz:(i+1)*sz, j*sz:(j+1)*sz] = modim0-medmodel
    p.figure(figsize=(24, 8), dpi=150)
    p.subplot(1, 3, 1)
    p.imshow(datim, aspect='equal', vmin=-0.005, vmax=0.005, cmap='binary')
    p.title('Stamps')
    p.subplot(1, 3, 2)
    p.imshow(modim, aspect='equal', vmin=-0.005, vmax=0.005, cmap='binary')
    p.title('Model')
    p.subplot(1, 3, 3)
    p.imshow(datim-modim, aspect='equal', vmin=-0.001, vmax=0.001,
             cmap='binary')
    p.title('Residuals')
    if save:
        import matplotlib
        matplotlib.use('Agg')
        p.style.use('dark_background')
        p.savefig('psf_'+name[1]+'_'+str(name[0])+'.png', dpi=150,
                  bbox_inches='tight', pad_inches=0.1)


def plot_psf_fits_brightness(stamp, x, y, model, isig):
    from matplotlib import pyplot as p
    import util_efs
    nx, ny = 10, 10
    datim = numpy.zeros((stamp.shape[1]*nx, stamp.shape[1]*ny), dtype='f4')
    modim = numpy.zeros((stamp.shape[1]*nx, stamp.shape[1]*ny), dtype='f4')
    medmodel = numpy.median(model, axis=0)
    s = numpy.argsort(-numpy.median(isig, axis=(1, 2)))
    sz = stamp.shape[-1]
    for i in range(nx):
        for j in range(ny):
            if i*ny+j >= len(s):
                continue
            ind = s[i*ny+j]
            datim0 = stamp[ind, :, :]
            modim0 = model[ind, :, :]
            datim[i*sz:(i+1)*sz, j*sz:(j+1)*sz] = datim0-medmodel
            modim[i*sz:(i+1)*sz, j*sz:(j+1)*sz] = modim0-medmodel
    p.figure('psfs')
    p.subplot(1, 3, 1)
    util_efs.imshow(datim, aspect='equal', vmin=-0.005, vmax=0.005)
    p.title('Stamps')
    p.subplot(1, 3, 2)
    util_efs.imshow(modim, aspect='equal', vmin=-0.005, vmax=0.005)
    p.title('Model')
    p.subplot(1, 3, 3)
    util_efs.imshow(datim-modim, aspect='equal', vmin=-0.001, vmax=0.001)
    p.title('Residuals')
    p.draw()


def damper(chi, damp):
    return 2*damp*numpy.sign(chi)*(numpy.sqrt(1+numpy.abs(chi)/damp)-1)


def fit_variable_moffat_psf(x, y, xcen, ycen, stamp, imstamp, modstamp,
                            isig, order=1, pixsz=9, nkeep=200, plot=False, name=None):
    # clean and shift the PSFs first.
    shiftx = xcen + x - numpy.round(x)
    shifty = ycen + y - numpy.round(y)
    okpsf = select_stamps(stamp, imstamp, isig, shiftx, shifty)
    x, y, xcen, ycen = (q[okpsf] for q in (x, y, xcen, ycen))
    stamp, modstamp, isig, imstamp, shiftx, shifty = (
        q[okpsf] for q in (stamp, modstamp, isig, imstamp, shiftx, shifty))
    if len(x) > nkeep:
        fluxes = numpy.sum(stamp, axis=(1, 2))
        s = numpy.argsort(-fluxes)
        okpsf = (fluxes >= fluxes[s][nkeep-1])
        x, y, xcen, ycen = (q[okpsf] for q in (x, y, xcen, ycen))
        stamp, modstamp, isig, imstamp, shiftx, shifty = (
            q[okpsf] for q in (stamp, modstamp, isig, imstamp, shiftx, shifty))
    stamp, isig = shift_and_normalize_stamps(stamp, modstamp, isig,
                                             shiftx, shifty)
    isig = numpy.clip(isig, 0., 1./(0.1*0.001))
    isig_nocen = isig.copy()
    if stamp.shape[0] > 50:
        central_stamp(isig_nocen, censize=pixsz)[:, :, :] = 0.

    def make_full_psf_model(param, order, pixsz):
        fwhm, xy, yy, resid = extract_params(param, order, pixsz)
        return VariableMoffatPixelizedPSF(resid, fwhm, 3., xy=xy, yy=yy)

    def make_moffat_psf_model(param, order):
        fwhm, xy, yy = extract_params_moffat(param, order)
        return VariableMoffatPSF(fwhm, 3., xy=xy, yy=yy)

    def chimoff(param, isig):
        norm = param[-1]
        psf = make_moffat_psf_model(param[:-1], order)
        tresid = (stamp -
                  norm*psf.render_model(x, y, stampsz=stamp.shape[-1]))
        tchi = damper(tresid*isig, 3.).reshape(-1).astype('f4')
        return tchi

    def chipix(param, resid, isig):
        from numpy.polynomial.polynomial import polyval2d
        mat = fill_param_matrix(param, order)
        tchi = (resid - polyval2d(x/1000., y/1000., mat))*isig
        return damper(tchi, 3.).reshape(-1)

    nperpar = (order+1)*(order+2)/2
    guess = numpy.zeros(3*nperpar+1, dtype='f4')
    constanttermindex = nperpar - order - 1
    guess[0+constanttermindex] = 4.  # 1" PSF
    # guess[nperpar+constanttermindex] = 3.  # beta
    guess[nperpar*2+constanttermindex] = 1.  # yy
    guess[-1] = 1.  # overall normalization
    # all others can be zero.
    from scipy import optimize
    resmoff = optimize.leastsq(chimoff, guess, args=(isig_nocen,),
                               full_output=True)
    residfitdict = {}
    residguess = numpy.zeros(nperpar, dtype='f4')
    moffpsf = make_moffat_psf_model(resmoff[0][:-1], order)
    resid = (stamp - resmoff[0][-1] *
             moffpsf.render_model(x, y, stampsz=stamp.shape[-1])).astype('f4')
    resid_cen = central_stamp(resid, censize=pixsz)
    isig_cen = central_stamp(isig, censize=pixsz)

    for i in range(pixsz):
        for j in range(pixsz):
            args = (resid_cen[:, i, j], isig_cen[:, i, j])
            residfitdict[i, j] = optimize.leastsq(chipix, residguess,
                                                  args=args, full_output=True)

    fullparam = numpy.zeros((3+pixsz**2)*nperpar+1, dtype='f4')
    fullparam[0:3*nperpar] = resmoff[0][0:3*nperpar]
    fullparam[-1] = resmoff[0][-1]
    resparam = numpy.array([[residfitdict[i, j][0]/fullparam[-1]
                             for j in range(pixsz)]
                            for i in range(pixsz)])
    resparam = resparam.transpose(2, 0, 1)
    fullparam[3*nperpar:(3+pixsz**2)*nperpar] = resparam.reshape(-1)

    psf = make_full_psf_model(fullparam[:-1], order, pixsz)
    if plot != 0:
        norm = fullparam[-1]
        modstamps = norm*psf.render_model(x, y, stampsz=stamp.shape[-1])
        if plot == 1:
            plot_psf_fits(stamp, x, y, modstamps, isig, name=name)
        else:
            plot_psf_fits(stamp, x, y, modstamps, isig, name=name, save=True)
    return psf


def fit_moffat(stamp, isig=None):
    if isig is None:
        isig = numpy.ones_like(stamp, dtype='f4')

    def chimoff(param):
        model = param[0]*moffat_psf(param[1], beta=param[2], xy=param[3],
                                    yy=param[4], stampsz=stamp.shape[0],
                                    deriv=False)
        chi = (stamp-model)*isig
        return damper(chi, 5).reshape(-1).astype('f4')

    from scipy import optimize
    guess = numpy.array([1., 4., 3., 0., 1.]).astype('f4')
    res = optimize.leastsq(chimoff, guess, full_output=True, epsfcn=1e-2)
    return res


def sum_prof(param, stampsz=59, prof='moffat'):
    res = numpy.zeros((stampsz, stampsz), dtype='f4')
    npar = 3 if prof == 'moffat' else 2
    ncomp = len(param) / npar
    for i in range(ncomp):
        if prof == 'moffat':
            tres = moffat_psf(param[i*npar+1], beta=param[i*npar+2], xy=0.,
                              yy=1, stampsz=stampsz,
                              deriv=False)
        elif prof == 'gaussian':
            tres = gaussian_psf(param[i*npar+1], stampsz=stampsz,
                                deriv=False)
        res += tres*param[i*npar]
    return res*param[0]


def fit_sum_prof(stamp, ncomp=3, isig=None, prof='moffat'):
    if isig is None:
        isig = numpy.ones_like(stamp, dtype='f4')

    def chiprof(param):
        chi = (stamp-sum_prof(param, stampsz=stamp.shape[-1], prof=prof))*isig
        return damper(chi, 5).reshape(-1).astype('f4')

    guessnorm = numpy.ones(ncomp)/1.0/ncomp
    guessfwhm = 4*numpy.exp(numpy.linspace(0, numpy.log(stamp.shape[-1]/10),
                                           ncomp))
    guessbeta = 3.5-1*numpy.linspace(0, 1, ncomp)
    guess = []
    if prof == 'moffat':
        for n, b, f in zip(guessnorm, guessfwhm, guessbeta):
            guess += [n, b, f]
    else:
        for n, f in zip(guessnorm, guessfwhm):
            guess += [n, f]
    from scipy import optimize
    guess = numpy.array(guess).astype('f4')
    res = optimize.leastsq(chiprof, guess, full_output=True)
    return res


def gaussian(major, minor, rotation, stampsz):
    sigmafac = 1 / numpy.sqrt(8*numpy.log(2))
    major = major * sigmafac
    minor = minor * sigmafac
    stampszo2 = stampsz // 2
    dx = numpy.arange(stampsz).reshape(1, -1, 1)-stampszo2
    dy = dx.copy().reshape(1, 1, -1)
    major = numpy.abs(major).reshape(-1, 1, 1)
    minor = numpy.abs(minor).reshape(-1, 1, 1)
    rotation = rotation.reshape(-1, 1, 1)
    r2 = ((dx*numpy.cos(rotation)-dy*numpy.sin(rotation))**2/major**2 +
          (dx*numpy.sin(rotation)+dy*numpy.cos(rotation))**2/minor**2)
    return 1./(2.*numpy.pi*major*minor)*numpy.exp(-0.5*r2)


def fit_gaussian(stamp, isig=None):
    if isig is None:
        isig = numpy.ones_like(stamp, dtype='f4')

    def chigauss(param):
        model = param[0]*gaussian(numpy.atleast_1d(param[1]),
                                  numpy.atleast_1d(param[2]),
                                  numpy.atleast_1d(param[3]),
                                  stampsz=stamp.shape[-1])
        return (stamp-model).reshape(-1).astype('f4')

    from scipy import optimize
    guess = numpy.array([1., 4., 4., 0.]).astype('f4')
    res = optimize.leastsq(chigauss, guess, full_output=True)
    return res


def chipix(param, resid, isig, x, y, order):
    from numpy.polynomial.polynomial import polyval2d
    mat = fill_param_matrix(param, order)
    tchi = (resid - polyval2d(x/1000., y/1000., mat))*isig
    return damper(tchi, 3.).reshape(-1)


def chipixlin(param, resid, isig, x, y, order):
    if order == 0:
        tchi = (resid - param[0])*isig
    else:
        tchi = (resid - param[1] - param[0]*x/1000. - param[2]*y/1000.)*isig
    return damper(tchi, 3.).reshape(-1)


def modelstampcorn(param, staticstamp, stampsz=None):
    from scipy.signal import fftconvolve
    stampsz = staticstamp.shape[-1] if stampsz is None else stampsz
    if len(param) > 4:
        tx = numpy.array([0, 1000, 0])
        ty = numpy.array([0, 0, 1000])
        fwhm = param[0]+tx/1000.*param[1]+ty/1000.*param[2]
        yy = param[3]+tx/1000.*param[4]+ty/1000.*param[5]
        xy = param[6]+tx/1000.*param[7]+ty/1000.*param[8]
        norm = param[9]
    else:
        fwhm = param[0]*numpy.ones(3, dtype='f4')
        yy = param[1]
        xy = param[2]
        norm = param[3]
    moffats = moffat_psf(fwhm, beta=3., xy=xy, yy=yy,
                         stampsz=stampsz+6, deriv=None)
    tstaticstamp = central_stamp(staticstamp, stampsz+6).copy()
    modcorn = fftconvolve(moffats, tstaticstamp[None, :, :], mode='same')
    # the full convolution is nice here, but we could probably replace with
    # minimal loss of generality with something like the sum of
    # the Moffat and an image convolved with only the center part of the
    # PSF.
    modcorn = central_stamp(modcorn, censize=stampsz).copy()
    return modcorn * norm


def modelstampcorn2(param, staticstamp, stampsz=None):
    from scipy.signal import fftconvolve
    stampsz = staticstamp.shape[-1] if stampsz is None else stampsz
    if len(param) > 5:
        tx = numpy.array([0, 1000, 0])
        ty = numpy.array([0, 0, 1000])
        fwhm = param[0]+tx/1000.*param[1]+ty/1000.*param[2]
        yy = param[3]+tx/1000.*param[4]+ty/1000.*param[5]
        xy = param[6]+tx/1000.*param[7]+ty/1000.*param[8]
        beta = param[9]+tx/1000.*param[10]+ty/1000.*param[11]
        norm = param[12]
    else:
        fwhm = param[0]*numpy.ones(3, dtype='f4')
        yy = param[1]
        xy = param[2]
        beta = param[3]
        norm = param[4]
    moffats = moffat_psf(fwhm, beta=beta, xy=xy, yy=yy,
                         stampsz=stampsz+6, deriv=None)
    tstaticstamp = central_stamp(staticstamp, stampsz+6).copy()
    modcorn = fftconvolve(moffats, tstaticstamp[None, :, :], mode='same')
    # the full convolution is nice here, but we could probably replace with
    # minimal loss of generality with something like the sum of
    # the Moffat and an image convolved with only the center part of the
    # PSF.
    modcorn = central_stamp(modcorn, censize=stampsz).copy()
    return modcorn * norm


def stamp2model(corn, normalize=-1):
    stamppar = numpy.zeros((2, 2, corn.shape[-1], corn.shape[-1]),
                           dtype='f4')
    stamppar[0, 0, :, :] = corn[0]
    stamppar[1, 0, :, :] = (corn[1]-corn[0])
    stamppar[0, 1, :, :] = (corn[2]-corn[0])
    return VariablePixelizedPSF(stamppar, normalize=normalize)


def fit_linear_static_wing(x, y, xcen, ycen, stamp, imstamp, modstamp,
                           isig, pixsz=9, nkeep=200, plot=False,
                           filter='g', name=None):
    # clean and shift the PSFs first.
    shiftx = xcen + x - numpy.round(x)
    shifty = ycen + y - numpy.round(y)
    okpsf = select_stamps(stamp, imstamp, isig, shiftx, shifty)
    if numpy.sum(okpsf) == 0:
        return None
    x, y, xcen, ycen = (q[okpsf] for q in (x, y, xcen, ycen))
    stamp, modstamp, isig, imstamp, shiftx, shifty = (
        q[okpsf] for q in (stamp, modstamp, isig, imstamp, shiftx, shifty))
    if len(x) > nkeep:
        fluxes = numpy.sum(stamp, axis=(1, 2))
        s = numpy.argsort(-fluxes)
        okpsf = (fluxes >= fluxes[s][nkeep-1])
        x, y, xcen, ycen = (q[okpsf] for q in (x, y, xcen, ycen))
        stamp, modstamp, isig, imstamp, shiftx, shifty = (
            q[okpsf] for q in (stamp, modstamp, isig, imstamp, shiftx, shifty))
    stamp, isig = shift_and_normalize_stamps(stamp, modstamp, isig,
                                             shiftx, shifty)
    maxisig = 1./(0.1*0.001)
    isig = numpy.clip(isig, 0., maxisig)

    import os
    from astropy.io import fits
    fname = os.path.join(os.environ['DECAM_DIR'], 'data', 'psfs',
                         'psf_%s_deconv_mod.fits.gz' % filter)

    staticstamp = fits.getdata(fname).T.copy()
    outstampsz = staticstamp.shape[-1]
    normalizesz = 59
    staticstamp /= numpy.sum(central_stamp(staticstamp, normalizesz))

    def modelconv(param, stampsz=None):
        if stampsz is None:
            stampsz = stamp.shape[-1]
        model = stamp2model(modelstampcorn(param, staticstamp,
                                           stampsz=stamp.shape[-1]))
        return model.render_model(x, y, stampsz=stampsz, deriv=False)

    def chiconv(param):
        tresid = stamp - modelconv(param)
        chi = damper(tresid*isig, 3).reshape(-1).astype('f4')
        return chi

    stampszo2 = isig.shape[-1] // 2
    nbright = numpy.sum(isig[:, stampszo2, stampszo2] >= min(1000, maxisig))
    if nbright < 10:
        order = 0
    else:
        order = 1

    from scipy import optimize
    if order == 1:
        guess = numpy.array([2., 0., 0.,
                             1., 0., 0.,
                             0., 0., 0.,
                             # 3., 0., 0.,
                             1.]).astype('f4')
    else:
        guess = numpy.array([2., 1., 0., 1.]).astype('f4')

    res = optimize.leastsq(chiconv, guess, full_output=True)

    resid = (stamp - modelconv(res[0])).astype('f4')
    resid_cen = central_stamp(resid, censize=pixsz)
    isig_cen = central_stamp(isig, censize=pixsz)
    residfitdict = {}
    nperpar = (order+1)*(order+2)/2
    residguess = numpy.zeros(int(nperpar), dtype='f4')

    for i in range(pixsz):
        for j in range(pixsz):
            args = (resid_cen[:, i, j].copy(), isig_cen[:, i, j].copy(), x, y,
                    order)
            residfitdict[i, j] = optimize.leastsq(chipixlin, residguess,
                                                  args=args, full_output=True)

    resparam = numpy.array([[residfitdict[i, j][0]
                             for j in range(pixsz)]
                            for i in range(pixsz)])
    resparam = resparam.transpose(2, 0, 1)
    modresid = VariablePixelizedPSF(fill_param_matrix(resparam, order),
                                    normalize=-1)
    modwing = stamp2model(modelstampcorn(res[0], staticstamp,
                                         stampsz=outstampsz))
    xx = numpy.array([0, 1000, 0])
    yy = numpy.array([0, 0, 1000])
    cornwing = modwing.render_model(xx, yy, deriv=False,
                                    stampsz=outstampsz)
    cornresid = modresid.render_model(xx, yy, deriv=False,
                                      stampsz=outstampsz)
    modtotal = stamp2model(cornwing+cornresid, normalize=normalizesz)
    nlinperpar = 3
    extraparam = numpy.zeros(
        1, dtype=[('convparam', 'f4', 4*nlinperpar+1),
                  ('resparam', 'f4', (nlinperpar, pixsz, pixsz))])
    extraparam['convparam'][0, 0:len(res[0])] = res[0]
    extraparam['resparam'][0, 0:resparam.shape[0], :, :] = resparam
    modtotal.extraparam = extraparam

    if plot != 0:
        modstamps = modtotal.render_model(x, y, deriv=False,
                                          stampsz=stamp.shape[-1])
        if plot == 1:
            plot_psf_fits(stamp, x, y, modstamps, isig, name=name)
        else:
            plot_psf_fits(stamp, x, y, modstamps, isig, name=name, save=True)
    return modtotal


def linear_static_wing_from_record(record, filter='g'):
    import os
    from astropy.io import fits
    fname = os.path.join(os.environ['DECAM_DIR'], 'data', 'psfs',
                         'psf_%s_deconv_mod.fits.gz' % filter)
    staticstamp = fits.getdata(fname).T.copy()
    outstampsz = staticstamp.shape[-1]
    normalizesz = 59
    staticstamp /= numpy.sum(central_stamp(staticstamp, normalizesz))
    order = 1 if numpy.any(record['resparam'][1:, ...]) else 0
    nperpar = int((order+1)*(order+2)/2)
    modresid = VariablePixelizedPSF(
        fill_param_matrix(record['resparam'][:nperpar], order), normalize=-1)
    modwing = stamp2model(modelstampcorn(record['convparam'], staticstamp,
                                         stampsz=outstampsz))
    xx = numpy.array([0, 1000, 0])
    yy = numpy.array([0, 0, 1000])
    cornwing = modwing.render_model(xx, yy, deriv=False,
                                    stampsz=outstampsz)
    cornresid = modresid.render_model(xx, yy, deriv=False,
                                      stampsz=outstampsz)
    modtotal = stamp2model(cornwing+cornresid, normalize=normalizesz)
    modtotal.offset = record['offset']
    return modtotal


def wise_psf_fit(x, y, xcen, ycen, stamp, imstamp, modstamp,
                 isig, pixsz=9, nkeep=200, plot=False,
                 psfstamp=None, grid=False, name=None):
    if psfstamp is None:
        raise ValueError('psfstamp must be set')
    # clean and shift the PSFs first.
    shiftx = xcen + x - numpy.round(x)
    shifty = ycen + y - numpy.round(y)

    okpsf = select_stamps(stamp, imstamp, isig, shiftx, shifty)
    if numpy.sum(okpsf) == 0:
        print('Giving up PSF fit...')
        return None
    x, y, xcen, ycen = (q[okpsf] for q in (x, y, xcen, ycen))
    stamp, modstamp, isig, imstamp, shiftx, shifty = (
        q[okpsf] for q in (stamp, modstamp, isig, imstamp, shiftx, shifty))
    if len(x) < 200:
        # really should never happen for WISE images, unless we're, say,
        # right in the Galactic center and things are horrible.
        print('Only %d PSF stars (of %d total), giving up PSF fit...' %
              (len(x), len(okpsf)))
        if not grid:
            return SimplePSF(psfstamp)
        else:
            return GridInterpPSF(*psfstamp)
    if len(x) > nkeep:
        fluxes = numpy.sum(stamp, axis=(1, 2))
        s = numpy.argsort(-fluxes)
        okpsf = (fluxes >= fluxes[s][nkeep-1])
        x, y, xcen, ycen = (q[okpsf] for q in (x, y, xcen, ycen))
        stamp, modstamp, isig, imstamp, shiftx, shifty = (
            q[okpsf] for q in (stamp, modstamp, isig, imstamp, shiftx, shifty))
    stamp, isig = shift_and_normalize_stamps(stamp, modstamp, isig,
                                             shiftx, shifty)
    maxisig = 1./(0.1*0.001)
    isig0 = isig.copy()
    isig = numpy.clip(isig, 0., maxisig)

    stampsz = isig.shape[-1]
    if not grid:
        normstamp = numpy.sum(central_stamp(psfstamp, censize=stampsz))
        psfstamp /= normstamp
        npsfstamp = psfstamp
    else:
        normstamp = numpy.sum(central_stamp(psfstamp[0], censize=stampsz),
                              axis=(2, 3))[:, :, None, None]
        psfstamp[0][...] = psfstamp[0] / normstamp
        npsfstamp = psfstamp[0]
        npsfstamp = numpy.mean(npsfstamp, axis=(0, 1))
        npsfstamp /= numpy.sum(central_stamp(npsfstamp, censize=stampsz))

    resid = (stamp - central_stamp(npsfstamp, censize=stampsz))
    resid = resid.astype('f4')
    resid_cen = central_stamp(resid, censize=pixsz)
    residmed = numpy.median(resid_cen, axis=0)
    if not grid:
        newstamp = psfstamp.copy()
        central_stamp(newstamp, censize=pixsz)[:, :] += residmed
    else:
        newstamp = psfstamp[0].copy()
        central_stamp(newstamp, censize=pixsz)[:, :, :, :] += (
            residmed[None, None, :, :])

    if plot:
        if not grid:
            modstamp = central_stamp(newstamp, censize=stampsz)
        else:
            # HACK; not clear which model to use... should use the right
            # one for each, but I don't really want to go there...
            modstamp = central_stamp(newstamp[0, 0, :, :], censize=stampsz)
        modstamp = modstamp[None, ...]*numpy.ones((stamp.shape[0], 1, 1))
        plot_psf_fits_brightness(stamp, x, y, modstamp, isig0)

    if not grid:
        return SimplePSF(newstamp)
    else:
        return GridInterpPSF(newstamp, psfstamp[1], psfstamp[2])
