import os
import pdb
import numpy
from skimage import restoration
from astropy.io import fits
import crowdsource.psf as psf

import os
if 'DECAM_DIR' not in os.environ:
    decam_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"decam_dir")
    os.environ['DECAM_DIR'] = decam_dir

filt = 'ugrizY'
deconv = {'u': 0.8, 'g': 0.75, 'r': 0.7, 'i': 0.6, 'z': 0.65, 'Y': 0.65}


def make_new_psfs(write=False, **kw):
    path = os.path.join(os.environ['DECAM_DIR'], 'data', 'psfs')
    res = {}
    for f in filt:
        tpsf = fits.getdata(os.path.join(path, 'psf_%s.fits.gz' % f))
        tpsf = psf.center_psf(tpsf)
        fitres = psf.fit_moffat(psf.central_stamp(tpsf, censize=19).copy())
        fit = fitres[0]
        deconvfac = 0.7
        kernel = psf.moffat_psf(fit[1]*deconvfac, yy=fit[4],
                                beta=fit[2], xy=fit[3],
                                stampsz=69, deriv=False)
        psfde = restoration.richardson_lucy(tpsf, kernel, 20)
        psfde = psf.center_psf(psfde)
        res[f] = psfde
    if write:
        for f in filt:
            fits.writeto(os.path.join(path, 'psf_%s_deconv.fits.gz' % f),
                         res[f], **kw)
    return res


def make_new_model_psfs(write=False, **kw):
    path = os.path.join(os.environ['DECAM_DIR'], 'data', 'psfs')
    res = {}
    for f in filt:
        tpsfd = fits.getdata(os.path.join(path, 'psf_%s_deconv.fits.gz' % f))
        tpsfdm = fit_outer_psf(tpsfd)
        tpsfdb = blend_psf(tpsfd, tpsfdm[2], 6, 10)
        res[f] = tpsfdb
    if write:
        for f in filt:
            fits.writeto(os.path.join(path, 'psf_%s_deconv_mod.fits.gz' % f),
                         res[f], **kw)
    return res


def medprofile(psf, binsz=3):
    stampsz = psf.shape[-1]
    stampszo2 = stampsz // 2
    xx = numpy.arange(stampsz, dtype='f4').reshape(1, -1)-stampszo2
    yy = xx.copy().reshape(-1, 1)
    rr = numpy.sqrt(xx**2+yy**2)
    binsz = 3
    return meanbin(rr, psf, binsz)


def meanbin(rr, pts, binsz=3):
    rbins = numpy.arange(0, numpy.ceil(numpy.max(rr)), binsz)
    medval = rbins * 0.0
    for i in range(len(rbins)):
        medval[i] = numpy.mean(pts[(rr >= rbins[i]) & (rr < rbins[i]+1)])
    return rbins + 0.5*binsz, medval


def make_approximate_spikes(fwhm1, fwhm2, stampsz, openingangle=4.0, vhfac=0.2,
                            dfac=0.0015, vfac=6e-4, reflfwhm=25.0):
    gpsf1 = psf.gaussian_psf(fwhm1, stampsz=29, deriv=False)
    gpsf2 = psf.gaussian_psf(fwhm2, stampsz=29, deriv=False)
    stampszo2 = stampsz // 2
    xx = numpy.arange(stampsz, dtype='f4').reshape(1, -1)-stampszo2
    yy = xx.copy().reshape(-1, 1)
    rr = numpy.sqrt(xx**2+yy**2)
    theta = numpy.arctan2(yy, xx)
    mv = numpy.abs((((theta % (numpy.pi)) + numpy.pi/8)
                    % (numpy.pi)) - numpy.pi/8) < openingangle*numpy.pi/180
    vspike = mv * 1.0 / numpy.sum(mv, axis=0).reshape(1, -1)
    vspike = vspike * (1+(rr/(1e-7+numpy.abs(reflfwhm))))**(-2.4)*vfac
    hspike = vspike.T * vhfac
    md = numpy.abs(((((theta+numpy.pi/4) % (numpy.pi/2)) + numpy.pi/8)
                    % (numpy.pi/2)) - numpy.pi/8) < 1.0*numpy.pi/180
    md = md * 1.0 / numpy.clip(numpy.sum(md, axis=0).reshape(1, -1), 1,
                               numpy.inf)
    md = (md + md.T)/2.
    dspike = md * (1+rr)**-1.75 * dfac
    imd = dspike+dspike.T
    imd[(xx == yy) & (xx < 0)] *= 3
    from scipy.signal import fftconvolve
    imd = fftconvolve(imd, gpsf1, mode='same')
    imh = fftconvolve(hspike+vspike, gpsf2, mode='same')
    return imd+imh


def fit_outer_psf(stamp):
    stampsz = stamp.shape[-1]
    stampszo2 = stampsz // 2
    xx = numpy.arange(stampsz, dtype='f4').reshape(1, -1)-stampszo2
    yy = xx.copy().reshape(-1, 1)
    rr = numpy.sqrt(xx**2+yy**2)
    openingangle = numpy.pi/180.*5.
    theta = numpy.arctan2(yy, xx)
    mspike1 = numpy.abs((((theta % (numpy.pi/2.)) + numpy.pi/8) %
                        (numpy.pi/2.)) - numpy.pi/8) < openingangle*4
    mspike2 = numpy.abs((((theta % (numpy.pi/4.)) + numpy.pi/8) %
                        (numpy.pi/4.)) - numpy.pi/8) < openingangle
    mspike = mspike1 | mspike2

    isig = (rr > 3)*5e7*(~mspike)
    dmres = psf.fit_sum_prof(psf.central_stamp(stamp, 149), ncomp=6,
                             isig=psf.central_stamp(isig, 149),
                             prof='moffat')
    dmstamp = psf.sum_prof(dmres[0], stampsz=stamp.shape[-1], prof='moffat')

    # from matplotlib.pyplot import figure, clf, draw, plot ; import util_efs
    # figure(1) ; clf() ; util_efs.imshow(numpy.arcsinh((stamp-dmstamp)*1e6)*(~mspike)*(rr < 150), aspect='equal', vmin=-1, vmax=1) ; draw()
    # pdb.set_trace()

    def model(param):
        tmod = make_approximate_spikes(param[1], param[2], stampsz,
                                       openingangle=param[3],
                                       vhfac=param[4], dfac=param[5],
                                       vfac=param[6], reflfwhm=param[7])
        return param[0]*tmod

    resid = stamp - dmstamp

    def chispike(param):
        chi = (resid-model(param))*1e5*(rr > 20)
        return damper(chi, 5).reshape(-1).astype('f4')

    from scipy import optimize
    guess = numpy.array([1., 2., 4., 4., 0.2, 0.0015, 6e-4, 25.0]).astype('f4')
    res = optimize.leastsq(chispike, guess, full_output=True)
    modim = dmstamp + model(res[0])
    return dmres, res, modim, dmstamp


def damper(chi, damp):
    return 2*damp*numpy.sign(chi)*(numpy.sqrt(1+numpy.abs(chi)/damp)-1)


def blend_psf(dstamp, mstamp, innerrad, outerrad):
    stampsz = dstamp.shape[-1]
    stampszo2 = stampsz // 2
    xx = numpy.arange(stampsz, dtype='f4').reshape(1, -1)-stampszo2
    yy = xx.copy().reshape(-1, 1)
    rr = numpy.sqrt(xx**2+yy**2)
    weight = numpy.clip(1 - (rr - innerrad) / float(outerrad - innerrad), 0, 1)
    dstamp = (dstamp + dstamp[::-1, :] +
              dstamp[:, ::-1] + dstamp[::-1, ::-1])/4.
    blended = dstamp*weight + mstamp*(1-weight)
    blended = psf.center_psf(blended)
    return blended
