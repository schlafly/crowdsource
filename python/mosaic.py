"""Routines for cutting a large CCD into regions for fitting.

This module wraps crowdsource.py to allow large CCDs to be individually fit
while maintaining consistent source on the boundaries between CCDs.
"""

import sys
import time
import pdb
import numpy
import crowdsource


def in_bounds(x, y, xbound, ybound):
    return ((x > xbound[0]) & (x <= xbound[1]) &
            (y > ybound[0]) & (y <= ybound[1]))


def fit_sections(im, psf, nx, ny, overlap=50, weight=None, **kw):
    bdx = numpy.round(numpy.linspace(0, im.shape[0], nx+1)).astype('i4')
    bdlx = numpy.clip(bdx - overlap, 0, im.shape[0])
    bdrx = numpy.clip(bdx + overlap, 0, im.shape[0])
    bdy = numpy.round(numpy.linspace(0, im.shape[1], ny+1)).astype('i4')
    bdly = numpy.clip(bdy - overlap, 0, im.shape[0])
    bdry = numpy.clip(bdy + overlap, 0, im.shape[0])
    modelim = numpy.zeros_like(im)
    skyim = numpy.zeros_like(im)
    stars = numpy.zeros(0, dtype=[('x', 'f4'), ('y', 'f4'), ('flux', 'f4'),
                                  ('primary', 'i4'), ('psf', 'i4')])
    t0 = time.time()
    if kw.get('verbose', False):
        print('Starting new CCD at %s' % time.ctime())
        sys.stdout.flush()
    psfs = []
    for i in range(nx):
        for j in range(ny):
            sall = numpy.s_[bdlx[i]:bdrx[i+1], bdly[j]:bdry[j+1]]
            spri = numpy.s_[bdx[i]:bdx[i+1], bdy[j]:bdy[j+1]]
            dx, dy = (bdrx[i+1]-bdlx[i], bdry[j+1]-bdly[j])
            sfit = numpy.s_[bdx[i]-bdlx[i]:dx+bdx[i+1]-bdrx[i+1],
                            bdy[j]-bdly[j]:dy+bdy[j+1]-bdry[j+1]]
            mfixed = in_bounds(stars['x'], stars['y'],
                               [bdlx[i]-0.5, bdrx[i+1]-0.5],
                               [bdly[j]-0.5, bdry[j+1]-0.5])
            mfixed &= ~in_bounds(stars['x'], stars['y'],
                                 [bdx[i]-0.5, bdx[i+1]-0.5],
                                 [bdy[j]-0.5, bdy[j+1]-0.5])
            fixedstars = {f: stars[f][mfixed] for f in stars.dtype.names}
            fixedstars['x'] -= bdlx[i]
            fixedstars['y'] -= bdly[j]
            fixedstars['stamp'] = psfs
            if (i == 0) and (j == 0):
                tpsf = psf
            elif j != 0:
                tpsf = psfs[-1]
            else:
                tpsf = psfs[-ny]
            res0 = crowdsource.fit_im(im[sall].copy(), tpsf,
                                      weight=weight[sall].copy(),
                                      fixedstars=fixedstars, **kw)
            x0, y0, flux0, skypar0, model0, sky0, psf0 = res0
            x0 += bdlx[i]
            y0 += bdly[j]
            primary0 = in_bounds(x0, y0,
                                 [bdx[i]-0.5, bdx[i+1]-0.5],
                                 [bdy[j]-0.5, bdy[j+1]-0.5])
            # pdb.set_trace()
            newstars = numpy.array(zip(
                x0, y0, flux0, primary0,
                len(psfs)*numpy.ones(len(x0), dtype='i4')),
                dtype=stars.dtype)
            stars = numpy.append(stars, newstars)
            psfs.append(psf0)
            modelim[spri] = model0[sfit]
            skyim[spri] = model0[sfit]
            if kw.get('verbose', False):
                t1 = time.time()
                print('Fit tile (%d, %d) of (%d, %d); %d sec elapsed' %
                      (i+1, j+1, nx, ny, t1-t0))
                t0 = t1            # import csplot
                sys.stdout.flush()
            # pdb.set_trace()
    return stars, modelim, skyim, psfs
