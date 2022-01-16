"""Routines for cutting a large CCD into regions for fitting.

This module wraps crowdsource.py to allow large CCDs to be individually fit
while maintaining consistent source on the boundaries between CCDs.
"""

import sys
import time
import pdb
import numpy
from crowdsource import crowdsource_base


def in_bounds(x, y, xbound, ybound):
    return ((x > xbound[0]) & (x <= xbound[1]) &
            (y > ybound[0]) & (y <= ybound[1]))


def fit_sections(im, psf, nx, ny, overlap=50, weight=None, dq=None,
                 blist=None, **kw):
    bdx = numpy.round(numpy.linspace(0, im.shape[0], nx+1)).astype('i4')
    bdlx = numpy.clip(bdx - overlap, 0, im.shape[0])
    bdrx = numpy.clip(bdx + overlap, 0, im.shape[0])
    bdy = numpy.round(numpy.linspace(0, im.shape[1], ny+1)).astype('i4')
    bdly = numpy.clip(bdy - overlap, 0, im.shape[1])
    bdry = numpy.clip(bdy + overlap, 0, im.shape[1])
    modelim = numpy.zeros_like(im)
    skyim = numpy.zeros_like(im)
    prisofar = numpy.zeros_like(im, dtype='bool')
    # this holder for stars gets filled out more completely later after
    # the first fit; for the moment, we just want the critical fields to
    # exist
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
            ol2 = overlap / 2
            mfixed &= ~in_bounds(stars['x'], stars['y'],
                                 [bdx[i]-0.5-ol2, bdx[i+1]-0.5+ol2],
                                 [bdy[j]-0.5-ol2, bdy[j+1]-0.5+ol2])
            xp, yp = (numpy.round(c).astype('i4')
                      for c in (stars['x'], stars['y']))
            mfixed &= (stars['primary'] == 1) | (prisofar[xp, yp] == 0)
            fixedstars = {f: stars[f][mfixed] for f in stars.dtype.names}
            fixedstars['x'] -= bdlx[i]
            fixedstars['y'] -= bdly[j]
            fixedstars['psfob'] = psfs
            fixedstars['offset'] = (bdlx[i], bdly[j])
            if (i == 0) and (j == 0):
                tpsf = psf
            elif j != 0:
                tpsf = psfs[-1]
            else:
                tpsf = psfs[-ny]
            if blist is not None:  # cut to bright stars in subimage
                mb = ((blist[0] >= bdlx[i]) & (blist[0] <= bdrx[i+1]) &
                      (blist[1] >= bdly[j]) & (blist[1] <= bdry[j+1]))
                blist0 = [blist[0][mb]-bdlx[i], blist[1][mb]-bdly[j],
                          blist[2][mb]]
                # offset X & Y to new positions
            else:
                blist0 = None
            res0 = crowdsource_base.fit_im(im[sall].copy(), tpsf,
                                      weight=weight[sall].copy(),
                                      dq=dq[sall].copy(),
                                      fixedstars=fixedstars,
                                      blist=blist0,
                                      **kw)
            newstars, skypar0, model0, sky0, psf0 = res0
            newstars['x'] += bdlx[i]
            newstars['y'] += bdly[j]
            primary0 = in_bounds(newstars['x'], newstars['y'],
                                 [bdx[i]-0.5, bdx[i+1]-0.5],
                                 [bdy[j]-0.5, bdy[j+1]-0.5])
            newstars['primary'] = primary0
            newstars['psf'] = (numpy.ones(len(newstars['x']), dtype='i4') *
                               len(psfs))
            dtypenames = newstars.keys()
            dtypeformats = [newstars[n].dtype for n in dtypenames]
            dtype = dict(names=dtypenames, formats=dtypeformats)
            newstars = numpy.fromiter(zip(*newstars.itervalues()),
                                      dtype=dtype, count=len(newstars['x']))
            stars = (newstars if len(stars) == 0
                     else numpy.append(stars, newstars))
            psf0.offset = (bdlx[i], bdly[j])
            psfs.append(psf0)
            modelim[spri] = model0[sfit]
            skyim[spri] = sky0[sfit]
            prisofar[spri] = sky0[sfit]
            if kw.get('verbose', False):
                t1 = time.time()
                print('Fit tile (%d, %d) of (%d, %d); %d sec elapsed' %
                      (i+1, j+1, nx, ny, t1-t0))
                t0 = t1            # import csplot
                sys.stdout.flush()
    stars = stars[stars['primary'] == 1]
    from matplotlib.mlab import rec_drop_fields
    stars = rec_drop_fields(stars, ['primary'])
    return stars, modelim, skyim, psfs
