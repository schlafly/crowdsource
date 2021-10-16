import os
import pdb
import numpy
from astropy.io import fits
from astropy import wcs
from pkg_resources import resource_filename


def get_overlaps(coadd_id):
    overlaps = getattr(get_overlaps, 'overlaps', None)
    if overlaps is None:
        fn = os.environ.get('UNWISE_OVERLAPS', None)
        if fn is None:
            fn = resource_filename('unwise_psf',
                                   'data/tile_overlaps-atlas.fits')
        overlaps = fits.getdata(fn)
        get_overlaps.overlaps = overlaps
    w = numpy.flatnonzero(overlaps['coadd_id'] == coadd_id)
    if len(w) == 0:
        raise ValueError('no matching coadd_id')
    toverlaps = overlaps['nearby_tiles'][w[0], :]
    toverlaps = numpy.array([s for s in toverlaps if s.strip() != ''])
    return toverlaps


def get_astr(coadd_id):
    astr = getattr(get_astr, 'astr', None)
    if astr is None:
        fn = resource_filename('unwise_psf', 'data/astrom-atlas.fits')
        astr = fits.getdata(fn)
        get_astr.astr = astr
    w = numpy.flatnonzero(astr['coadd_id'] == coadd_id)
    if len(w) == 0:
        raise ValueError('no matching coadd_id')
    return astr[w[0]]


def min_edge_dist(x, y):
    if len(x) != len(y):
        raise ValueError('len(x) must equal len(y)')
    return numpy.min(numpy.array([x + 0.5, y + 0.5, 2047.5-x, 2047.5-y]),
                     axis=0)


def make_wcs(astr):
    twcs = wcs.WCS(naxis=2)
    twcs.wcs.crpix = astr['crpix']
    twcs.wcs.cdelt = astr['cdelt']
    twcs.wcs.crval = astr['crval']
    twcs.wcs.cd = astr['cd']
    twcs.wcs.ctype = (str(astr['ctype'][0]),
                      str(astr['ctype'][1]))
    twcs.wcs.lonpole = astr['longpole']
    twcs.wcs.latpole = astr['latpole']
    return twcs


def is_primary(coadd_id, ra, dec):
    if not numpy.isscalar(coadd_id):
        raise ValueError('coadd_id must be a scalar')
    scalar = numpy.isscalar(ra)
    ra = numpy.atleast_1d(ra)
    if len(ra) != len(dec):
        raise ValueError('lengths of ra and dec must match')
    if ra.dtype.itemsize <= 4:
        raise ValueError('ra and dec must be at least double precision.')

    twcs = make_wcs(get_astr(coadd_id))
    x, y = twcs.all_world2pix(ra, dec, 0)

    worst_edge_dist = min_edge_dist(x, y)

    nearby_coadd_ids = get_overlaps(coadd_id)
    n_nearby = len(nearby_coadd_ids)
    is_primary = numpy.ones(len(ra), dtype='bool')
    for nearby_coadd_id in nearby_coadd_ids:
        twcs = make_wcs(get_astr(nearby_coadd_id))
        x, y = twcs.all_world2pix(ra, dec, 0)
        is_primary &= worst_edge_dist > min_edge_dist(x, y)

    return is_primary
