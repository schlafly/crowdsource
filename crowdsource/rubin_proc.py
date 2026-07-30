
import numpy as np
import argparse, os, pdb
import crowdsource.psf as psfmod
from crowdsource import crowdsource_base
from lsst.daf.butler import Butler
from functools import partial


def process(visitId, detector, nx = 4, ny = 4, maxstars = 10000000, fewstars = 60, threshold = 5, **kw):
    """
    Parameters
    ----------
    visitId : Rubin images are taken with an associated visitId attatched in the metadata. There are nine
              images associated with one visitId.
    detector: Determines which of the nine images will be processed. 
    
    Process: 
    1. Use the RSP Butler to find the Rubin image
    2. Get the PSF using wise_psf_fit
    3. Making the variables for CROWDSOURCE
    4. Run CROWDSOURCE on an image

    Returns
    -------
    res : The processed image as an array. 

    """
 
    butler = Butler("dp1", collections="LSSTComCam/DP1")
    
    dataset_refs = list(butler.query_datasets(
        "visit_image",
        where="visit.id = :visitId AND detector.id = :detector",
        bind={"visitId": visitId, "detector": detector},
    ))
 
    if len(dataset_refs) == 0:
        raise RuntimeError(
            f"No visit_image found for visit={visitId}, detector={detector}")
 
    else:
        print(f"Visit image found for visit={visitId}, detector={detector}")
            
    ref = list(dataset_refs)[0]
    visit_image = butler.get(ref)

    #Getting the PSF    
    rubin_psf = visit_image.getPsf()
    visit_center = visit_image.getBBox().getCenter()

    psf_stamp_visit = rubin_psf.computeKernelImage(visit_center).array
    
    stamp = np.clip(np.array(psf_stamp_visit), 1e-10, np.inf)
    stamp = stamp / np.sum(stamp)
    
    psf = psfmod.SimplePSF(stamp)

    print("Using fit_variable_moffat_psf")
    psf.fitfun = psfmod.fit_variable_moffat_psf

    #crowdsource variables
    im = visit_image.image.array.astype(np.float32)
    
    var = visit_image.variance.array
    sqivar = np.where(var > 0, 1.0 / np.sqrt(var), 0.0).astype(np.float32)
    
    mask = visit_image.mask
    
    bad_bits = 0
    for plane in ("BAD", "SAT", "CR", "NO_DATA", "EDGE", "INTRP", "STREAK"):
        if plane in mask.getMaskPlaneDict():
            bad_bits |= mask.getPlaneBitMask(plane)
            
    flag = (mask.array & bad_bits).astype(np.uint32)
    sqivar[(mask.array & bad_bits) != 0] = 0.0

    #run crowdsource
    res = crowdsource_base.fit_im(im, psf, sqivar, dq=flag, refit_psf=True,
                                  verbose = True, ntilex=nx, ntiley=ny, maxiter = 10, threshold = threshold, **kw)
 
    print("CROWDSOURCE is done!")
    
    return res

if __name__ == "__main__":
    from astropy.io import fits
 
    parser = argparse.ArgumentParser(description='Run crowdsource on a Rubin visit_image')
    # 3 arguments: visitId, detector, outfn
    parser.add_argument('visitId', type=int, nargs=1)
    parser.add_argument('detector', type=int, nargs=1)
    parser.add_argument('outfn', type=str, nargs=1)
    parser.add_argument('--nx', '-x', type=int, default=4,
                        help='number of tiles in x')
    parser.add_argument('--ny', '-y', type=int, default=4,
                        help='number of tiles in y')
    parser.add_argument('--maxstars', type=int, default=10000000,
                        help='maximum number of stars to fit')
    parser.add_argument('--fewstars', type=int, default=60,
                        help='number of stars below which a tile is considered to have few stars')
    parser.add_argument('--threshold', '-t', type=float, default=5,
                        help='detection threshold in sigma')
    args = parser.parse_args()
 
    visitId = args.visitId[0]
    detector = args.detector[0]
    outfn = args.outfn[0]
 
    res = process(visitId, detector, nx=args.nx, ny=args.ny,
                  maxstars=args.maxstars, fewstars=args.fewstars,
                  threshold=args.threshold)
 
    fits.writeto(outfn, res[0], overwrite=True)
    fits.append(outfn, res[1][0])
    fits.append(outfn, res[2][0])
