# crowdsource

A crowded field photometry pipeline

| PyPi | Conda | Platforms|
| --- | --- | --- |
|[![][pypi-img]][pypi-url]|[![][conda-img]][conda-url]|[![][plat-img]][plat-url]
|[![][pypid-img]][pypid-url]|[![][condad-img]][condad-url]|

<!-- URLS -->
[pypi-img]: https://img.shields.io/pypi/v/crowdsourcephoto.svg
[pypi-url]: https://pypi.org/project/crowdsourcephoto

[conda-img]: https://img.shields.io/conda/vn/conda-forge/crowdsourcephoto.svg
[conda-url]: https://anaconda.org/conda-forge/crowdsourcephoto

[pypid-img]: https://img.shields.io/pypi/dm/crowdsourcephoto.svg?label=Pypi%20downloads
[pypid-url]: https://pypi.org/project/crowdsourcephoto/

[condad-img]: https://img.shields.io/conda/dn/conda-forge/crowdsourcephoto.svg?label=Conda%20downloads
[condad-url]: https://anaconda.org/conda-forge/crowdsourcephoto

[plat-img]: https://img.shields.io/conda/pn/conda-forge/crowdsourcephoto.svg
[plat-url]: https://anaconda.org/conda-forge/crowdsourcephoto

## Installation

The recommended installation uses conda-forge as

```python
conda install -c conda-forge crowdsourcephoto
```

A conda-forge version of `crowdsourcephoto` is available on windows, but if there are issues with `tensorflow`, Windows users should fall back to pip (below).

This package can also be installed using pip as

```python
pip install crowdsourcephoto
```

## Developer Notes

Strategy:
  * main loop:
    - remove sky from image
    - identify peaks in residual
    - add new peaks to source list
    - fit all sources (to sky subtracted image)
    - compute centroids
    - shift peak locations according to centroids
    - compute new psf
    - loop
  * generally: we start with no knowledge, remove a rough sky (the median),
    find the brighter peaks (no fainter ones, since the sky is biased high)
    fit the sources we have so far, compute centroids, compute an improved
    PSF.  We now have a model for the image, and can now:
    - subtract it, recompute median to get better sky estimate
    - find fainter peaks (former blends, or not seen because of too-high sky)
    - better centroids (better removal of other sources)
    - better PSF (better centroids & better removal of other sources)
    - repeat.
  * sky
    - generally: subtract median of image - current model
    - first two iterations: subtract single median for entire image
    - afterwards: subtract median of each 20x20 pixel region
      this seems needed to deal with:
      - wings of bright stars
      - wings of bright stars not in image
      - scattered light from dust
      - other diffuse features?
  * peaks
    - two kinds: bright and faint
    - bright: pixels with individual S/N > 5/0.3
    - faint: pixels with PSF-convolved significance > 5
    - bit of a mess to determine whether or not to keep a peak
      - only accept bright peaks if they're not on the edge of a mask
      - only accept faint peaks if their peak pixel is faint enough
        (otherwise they should be bright peaks)
      - only accept faint peaks if the centers aren't saturated
        (or this is the first iteration; we don't want to find lots
         of peaks in masked saturated regions), or if the peak is on
         the edge of a mask.
      - in both cases, we need the peak to also be "significant"
        - the PSF-convolved significance is greater than the model
          PSF-convolved significance times some factor
          (0.2 at the moment?)
          we're trying to allow faint sources to be accepted, but
          don't want to pull out too much garbage in PSF residuals
        - the pixel flux over the model flux exceeds some factor
          (0.1 at the moment?), and the pixel significance exceeds
          the PSF-convolved model significance by some factor
          (0.4 at the moment).  This seems especially arbitrary; was
          put in to avoid depending only on a single pixel's
          significance: negative neighbors are a sign we don't want
          to add a new source there.
  * fit
    - build large sparse matrix with PSFs and PSF derivative in x & y
      shifted to be at current source locations
      larger PSFs for objects with central pixel brightnesses exceeding
      some value
      PSF derivatives excluded in last iteration; just forced photometry
    - perform some renormalizations of the columns of the matrix to
      speed convergence
    - solve using LSQR
    - build model image & residual image
  * centroids
    - compute residual image
    - for each source:
     - add back in this source's model (PSF, derivatives)
     - compute centroid on this other-source-subtracted image
     - pixels in centroid are weighted according to weight image and
       the current psf estimate
     - final centroid multiplied by two to address multiplicative bias
     - additive bias computed from computed centroid of a true PSF
     - centroids of true PSFs at each position in the weight image
       also computed, to address biases due to weight images
     - possible multiplicative bias due to weight image not addressed
     - Hogg has recent paper saying finding the peak in the PSF-convolved
       image is better.  Worries:
      - current scheme really only cares about center of stamp;
        psf-convolution would be more likely to chase things in the
        wings
      - would need much bigger stamps for PSF convolution
      - but PSF convolution would be much more straightforward
        mathematically (I think?)
      - note: in initial iteration, the PSF may be quite bad
        accordingly, we use dumb ~aperture model for PSF in this iteration,
        basically not using the PSF weight and just computing the barycenter
        of the light.
  * find new psf
    - compute "psf_qf" quality factors for all the stars
    - compute fluxes on other-source-subtracted images
    - compute fluxes on other-source-subtracted images after removing a
      median
    - compute total fluxes in the images, without subtracting other sources
    - throw away sources with any of the following
     - psf_qf different from 1,
     - large centroid shifts (>0.5 pixels)
     - less than half the total image flux coming from the model star
     - median subtraction removes more than 80% of the flux
    - median of the surviving stamps (after shifting according to centroids)
    - this is for central 19x19 stamp
    - to extend to 59x59 stamp to deal with bright stars, find best fit
      Moffat profile fit to 19x19 stamp
    - blend this with 19x19 stamp at boundary and extend to 59x59 by Moffat.


TODO:
 * flags based on flags in central pixel of DQ file
 * convert matrix construction step to Cython
  - not clear how big the win is (25%?), have to find a good interpolation package in C/C++ to link to.  Maybe possible to use the numpy one, but sounds like a bad idea?
 * increase match radius to at least ~0.375"; this is 0.25"* 7.5 / 5, 7.5 pix is roughly our worst seeing.
 * During average flux computation, when encountering multiple sources within 1 match radius of a single object, do something.  
   Options:
   - sum the fluxes
   - flag both detections; don't use their fluxes in averaging!

Improvements:
* consider multiple images simultaneously (major change)
* allow PSF to vary (significant change)
  - seems necessary to do better photometry (~2% level)
  - could require change the convolution kernel?  significance_image needs to know the PSF, but this is probably deeply second-order
* improve sky background estimation
  - reject regions around stars?
  - model background as sum of power law & Gaussian distributions?

My sense has been that the biggest problem is that we _have_ to do small scale sky corrections to deal with the wings of very bright stars, very bright stars off the focal plane, dust-associated diffuse light, etc.  Looking only in small regions, it's hard to imagine that a more sophisticated fit will actually be a good idea.  In practice, I guess we need to assess how bad our sky subtraction is actually doing by comparison of the photometry with much deeper data.
