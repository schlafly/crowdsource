#!/usr/bin/env python

from __future__ import print_function, division

import os
import numpy as np
from scipy.ndimage import gaussian_filter
# quiet all of the annoying tensorflow compile model warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import models as kmodels
# import resource

def equalize_histogram(img, n_bins=256, asinh_stretch=False):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # Stretch the image with asinh in order to get more even histogram
    if asinh_stretch:
        vmin = np.nanmin(img)
        scale = np.nanpercentile(img-vmin, 50.)
        img = np.arcsinh((img-vmin) / scale)

    # get image histogram
    img_histogram, bins = np.histogram(img.flatten(), n_bins, density=False)
    cdf = img_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    img_equalized = np.interp(img.flatten(), bins[:-1], cdf)

    return img_equalized.reshape(img.shape), cdf


def equalize_histogram_wise(img, n_bins=256, asinh_stretch=False):
    # tweaked version for WISE

    # Stretch the image with asinh in order to get more even histogram
    if asinh_stretch:
        vmed = np.nanmedian(img)
        scale = np.nanpercentile(img, 30.)-np.nanpercentile(img, 10)
        scale = np.clip(scale, 100, np.inf)
        img = np.arcsinh((img-vmed) / scale)

    # get image histogram
    img_histogram, bins = np.histogram(img.flatten(), n_bins, density=False)
    cdf = img_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    img_equalized = np.interp(img.flatten(), bins[:-1], cdf)

    return img_equalized.reshape(img.shape), cdf


def load_model(fname_base):
    with open(fname_base + '.json', 'r') as f:
        model_json = f.read()

    model = kmodels.model_from_json(model_json)
    model.load_weights(fname_base + '.h5')

    return model


def subimages(img, shape, shiftx=0, shifty=0):
    j = np.arange(shiftx, img.shape[0]+shape[0]-1+shiftx, shape[0], dtype=int)
    k = np.arange(shifty, img.shape[1]+shape[1]-1+shifty, shape[1], dtype=int)

    jm = j[j <= img.shape[0]]
    km = k[k <= img.shape[1]]

    for j0, j1 in zip(jm[:-1], jm[1:]):
        for k0, k1 in zip(km[:-1], km[1:]):
            yield j0, k0, img[j0:j1, k0:k1]


def gen_mask(model, img):
    img = np.pad(img, 1, mode='constant', constant_values=np.median(img))
    _, h, w, _ = model.layers[0].input_shape

    mask = np.empty(img.shape, dtype='u1')

    for j0, k0, subimg in subimages(img, (h, w)):
        subimg, _ = equalize_histogram(subimg.astype('f8'),
                                       asinh_stretch=True, n_bins=3000)
        subimg /= 255.
        subimg.shape = (1, subimg.shape[0], subimg.shape[1], 1)
        pred = model.predict(subimg, batch_size=1)[0]
        mask[j0:j0+h, k0:k0+w] = np.argmax(pred*[0.25, 1, 1, 1])

    return mask[1:-1, 1:-1]


def gen_prob(model, img, return_prob=False):
    img = np.pad(img, 1, mode='constant', constant_values=np.median(img))
    _, h, w, _ = model.layers[0].input_shape

    mask = np.zeros((img.shape[0]-2, img.shape[1]-2, 4),
                    dtype=np.float32)
    mask_cnt = np.zeros((img.shape[0]-2, img.shape[1]-2, 4),
                        dtype=np.float32)

    # print ('Memory %s (KB)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    eps = 1e-4
    for shx in [0, 128, 256, 384]:
        for shy in [0, 128, 256, 384]:
            for j0, k0, subimg in subimages(img, (h, w), shiftx=shx,
                                            shifty=shy):
                subimg, _ = equalize_histogram(
                    subimg.astype('f8'), asinh_stretch=True, n_bins=3000)
                subimg /= 255.
                subimg.shape = (1, subimg.shape[0], subimg.shape[1], 1)
                pred = model.predict(subimg, batch_size=1)[0]

                x0, x1 = np.clip([j0, j0+h], 0, img.shape[0])
                y0, y1 = np.clip([k0, k0+w], 0, img.shape[1])

                mask[x0:x1, y0:y1, 0] += pred[0]*(pred[0]+eps)
                mask[x0:x1, y0:y1, 1] += pred[1]*(pred[0]+eps)
                mask[x0:x1, y0:y1, 2] += pred[2]*(pred[0]+eps)
                mask[x0:x1, y0:y1, 3] += pred[3]*(pred[0]+eps)
                mask_cnt[x0:x1, y0:y1] += (pred[0]+eps)
    np.divide(mask, mask_cnt, out=mask)
    for i in range(mask.shape[2]):
        gaussian_filter(mask[:, :, i], sigma=128, truncate=1,
                        mode='nearest', output=mask[:, :, i])
    # hard code decision boundary for now
    alpha = 2.0
    gam = 0.5
    decnum = np.zeros((mask.shape[0],mask.shape[1]),dtype=np.float32)
    np.divide(mask[:,:,0] + gam*mask[:,:,1],
        eps + mask[:,:,1] + mask[:,:,2] + mask[:,:,3],out=decnum)
    nebmask = decnum > alpha

    if return_prob:
        res = (nebmask, mask)
    else:
        res = nebmask
    return res


def gen_mask_wise(model, img, return_prob=False):
    _, h, w, _ = model.layers[0].input_shape

    mask = np.zeros(img.shape + (3,), dtype='f4')
    nn = np.zeros(img.shape, dtype='i4')

    # loop over shifts.  Size is 256, so 64 pixel shifts?
    # Just do simple average.
    for shx in [0, 64, 128, 192]:
        for shy in [0, 64, 128, 192]:
            for j0, k0, subimg in subimages(img, (h, w), shiftx=shx,
                                            shifty=shy):
                subimg, _ = equalize_histogram_wise(
                    subimg.astype('f8'), asinh_stretch=True, n_bins=3000)
                subimg /= 255.
                subimg.shape = (1, subimg.shape[0], subimg.shape[1], 1)
                pred = model.predict(subimg, batch_size=1)[0]
                x0, x1 = np.clip([j0, j0+h], 0, img.shape[0])
                y0, y1 = np.clip([k0, k0+w], 0, img.shape[1])
                mask[x0:x1, y0:y1, :] += pred
                nn[x0:x1, y0:y1] += 1
                # light, normal, nebulosity
    mask /= nn[:, :, None]
    mask *= np.array([1, 1, 0.5])[None, None, :]
    for i in range(mask.shape[2]):
        gaussian_filter(mask[:, :, i], sigma=64, truncate=1,
                        output=mask[:, :, i])

    nebmask = np.argmax(mask, axis=2)
    nebmask[nebmask == 0] = 1  # nebulosity_light -> normal

    if return_prob:
        res = (nebmask, mask)
    else:
        res = nebmask
    return res


def test_plots(model, imfns, extname='N26'):
    from matplotlib import pyplot as p
    from astropy.io import fits
    import os
    for timfn in imfns:
        tim = fits.getdata(timfn, extname='S7')
        mask = gen_mask(model, tim)
        if np.any(mask != 2):
            print(timfn, np.sum(mask == 0)/1./np.sum(np.isfinite(mask)),
                  np.sum(mask == 1)/1./np.sum(np.isfinite(mask)),
                  np.sum(mask == 3)/1./np.sum(np.isfinite(mask)))
            p.clf()
            p.imshow(((tim-np.median(tim))).T, aspect='equal', vmin=-50,
                     vmax=50, interpolation='none', cmap='binary',
                     origin='lower')
            p.imshow(mask.T, cmap='jet', alpha=0.2, vmin=0, vmax=3,
                     interpolation='none', origin='lower')
            p.draw()
            p.savefig(os.path.basename(timfn)+'.mask.png')


def main():
    from PIL import Image

    model = load_model('toy_data/19th_try')

    img = Image.open('toy_data/test_image.png')
    img = np.array(img)

    mask = gen_mask(model, img)
    mask = Image.fromarray((255.*mask/2.).astype('u1'), mode='L')
    mask.save('toy_data/test_image_mask.png')

    return 0


if __name__ == '__main__':
    main()
