import numpy as np
import numpy.ma as ma
import seaborn as sns
import os as os
import operator as op
import pickle
import time

from numpy.fft import ifft2, fftshift
from itertools import product
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel, convolve
from scipy.optimize import curve_fit, brent
from collections import defaultdict

sns.set_style('whitegrid')
sns.color_palette('colorblind')


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def colourbar(mappable):
    """
    :param mappable: a map axes object taken as input to apply a colourbar to

    :return: Edits the figure and subplot to include a colourbar which is scaled to the map correctly
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    figure_one = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return figure_one.colorbar(mappable, cax=cax, format='%g')


def fourier_gaussian_function(axis_one, axis_two, scale=1.0, sigma_x=1.0, sigma_y=1, theta=0):
    xo = axis_one.shape[0] // 2
    yo = axis_two.shape[1] // 2
    if sigma_x == 0:
        sigma_x = 1
    if sigma_y == 0:
        sigma_y = 1
    sigma_x = 1 / sigma_x
    sigma_y = 1 / sigma_y

    a = np.cos(theta) ** 2 / (2 * sigma_x ** 2) + np.sin(theta) ** 2 / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
    c = np.sin(theta) ** 2 / (2 * sigma_x ** 2) + np.cos(theta) ** 2 / (2 * sigma_y ** 2)

    fourier_gaussian = scale * np.exp(
        (-4 * np.pi ** 2 / (axis_one.shape[0] ** 2)) *
        (a * (axis_one - xo) ** 2 +
         2 * b * (axis_one - xo) * (axis_two - yo) +
         c * (axis_two - yo) ** 2))
    return fourier_gaussian


def gaussian_fit_ac(auto_correlation):
    # figuring out where I need to clip to, realistically, this SHOULD be at the physical centre (200,200)
    width = 7
    y_max, x_max = np.where(auto_correlation == auto_correlation.max())
    y_max, x_max = int(np.amax(y_max)), int(np.amax(x_max))
    # Setting the middle auto_correlation point to be our estimated value of B for a better fit.

    mask = np.zeros(auto_correlation.shape)
    mask[y_max, x_max] = 1
    ac_masked = ma.masked_array(auto_correlation, mask=mask)

    # clipping map further to better fit a gaussian profile to it
    auto_correlation = ac_masked[y_max - width:y_max + width + 1, x_max - width:x_max + width + 1]

    # generating the gaussian to fit
    x_mesh, y_mesh = np.meshgrid(np.arange(auto_correlation.shape[0]), np.arange(auto_correlation.shape[1]))
    gauss_init = Gaussian2D(
        amplitude=auto_correlation.max(),
        x_mean=auto_correlation.shape[1] // 2,  # location to start fitting gaussian
        y_mean=auto_correlation.shape[0] // 2,  # location to start fitting gaussian
    )
    fitting_gauss = LevMarLSQFitter()  # Fitting method; Levenberg-Marquardt Least Squares algorithm
    best_fit_gauss = fitting_gauss(gauss_init, x_mesh, y_mesh, auto_correlation)  # The best fit for the map
    gauss_model = best_fit_gauss(x_mesh, y_mesh)  # the model itself (if we want to plot it
    try:
        ac_error = np.sqrt(np.diag(fitting_gauss.fit_info['param_cov']))
    except ValueError:
        ac_error = np.ones(10) * -5
    amplitude = float(best_fit_gauss.amplitude.value)
    amplitude_error = ac_error[0]
    sigma_x = float(best_fit_gauss.x_stddev.value)
    sigma_x_error = ac_error[3]
    sigma_y = float(best_fit_gauss.y_stddev.value)
    sigma_y_error = ac_error[4]
    theta = float(best_fit_gauss.theta.value)
    theta_error = ac_error[5]

    return [[amplitude, sigma_x, sigma_y, theta],
            [amplitude_error, sigma_x_error, sigma_y_error, theta_error]], gauss_model


def gaussian_fit_xc(x_correlation):
    # import numpy as np
    # from astropy.modeling.models import Gaussian2D
    # from astropy.modeling.fitting import LevMarLSQFitter
    # figuring out where i need to clip to
    y_center = x_correlation.shape[0] // 2
    x_center = x_correlation.shape[1] // 2  # centre of the Cross-Corr maps default: (200,200)
    width = 7
    y_max, x_max = np.where(x_correlation == x_correlation.max())
    y_max = int(y_max)
    x_max = int(x_max)

    # clipping map further to better fit a gaussian profile to it
    x_correlation = x_correlation[y_max - width:y_max + width + 1, x_max - width:x_max + width + 1]
    # subtracting half the side to then add the mean values after
    x_max -= x_correlation.shape[1] // 2
    y_max -= x_correlation.shape[0] // 2
    # generating the gaussian to fit.

    x_mesh, y_mesh = np.meshgrid(np.arange(x_correlation.shape[0]), np.arange(x_correlation.shape[1]))
    gauss_init = Gaussian2D(
        amplitude=x_correlation.max(),
        x_mean=np.where(x_correlation == x_correlation.max())[1],  # location to start fitting gaussian
        y_mean=np.where(x_correlation == x_correlation.max())[0],  # location to start fitting gaussian
        # fixed={},  # any fixed parameters
        bounds={
            # 'amplitude': (x_correlation.max() * 0.90, x_correlation.max() * 1.10),
            'x_mean': (int(np.where(x_correlation == x_correlation.max())[1]) - 1,
                       int(np.where(x_correlation == x_correlation.max())[1]) + 1),
            'y_mean': (int(np.where(x_correlation == x_correlation.max())[0]) - 1,
                       int(np.where(x_correlation == x_correlation.max())[0]) + 1)
        },  # allowing var in amplitude to better fit gauss
    )
    fitting_gauss = LevMarLSQFitter()  # Fitting method; Levenberg-Marquardt Least Squares algorithm
    best_fit_gauss = fitting_gauss(gauss_init, x_mesh, y_mesh, x_correlation)  # The best fit for the map

    # now we can get the location of our peak fitted gaussian and add them back to get a total offset
    y_max += best_fit_gauss.y_mean.value  # Finding the distance from 0,0 to the centre gaussian
    x_max += best_fit_gauss.x_mean.value  # and y.
    try:
        x_correlation_error = np.sqrt(np.diag(fitting_gauss.fit_info['param_cov']))
    except ValueError:
        x_correlation_error = np.ones(10) * -5
    offset = (x_center - x_max, y_center - y_max)
    offset_err = (x_correlation_error[1], x_correlation_error[2])
    return offset, offset_err


def correlate(epoch_1=None, epoch_2=None, clipped_side=400, clip_only=False, psd=False):
    """
    :param epoch_1:
        2-Dimensional numpy array. Default: None
        When only epoch_1 is passed it is auto correlated with itself
    :param epoch_2:
        2-Dimensional numpy array. Default: None
        When both epoch_1 and epoch_2 are passed the two arrays are cross correlated
    :param clipped_side:
        Integer. Default: 400.
        The length of one side of the clipped array.
    :param clip_only:
        Boolean. Default: False
        When True is passed to clip_only it will only clip epoch_1
    :param psd:
        Boolean. Default: False
        When true is passed the power spectrum is returned
    :return:
    """
    from numpy.fft import fft2, ifft2, fftshift
    if clip_only:
        mid_map_x, mid_map_y = epoch_1.shape[1] // 2, epoch_1.shape[0] // 2
        clipped_epoch = epoch_1[mid_map_y - clipped_side // 2:mid_map_y + clipped_side // 2 + 1,
                                mid_map_x - clipped_side // 2:mid_map_x + clipped_side // 2 + 1
                                ]
        return clipped_epoch

    elif psd:
        mid_map_x, mid_map_y = epoch_1.shape[1] // 2, epoch_1.shape[0] // 2
        clipped_epoch = epoch_1[mid_map_y - clipped_side // 2:mid_map_y + clipped_side // 2 + 1,
                                mid_map_x - clipped_side // 2:mid_map_x + clipped_side // 2 + 1
                                ]
        psd = fft2(clipped_epoch) * fft2(clipped_epoch).conj()
        return fftshift(psd)

    elif epoch_1 is None:
        raise Exception('You need to pass a 2D map for this function to work')

    elif epoch_2 is None:
        mid_map_x, mid_map_y = epoch_1.shape[1] // 2, epoch_1.shape[0] // 2
        clipped_epoch = epoch_1[mid_map_y - clipped_side // 2:mid_map_y + clipped_side // 2 + 1,
                                mid_map_x - clipped_side // 2:mid_map_x + clipped_side // 2 + 1
                                ]
        ac = ifft2(fft2(clipped_epoch) * fft2(clipped_epoch).conj())
        return fftshift(ac)

    else:
        mid_map_x_1, mid_map_y_1 = epoch_1.shape[1] // 2, epoch_1.shape[0] // 2
        mid_map_x_2, mid_map_y_2 = epoch_2.shape[1] // 2, epoch_2.shape[0] // 2
        clipped_epoch_1 = epoch_1[mid_map_y_1 - clipped_side // 2:mid_map_y_1 + clipped_side // 2 + 1,
                                  mid_map_x_1 - clipped_side // 2:mid_map_x_1 + clipped_side // 2 + 1
                                  ]
        clipped_epoch_2 = epoch_2[mid_map_y_2 - clipped_side // 2:mid_map_y_2 + clipped_side // 2 + 1,
                                  mid_map_x_2 - clipped_side // 2:mid_map_x_2 + clipped_side // 2 + 1
                                  ]
        x_correlation = ifft2(fft2(clipped_epoch_1) * fft2(clipped_epoch_2).conj())
        return fftshift(x_correlation)


def f(independent, m, b):
    """
    :param independent: independent variable
    :param m: slope
    :param b: intercept
    :return: y: a quadratic
    """
    dependent = m * independent ** 2 + b
    return dependent


def f_linear(p, independent):
    """
    :param independent: independent variable
    :param p: fitting parameters
    :return: y: a linear monomial
    """

    dependent = p[0] * independent + p[1]
    return dependent


def amp(epoch):
    from numpy import sqrt
    return sqrt(epoch.real ** 2 + epoch.imag ** 2)


def beam_fit(sigma, power_spectrum, required_length_scale):
    from numpy import meshgrid, arange, sqrt
    from numpy.fft import ifft2, fftshift

    axis_1_size = axis_2_size = power_spectrum.shape[0]
    axis_1, axis_2 = meshgrid(arange(axis_1_size), arange(axis_2_size))
    numeric_gaussian = fourier_gaussian_function(axis_1, axis_2, sigma_x=sigma, sigma_y=sigma)  # guess!
    output_gaussian = amp(fftshift(ifft2(numeric_gaussian * numeric_gaussian * power_spectrum)))  # guess amplitude
    [[_, sigma_x, sigma_y, _], _], _ = gaussian_fit_ac(output_gaussian)

    length_scale = sqrt(sigma_x * sigma_y)

    dif = required_length_scale - length_scale
    return abs(dif)


# + ===================== +
# | Root project location |
# + ===================== +
LOCAL_ROOT = '/home/cobr/Documents/jcmt-variability/'
ROOT = '/media/cobr/JCMT-TRANSIENT/'

# + ===================== +
# | Global parameters     |
# + ===================== +
DIST = 7  # the distance used for linear fitting and gaussian fitting (use width = RADIUS*2 + 1)
length = 200  # The size we clip the reference matrix to. size MxM = length*2 x length*2
TEST = False # was i testing code?
kernel_sigma = 6  # for the Cross-Correlation High-Pass filtering
kernel = Gaussian2DKernel(x_stddev=kernel_sigma, y_stddev=kernel_sigma) # Gaussian Kernel for High-Pass filter
TX = 'wavelength: {:}, epoch: {:}, Pass: {:}, length_scale: {:}, required_length_scale: {:}\n' #print out string
REGIONS = {
    'IC348': {'450': 5.7, '850': 5.9},
    'NGC1333': {'450': 5.1, '850': 5.9},
    'NGC2024': {'450': 7.8, '850': 10.7},
    'NGC2071': {'450': 5.1, '850': 5.8},
    'OMC23': {'450': 6.3, '850': 8.8},
    'OPH_CORE': {'450': 7.2, '850': 9.7},
    'SERPENS_MAIN': {'450': 5.3, '850': 6.9},
    'SERPENS_SOUTH': {'450': 6.9, '850': 8.6}}  # Dictionary of region and an ideal "beam" at both wavelenghts
tol = 0.05  # tolerance for beam convolution

wavelengths = ['450', '850']
data = defaultdict(dict)

for region in list(REGIONS.keys()):

    """
    I used default dict to quickly circumvent "missing" data epochs, 
    so that it would default to a certain value if it was looked up.
    """
    data[region] = defaultdict(dict)
    data[region]['850'] = defaultdict(dict)
    data[region]['850']['epoch'] = defaultdict(list)

    data[region]['850']['dates'] = list()  # to collect all of the dates in the data[region] set
    data[region]['850']['JCMT_offset'] = defaultdict(str)  # to use the date as the index

    data[region]['850']['XC'] = defaultdict(dict)
    data[region]['850']['XC']['offset'] = defaultdict(list)
    data[region]['850']['XC']['offset_err'] = defaultdict(list)

    data[region]['850']['linear'] = defaultdict(dict)
    data[region]['850']['linear']['m'] = defaultdict(dict)
    data[region]['850']['linear']['m_err'] = defaultdict(dict)
    data[region]['850']['linear']['b'] = defaultdict(dict)
    data[region]['850']['linear']['b_err'] = defaultdict(dict)

    data[region]['850']['linear_new'] = defaultdict(list)
    data[region]['850']['linear_new']['m'] = defaultdict(dict)
    data[region]['850']['linear_new']['m_err'] = defaultdict(dict)
    data[region]['850']['linear_new']['b'] = defaultdict(dict)
    data[region]['850']['linear_new']['b_err'] = defaultdict(dict)

    data[region]['850']['AC'] = defaultdict(dict)
    data[region]['850']['AC']['amp'] = defaultdict(list)
    data[region]['850']['AC']['amp_err'] = defaultdict(list)
    data[region]['850']['AC']['sig_x'] = defaultdict(list)
    data[region]['850']['AC']['sig_x_err'] = defaultdict(list)
    data[region]['850']['AC']['sig_y'] = defaultdict(list)
    data[region]['850']['AC']['sig_y_err'] = defaultdict(list)
    data[region]['850']['AC']['theta'] = defaultdict(list)
    data[region]['850']['AC']['theta_err'] = defaultdict(list)

    data[region]['850']['AC_New'] = defaultdict(dict)
    data[region]['850']['AC_New']['N'] = defaultdict(list)
    data[region]['850']['AC_New']['sigma'] = defaultdict(list)
    data[region]['850']['AC_New']['amp'] = defaultdict(list)
    data[region]['850']['AC_New']['amp_err'] = defaultdict(list)
    data[region]['850']['AC_New']['sig_x'] = defaultdict(list)
    data[region]['850']['AC_New']['sig_x_err'] = defaultdict(list)
    data[region]['850']['AC_New']['sig_y'] = defaultdict(list)
    data[region]['850']['AC_New']['sig_y_err'] = defaultdict(list)
    data[region]['850']['AC_New']['theta'] = defaultdict(list)
    data[region]['850']['AC_New']['theta_err'] = defaultdict(list)

    data[region]['450'] = defaultdict(dict)
    data[region]['450']['epoch'] = defaultdict(list)
    data[region]['450']['dates'] = list()  # to collect all of the dates in the data[region] set
    data[region]['450']['JCMT_offset'] = defaultdict(str)  # to use the date as the index

    data[region]['450']['XC'] = defaultdict(list)
    data[region]['450']['XC']['offset'] = defaultdict(list)
    data[region]['450']['XC']['offset_err'] = defaultdict(list)

    data[region]['450']['linear'] = defaultdict(list)
    data[region]['450']['linear']['m'] = defaultdict(dict)
    data[region]['450']['linear']['m_err'] = defaultdict(dict)
    data[region]['450']['linear']['b'] = defaultdict(dict)
    data[region]['450']['linear']['b_err'] = defaultdict(dict)

    data[region]['450']['linear_new'] = defaultdict(list)
    data[region]['450']['linear_new']['m'] = defaultdict(dict)
    data[region]['450']['linear_new']['m_err'] = defaultdict(dict)
    data[region]['450']['linear_new']['b'] = defaultdict(dict)
    data[region]['450']['linear_new']['b_err'] = defaultdict(dict)

    data[region]['450']['AC'] = defaultdict(dict)
    data[region]['450']['AC']['amp'] = defaultdict(int)
    data[region]['450']['AC']['amp_err'] = defaultdict(int)
    data[region]['450']['AC']['sig_x'] = defaultdict(int)
    data[region]['450']['AC']['sig_x_err'] = defaultdict(int)
    data[region]['450']['AC']['sig_y'] = defaultdict(int)
    data[region]['450']['AC']['sig_y_err'] = defaultdict(int)
    data[region]['450']['AC']['theta'] = defaultdict(int)
    data[region]['450']['AC']['theta_err'] = defaultdict(int)

    data[region]['450']['AC_New'] = defaultdict(dict)
    data[region]['450']['AC_New']['N'] = defaultdict(int)
    data[region]['450']['AC_New']['sigma'] = defaultdict(int)
    data[region]['450']['AC_New']['amp'] = defaultdict(int)
    data[region]['450']['AC_New']['amp_err'] = defaultdict(int)
    data[region]['450']['AC_New']['sig_x'] = defaultdict(int)
    data[region]['450']['AC_New']['sig_x_err'] = defaultdict(int)
    data[region]['450']['AC_New']['sig_y'] = defaultdict(int)
    data[region]['450']['AC_New']['sig_y_err'] = defaultdict(int)
    data[region]['450']['AC_New']['theta'] = defaultdict(int)
    data[region]['450']['AC_New']['theta_err'] = defaultdict(int)
    
    TIME_START = time.time()

    print(region+'\n'+'='*len(region))
    with open('/home/cobr/Documents/jcmt-variability/log/{:}_BC.log'.format(region), 'w+') as LOG:
        Dates850 = []
        Dates450 = []
        DataRoot = ROOT + region + "/A3_images/"  # where all the data is stored
        files = os.listdir(DataRoot)  # listing all the files in root
        files = sorted(files)  # sorting to ensure we select the correct first region

        MetaData850 = np.loadtxt(ROOT + region + '/A3_images_cal/' + region + '_850_EA3_cal_metadata.txt',
                                 dtype=str)
        MetaData450 = np.loadtxt(ROOT + region + '/A3_images_cal_450/' + region + '_450_EA3_cal_metadata.txt',
                                 dtype=str)

        FN850 = MetaData850.T[1]  # filename of the 850 metadata files (ordered)
        FN450 = MetaData450.T[1]  # filename of the 450 metadata files (ordered)

        Dates850.extend([''.join(d[1:].split('-')) for d in MetaData850.T[2]])  # dates of all the 850 metadata files
        Dates450.extend([''.join(d[1:].split('-')) for d in MetaData450.T[2]])  # dates of all the 450 metadata files

        for wavelength in wavelengths:
            print(wavelength)
            if wavelength in files[0]:
                FirstEpochName = files[0]  # the name of the first epoch
            elif wavelength in files[1]:
                FirstEpochName = files[1]
            else:
                print('Issue with first epoch...')
                break

            """
            First epoch data for cross-correlation, applying high-pass filter as well
            """
            FirstEpoch = fits.open(DataRoot + '/' + FirstEpochName)
            FirstEpochData = FirstEpoch[0].data[0]  # Numpy data array for the first epoch
            FirstEpochCentre = np.array(
                [FirstEpoch[0].header['CRPIX1'], FirstEpoch[0].header['CRPIX2']])  # loc of actual centre

            # middle of the map of the first epoch
            FED_MidMapX = FirstEpochData.shape[1] // 2
            FED_MidMapY = FirstEpochData.shape[0] // 2
            FirstEpochVec = np.array([FirstEpochCentre[0] - FED_MidMapX,
                                      FirstEpochCentre[1] - FED_MidMapY])
            FirstEpochData = FirstEpochData[
                             FED_MidMapY - length:FED_MidMapY + length + 1,
                             FED_MidMapX - length:FED_MidMapX + length + 1]
            FirstEpochData_smooth = convolve(FirstEpochData, kernel)
            FirstEpochData -= FirstEpochData_smooth
            Files = []
            for fn in files:
                SUCCESS = False  # was the numerical brent method successful? assume no to begin.
                if wavelength in fn:
                    Files.append(fn)
                    FilePath = ROOT + region + "/A3_images/" + fn
                    if os.path.isfile(FilePath) and (fn[-4:].lower() == ('.fit' or '.fits')):
                        hdul = fits.open(FilePath)  # opening the file in astropy
                        date = str(hdul[0].header['UTDATE'])  # extracting the date from the header
                        date += '-' + str(hdul[0].header['OBSNUM'])
                        print('Epoch: {:14}'.format(date))
                        data[region][wavelength]['dates'].append(date)
                        centre = (hdul[0].header['CRPIX1'], hdul[0].header['CRPIX2'])  # JCMT's alleged centre is
                        Vec = np.array([centre[0] - (hdul[0].shape[2] // 2),
                                        centre[1] - (hdul[0].shape[1] // 2)])
                        JCMT_offset = FirstEpochVec - Vec  # JCMT offset from headers
                        data[region][wavelength]['JCMT_offset'][date] = JCMT_offset  # used for accessing data later.

                        hdu = hdul[0]  # a nice compact way to store the data for later.

                        Epoch = hdu.data[0]  # map of the region
                        Map_of_Region = interpolate_replace_nans(
                            correlate(Epoch, clip_only=True),
                            Gaussian2DKernel(5)
                        )  # replacing any NaNs from steve's smoothing/filtering using interpolation
                        Map_of_Region_smooth = convolve(Map_of_Region, kernel)  # making smoothed map
                        Map_of_RegionXC = Map_of_Region - Map_of_Region_smooth  # High-pass filter!


                        """
                        NOTE:
                        In the following I take the Real part of my correlation;
                        Python will cast the FFT to a complex number, and then when inverting w/ 
                        IFFT the data remains complex. Since we pass Real data into the FFT, and do not synthetically
                        alter the imaginary parts, we ***should*** expect real valued Cross/Auto correlations.
                        """
                        XC = correlate(epoch_1=Map_of_RegionXC, epoch_2=FirstEpochData).real
                        PS = correlate(Map_of_Region, psd=True)
                        AC = correlate(Map_of_Region).real  # auto correlation of the map

                        try:
                            XC_Offset, XC_Error = gaussian_fit_xc(XC)
                        except ValueError:
                            XC_Offset = (-1, -1)
                            XC_Error = (-1, -1)

                        [[AMP, SIGX, SIGY, THETA], [AMP_ERR, SIGX_ERR, SIGY_ERR, THETA_ERR]], _ = gaussian_fit_ac(AC)
                        # Above, the _ is trashing a the model that is returned from my gaussian fitting function

                        data[region][wavelength]['XC']['offset'][date] = (XC_Offset[0], XC_Offset[1])
                        data[region][wavelength]['XC']['offset_err'][date] = (XC_Error[0], XC_Error[1])

                        data[region][wavelength]['AC']['amp'][date] = AMP
                        data[region][wavelength]['AC']['amp_err'][date] = AMP_ERR
                        data[region][wavelength]['AC']['sig_x'][date] = SIGX
                        data[region][wavelength]['AC']['sig_x_err'][date] = SIGX_ERR
                        data[region][wavelength]['AC']['sig_y'][date] = SIGY
                        data[region][wavelength]['AC']['sig_y_err'][date] = SIGY_ERR
                        data[region][wavelength]['AC']['theta'][date] = THETA
                        data[region][wavelength]['AC']['theta_err'][date] = THETA_ERR

                        Clipped_Map_of_Region_LENGTH = np.arange(0, Map_of_Region.shape[0])

                        # creating a set of all possible index locations within the map
                        loc = list(product(Clipped_Map_of_Region_LENGTH, Clipped_Map_of_Region_LENGTH))

                        MidMapX = AC.shape[1] // 2  # middle of the map x
                        MidMapY = AC.shape[0] // 2  # and y

                        radius, AC_pows = [], []
                        for idx in loc:  # Determining the power at a certain radius
                            r = ((idx[0] - MidMapX) ** 2 + (idx[1] - MidMapY) ** 2) ** (1 / 2)
                            AC_pow = AC[idx[0], idx[1]].real
                            radius.append(r)
                            AC_pows.append(AC_pow)
                        radius, AC_pows = zip(*sorted(list(zip(radius, AC_pows)), key=op.itemgetter(0)))
                        radius = np.array(radius)
                        AC_pows = np.array(AC_pows)

                        """
                        fitting the first N=num points related to being at or withing a distance=DIST.
                        This is the source of the parameters:
                        m
                        b
                        and their associated errors.
                        """

                        num = len(radius[np.where(radius <= DIST)])
                        opt_fit_AC, cov_mat_AC = curve_fit(f, radius[1:num], AC_pows[1:num])
                        err = np.sqrt(np.diag(cov_mat_AC))

                        M = opt_fit_AC[0]
                        M_err = err[0]
                        B = opt_fit_AC[1]
                        B_err = err[1]

                        data[region][wavelength]['linear']['m'][date] = M
                        data[region][wavelength]['linear']['m_err'][date] = M_err
                        data[region][wavelength]['linear']['b'][date] = B
                        data[region][wavelength]['linear']['b_err'][date] = B_err


                        """
                        Now we get into the Beam Convolution portion.
                    
                        I use a brent-dekker numerical method implemented by scipy to find the sigma required to 
                        convolve an epoch to the desired final length scale as determined at the beginning of this
                        file in the REGIONS dictionary.
                        """
                        FINAL_LENGTH_SCALE = float(REGIONS[region][wavelength])
                        Length_Scale = np.sqrt(SIGX*SIGY)
                        print('Original Length Scale: {: 0.2f}'.format(Length_Scale))

                        if (Length_Scale > FINAL_LENGTH_SCALE) or ((FINAL_LENGTH_SCALE - Length_Scale) <= tol):
                            # if the length scale is within tol or above the final length scale it is ignored.
                            Sigma_Opt = 0
                            N = 0
                            data[region][wavelength]['AC_New']['sigma'][date] = Sigma_Opt
                            data[region][wavelength]['AC_New']['N'][date] = N
                            data[region][wavelength]['AC_New']['amp'][date] = AMP
                            data[region][wavelength]['AC_New']['amp_err'][date] = AMP_ERR
                            data[region][wavelength]['AC_New']['sig_x'][date] = SIGX
                            data[region][wavelength]['AC_New']['sig_x_err'][date] = SIGX_ERR
                            data[region][wavelength]['AC_New']['sig_y'][date] = SIGY
                            data[region][wavelength]['AC_New']['sig_y_err'][date] = SIGY_ERR
                            data[region][wavelength]['AC_New']['theta'][date] = THETA
                            data[region][wavelength]['AC_New']['theta_err'][date] = THETA_ERR
                            FLS = Length_Scale
                            SUCCESS = True
                        else:
                            # if the difference in length scale is below the tolerance it is put to the test!
                            # IDK who brent is, but they seem like a great person to get to the bottom of things.
                            # (minimization puns anyone?)
                            x_size = y_size = PS.shape[0]
                            x, y = np.meshgrid(np.arange(x_size), np.arange(y_size))
                            Sigma_Opt, _, N, _ = brent(beam_fit,
                                                       args=(PS, FINAL_LENGTH_SCALE),
                                                       tol=tol,
                                                       brack=(0.01, 0.5 * FINAL_LENGTH_SCALE),
                                                       full_output=True)
                            Sigma_Opt = abs(Sigma_Opt)
                            if N < 15:
                                SUCCESS = True
                            print('Brent: Sigma {: 0.2f} N {:d}'.format(Sigma_Opt, N))

                            Sol_G2d = fourier_gaussian_function(x, y, sigma_x=Sigma_Opt, sigma_y=Sigma_Opt)
                            try:
                                AC = amp(fftshift(ifft2(Sol_G2d * Sol_G2d * PS)))  # amplitude
                                [
                                    [AMP, OptSigX, OptSigY, THETA],
                                    [AMP_ERR, SIGX_ERR, SIGY_ERR, THETA_ERR]
                                ], OptModel = gaussian_fit_ac(AC)

                                FLS = np.sqrt(OptSigX * OptSigY)

                                data[region][wavelength]['pass_fail'] = SUCCESS
                                data[region][wavelength]['AC_New']['sigma'][date] = Sigma_Opt
                                data[region][wavelength]['AC_New']['N'][date] = N
                                data[region][wavelength]['AC_New']['amp'][date] = AMP
                                data[region][wavelength]['AC_New']['amp_err'][date] = AMP_ERR
                                data[region][wavelength]['AC_New']['sig_x'][date] = OptSigX
                                data[region][wavelength]['AC_New']['sig_x_err'][date] = SIGX_ERR
                                data[region][wavelength]['AC_New']['sig_y'][date] = OptSigY
                                data[region][wavelength]['AC_New']['sig_y_err'][date] = SIGY_ERR
                                data[region][wavelength]['AC_New']['theta'][date] = THETA
                                data[region][wavelength]['AC_New']['theta_err'][date] = THETA_ERR
                            except ValueError:
                                data[region][wavelength]['AC_New']['amp'][date] = -1
                                data[region][wavelength]['AC_New']['amp_err'][date] = -1
                                data[region][wavelength]['AC_New']['sig_x'][date] = -1
                                data[region][wavelength]['AC_New']['sig_x_err'][date] = -1
                                data[region][wavelength]['AC_New']['sig_y'][date] = -1
                                data[region][wavelength]['AC_New']['sig_y_err'][date] = -1
                                data[region][wavelength]['AC_New']['theta'][date] = -1
                                data[region][wavelength]['AC_New']['theta_err'][date] = -1
                        radius, AC_pows = [], []
                        for idx in loc:  # Determining the power at a certain radius
                            r = ((idx[0] - MidMapX) ** 2 + (idx[1] - MidMapY) ** 2) ** (1 / 2)
                            AC_pow = AC[idx[0], idx[1]].real
                            radius.append(r)
                            AC_pows.append(AC_pow)
                        radius, AC_pows = zip(*sorted(list(zip(radius, AC_pows)), key=op.itemgetter(0)))
                        radius = np.array(radius)
                        AC_pows = np.array(AC_pows)
                        num = len(radius[np.where(radius <= DIST)])
                        opt_fit_AC, cov_mat_AC = curve_fit(f, radius[1:num], AC_pows[1:num])
                        err = np.sqrt(np.diag(cov_mat_AC))

                        M = opt_fit_AC[0]
                        M_err = err[0]
                        B = opt_fit_AC[1]
                        B_err = err[1]

                        data[region][wavelength]['linear_new']['m'][date] = M
                        data[region][wavelength]['linear_new']['m_err'][date] = M_err
                        data[region][wavelength]['linear_new']['b'][date] = B
                        data[region][wavelength]['linear_new']['b_err'][date] = B_err

                        print('Final Length Scale: {:0.2f}'.format(FLS))

                        if SUCCESS:
                            pass
                        elif Length_Scale > FINAL_LENGTH_SCALE:
                            pass
                        else:
                            LOG_OUT = TX.format(str(wavelength), str(date), str(SUCCESS), str(FLS),
                                                str(FINAL_LENGTH_SCALE))
                            LOG.write(LOG_OUT)
    TIME_END = time.time()
    TIME_TOTAL = TIME_END - TIME_START
    head = 'Time to run:'
    print("\n" + head + "\n" + '=' * len(head))
    print('{:d} min : {:d} sec'.format(int(TIME_TOTAL // 60), int(TIME_TOTAL % 60)))  # time to run in minutes
    print()


# Saving data to a pickle file so I don't have to run this everytime, only when there is new data. 
data = default_to_regular(data)
with open('/home/cobr/Documents/jcmt-variability/data/data.pickle', 'wb') as OUT:
    pickle.dump(data, OUT)
