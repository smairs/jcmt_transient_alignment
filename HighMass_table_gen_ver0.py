import pickle
import numpy as np

REGIONS = {"DR21C":   {"450":1,
                       "850":1},
           "DR21N":   {"450":1,
                       "850":1},
           "DR21S":   {"450":1,
                       "850":1},
           "M17":     {"450":1,
                       "850":1},
           "M17SWex": {"450":1,
                       "850":1},
           "S255":    {"450":1,
                       "850":1}
           }

for region in REGIONS.keys():

    with open("/data/data_HM.pickle", 'rb') as data:
        data = pickle.load(data)
    data=data[region]
    px2fwhm = 2.355 / np.sqrt(2)

    JD = np.array(list(data['450']['header']['julian_date'].values()),dtype=str).T
    key = np.array(list(data['450']['header']['airmass'].keys()), dtype=str).T
    airmass = np.array(list(data['450']['header']['airmass'].values()), dtype=float).T
    tau225 = np.array(list(data['450']['header']['t225'].values()), dtype=float).T

    elev = np.array(list(data['450']['header']['elevation'].values()), dtype=float).T

    M_850 = np.array(list(data['850']['linear']['m'].values()), dtype=float).T
    M_err_850 = np.array(list(data['850']['linear']['m_err'].values()), dtype=float).T

    B_850 = np.array(list(data['850']['linear']['b'].values()), dtype=float).T
    B_err_850 = np.array(list(data['850']['linear']['b_err'].values()), dtype=float).T

    AMP_850 = np.array(list(data['850']['AC']['amp'].values()), dtype=float).T
    AMP_err_850 = np.array(list(data['850']['AC']['amp_err'].values()), dtype=float).T
    SIGX_850 = np.array(list(data['850']['AC']['sig_x'].values()), dtype=float).T
    SIGX_fwhm_850 = SIGX_850 * px2fwhm
    SIGX_err_850 = np.array(list(data['850']['AC']['sig_x_err'].values()), dtype=float).T
    SIGY_850 = np.array(list(data['850']['AC']['sig_y'].values()), dtype=float).T
    SIGY_fwhm_850 = SIGY_850 * px2fwhm
    SIGY_err_850 = np.array(list(data['850']['AC']['sig_y_err'].values()), dtype=float).T
    THETA_850 = np.array(list(data['850']['AC']['theta'].values()), dtype=float).T
    THETA_err_850 = np.array(list(data['850']['AC']['theta_err'].values()), dtype=float).T

    M_450 = np.array(list(data['450']['linear']['m'].values()), dtype=float).T
    M_err_450 = np.array(list(data['450']['linear']['m_err'].values()), dtype=float).T

    B_450 = np.array(list(data['450']['linear']['b'].values()), dtype=float).T
    B_err_450 = np.array(list(data['450']['linear']['b_err'].values()), dtype=float).T

    AMP_450 = np.array(list(data['450']['AC']['amp'].values()), dtype=float).T
    AMP_err_450 = np.array(list(data['450']['AC']['amp_err'].values()), dtype=float).T
    SIGX_450 = np.array(list(data['450']['AC']['sig_x'].values()), dtype=float).T
    SIGX_fwhm_450 = SIGX_450 * px2fwhm
    SIGX_err_450 = np.array(list(data['450']['AC']['sig_x_err'].values()), dtype=float).T
    SIGY_450 = np.array(list(data['450']['AC']['sig_y'].values()), dtype=float).T
    SIGY_fwhm_450 = SIGY_450 * px2fwhm
    SIGY_err_450 = np.array(list(data['450']['AC']['sig_y_err'].values()), dtype=float).T
    THETA_450 = np.array(list(data['450']['AC']['theta'].values()), dtype=float).T
    THETA_err_450 = np.array(list(data['450']['AC']['theta_err'].values()), dtype=float).T

    dx = np.array(list(data['850']['XC']['alignment'].values()), dtype=float).T[0] * 3
    dy = np.array(list(data['850']['XC']['alignment'].values()), dtype=float).T[1] * 3
    dx450 = np.array(list(data['450']['XC']['alignment'].values()), dtype=float).T[0] * 2
    dy450 = np.array(list(data['450']['XC']['alignment'].values()), dtype=float).T[1] * 2
    ddx = dx - dx450
    ddy = dy - dy450

    cal_measure_850 = np.sqrt(-M_850)
    cal_frac_linear_850 = cal_measure_850/cal_measure_850.mean()

    beam_linear_850 = np.sqrt(-B_850/M_850)
    beam_linear_fwhm_850 = beam_linear_850 * px2fwhm
    beam_gaussian_850 = np.sqrt(SIGY_850*SIGX_850)
    beam_gaussian_fwhm_850 = beam_gaussian_850 * px2fwhm
    cal_frac_gauss_850 = np.sqrt(AMP_850/AMP_850.mean())/(beam_gaussian_850/beam_gaussian_850.mean())

    M_450[M_450 > 0] = -0.0001

    cal_measure_450 = np.sqrt(-M_450)
    cal_frac_linear_450 = cal_measure_450/cal_measure_450.mean()

    beam_linear_450 = np.sqrt(-B_450/M_450) / np.sqrt(2)
    beam_linear_fwhm_450 = beam_linear_450 * px2fwhm
    beam_gaussian_450 = np.sqrt(SIGY_450*SIGX_450)
    beam_gaussian_fwhm_450 = beam_gaussian_450 * px2fwhm
    cal_frac_gauss_450 = np.sqrt(AMP_450/AMP_450.mean())/(beam_gaussian_450/beam_gaussian_450.mean())

    hdr = "Key JulianDate ELEV Airmass Tau225 " \
          "CalF_lin CalF_gauss Beam_lin Beam_gauss " \
          "CalF_lin_450 CalF_gauss_450 Beam_lin_450 Beam_gauss_450 " \
          "Beam_lin_fwhm Beam_gauss_fwhm Sigx_fwhm Sigy_fwhm " \
          "Beam_lin_fwhm_450 Beam_gauss_fwhm_450 Sigx_fwhm_450 Sigy_fwhm_450 " \
          "dx dy " \
          "dx_450 dy_450 " \
          "ddx ddy " \
          "B B_err M M_err " \
          "B_450 B_err_450 M_450 M_err_450 " \
          "Amp Amp_err Sigx Sigx_err Sigy Sigy_err Theta Theta_err " \
          "Amp_450 Amp_err_450 Sigx_450 Sigx_err_450 Sigy_450 Sigy_err_450 Theta_450 Theta_err_450 "

    arr = [key, JD, elev, airmass, tau225,
           cal_frac_linear_850, cal_frac_gauss_850, beam_linear_850, beam_gaussian_850,
           cal_frac_linear_450, cal_frac_gauss_450, beam_linear_450, beam_gaussian_450,
           beam_linear_fwhm_850, beam_gaussian_fwhm_850, SIGX_fwhm_850, SIGY_fwhm_850,
           beam_linear_fwhm_450, beam_gaussian_fwhm_450, SIGX_fwhm_450, SIGY_fwhm_450,
           dx, dy, dx450, dy450, ddx, ddy,
           B_850, B_err_850, M_850, M_err_850,
           B_450, B_err_450, M_450, M_err_450,
           AMP_850, AMP_err_850, SIGX_850, SIGX_err_850, SIGY_850, SIGY_err_850, THETA_850, THETA_err_850,
           AMP_450, AMP_err_450, SIGX_450, SIGX_err_450, SIGY_450, SIGY_err_450, THETA_450, THETA_err_450,
           ]

    table = np.array(np.zeros(len(key)))
    for val in arr:
        table = np.vstack((np.array(table), np.array(val, dtype=str)))
    np.savetxt("/home/cobr/Documents/jcmt-variability/tables/HM_{:}.table".format(region), np.array(table)[1:].T, fmt="%s", header=hdr)
