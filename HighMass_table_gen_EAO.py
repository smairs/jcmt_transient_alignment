import pickle
import numpy as np
import os

def make_table(regions=['DR21C'],alignment_iteration=0,wavelength='850')
    """
    :param regions: a list of regions to run
    :param alignment_iteration: there will be multiple iterations of the alignment run - this 0-based integer designates which alignment iteration the output table describes
    """
    
    REGIONS = {}
    for i in regions:
        REGIONS[i] = {wavelength:1}
    
    for region in REGIONS.keys():
    
        with open("data/data_Transient_run_"+str(alignment_iteration)+"_"+wavelength+".pickle", 'rb') as data:
            data = pickle.load(data)
        data=data[region]
        px2fwhm = 2.355 / np.sqrt(2)
    
        keyarray = np.array(list(data[wavelength]['header']['airmass'].keys()))
    
        JD = np.array(list(data[wavelength]['header']['julian_date'].values()),dtype=str)[np.argsort(keyarray)].T
        #key = np.array(list(data[wavelength]['header']['airmass'].keys()), dtype=str)[np.argsort(keyarray)].T
        airmass = np.array(list(data[wavelength]['header']['airmass'].values()), dtype=float)[np.argsort(keyarray)].T
        tau225 = np.array(list(data[wavelength]['header']['t225'].values()), dtype=float)[np.argsort(keyarray)].T
    
        elev = np.array(list(data[wavelength]['header']['elevation'].values()), dtype=float)[np.argsort(keyarray)].T
   
        if wavelength  == '850': 
            M_850 = np.array(list(data['850']['linear']['m'].values()), dtype=float)[np.argsort(keyarray)].T
            M_err_850 = np.array(list(data['850']['linear']['m_err'].values()), dtype=float)[np.argsort(keyarray)].T
    
            B_850 = np.array(list(data['850']['linear']['b'].values()), dtype=float)[np.argsort(keyarray)].T
            B_err_850 = np.array(list(data['850']['linear']['b_err'].values()), dtype=float)[np.argsort(keyarray)].T
    
            AMP_850 = np.array(list(data['850']['AC']['amp'].values()), dtype=float)[np.argsort(keyarray)].T
            AMP_err_850 = np.array(list(data['850']['AC']['amp_err'].values()), dtype=float)[np.argsort(keyarray)].T
            SIGX_850 = np.array(list(data['850']['AC']['sig_x'].values()), dtype=float)[np.argsort(keyarray)].T
            SIGX_fwhm_850 = SIGX_850 * px2fwhm
            SIGX_err_850 = np.array(list(data['850']['AC']['sig_x_err'].values()), dtype=float)[np.argsort(keyarray)].T
            SIGY_850 = np.array(list(data['850']['AC']['sig_y'].values()), dtype=float)[np.argsort(keyarray)].T
            SIGY_fwhm_850 = SIGY_850 * px2fwhm
            SIGY_err_850 = np.array(list(data['850']['AC']['sig_y_err'].values()), dtype=float)[np.argsort(keyarray)].T
            THETA_850 = np.array(list(data['850']['AC']['theta'].values()), dtype=float)[np.argsort(keyarray)].T
            THETA_err_850 = np.array(list(data['850']['AC']['theta_err'].values()), dtype=float)[np.argsort(keyarray)].T
            dx = np.array(list(data['850']['XC']['alignment'].values()), dtype=float)[np.argsort(keyarray)].T[0][0] * 3
            dy = np.array(list(data['850']['XC']['alignment'].values()), dtype=float)[np.argsort(keyarray)].T[0][1] * 3
            key = np.array(sorted(list(data[wavelength]['header']['airmass'].keys())), dtype=str).T

            cal_measure_850 = np.sqrt(-M_850)
            cal_frac_linear_850 = cal_measure_850/cal_measure_850.mean()
    
            beam_linear_850 = np.sqrt(-B_850/M_850)
            beam_linear_fwhm_850 = beam_linear_850 * px2fwhm
            beam_gaussian_850 = np.sqrt(SIGY_850*SIGX_850)
            beam_gaussian_fwhm_850 = beam_gaussian_850 * px2fwhm
            cal_frac_gauss_850 = np.sqrt(AMP_850/AMP_850.mean())/(beam_gaussian_850/beam_gaussian_850.mean())
            
            hdr = "Key JulianDate ELEV Airmass Tau225 " \
                  "CalF_lin CalF_gauss Beam_lin Beam_gauss " \
                  "Beam_lin_fwhm Beam_gauss_fwhm Sigx_fwhm Sigy_fwhm " \
                  "dx dy " \
                  "B B_err M M_err " \
                  "Amp Amp_err Sigx Sigx_err Sigy Sigy_err Theta Theta_err "
    
            arr = [key, JD, elev, airmass, tau225,
                   cal_frac_linear_850, cal_frac_gauss_850, beam_linear_850, beam_gaussian_850,
                   beam_linear_fwhm_850, beam_gaussian_fwhm_850, SIGX_fwhm_850, SIGY_fwhm_850,
                   dx, dy,
                   B_850, B_err_850, M_850, M_err_850,
                   AMP_850, AMP_err_850, SIGX_850, SIGX_err_850, SIGY_850, SIGY_err_850, THETA_850, THETA_err_850,
                   ]

    
        elif wavelength == '450':
            M_450 = np.array(list(data['450']['linear']['m'].values()), dtype=float)[np.argsort(keyarray)].T
            M_err_450 = np.array(list(data['450']['linear']['m_err'].values()), dtype=float)[np.argsort(keyarray)].T
    
            B_450 = np.array(list(data['450']['linear']['b'].values()), dtype=float)[np.argsort(keyarray)].T
            B_err_450 = np.array(list(data['450']['linear']['b_err'].values()), dtype=float)[np.argsort(keyarray)].T
    
            AMP_450 = np.array(list(data['450']['AC']['amp'].values()), dtype=float)[np.argsort(keyarray)].T
            AMP_err_450 = np.array(list(data['450']['AC']['amp_err'].values()), dtype=float)[np.argsort(keyarray)].T
            SIGX_450 = np.array(list(data['450']['AC']['sig_x'].values()), dtype=float)[np.argsort(keyarray)].T
            SIGX_fwhm_450 = SIGX_450 * px2fwhm
            SIGX_err_450 = np.array(list(data['450']['AC']['sig_x_err'].values()), dtype=float)[np.argsort(keyarray)].T
            SIGY_450 = np.array(list(data['450']['AC']['sig_y'].values()), dtype=float)[np.argsort(keyarray)].T
            SIGY_fwhm_450 = SIGY_450 * px2fwhm
            SIGY_err_450 = np.array(list(data['450']['AC']['sig_y_err'].values()), dtype=float)[np.argsort(keyarray)].T
            THETA_450 = np.array(list(data['450']['AC']['theta'].values()), dtype=float)[np.argsort(keyarray)].T
            THETA_err_450 = np.array(list(data['450']['AC']['theta_err'].values()), dtype=float)[np.argsort(keyarray)].T
            dx450 = np.array(list(data['450']['XC']['alignment'].values()), dtype=float)[np.argsort(keyarray)].T[0][0] * 2
            dy450 = np.array(list(data['450']['XC']['alignment'].values()), dtype=float)[np.argsort(keyarray)].T[0][1] * 2
            key = np.array(sorted(list(data[wavelength]['header']['airmass'].keys())), dtype=str).T

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
                  "Beam_lin_fwhm Beam_gauss_fwhm Sigx_fwhm Sigy_fwhm " \
                  "dx dy " \
                  "B B_err M M_err " \
                  "Amp Amp_err Sigx Sigx_err Sigy Sigy_err Theta Theta_err "

            arr = [key, JD, elev, airmass, tau225,
                   cal_frac_linear_450, cal_frac_gauss_450, beam_linear_450, beam_gaussian_450,
                   beam_linear_fwhm_450, beam_gaussian_fwhm_450, SIGX_fwhm_450, SIGY_fwhm_450,
                   dx450, dy450,
                   B_450, B_err_450, M_450, M_err_450,
                   AMP_450, AMP_err_450, SIGX_450, SIGX_err_450, SIGY_450, SIGY_err_450, THETA_450, THETA_err_450,
                   ]

        table = np.array(np.zeros(len(key)))
        if not os.path.exists('tables'):
            os.system('mkdir tables')
        for val in arr:
            table = np.vstack((np.array(table), np.array(val, dtype=str)))
        np.savetxt("tables/Transient_"+region+"_run_"+str(alignment_iteration)+"_"+wavelength+".table", np.array(table)[1:].T, fmt="%s", header=hdr)
