import numpy as np
import seaborn as sns
from scipy import odr
import pickle


sns.set_style('whitegrid')
sns.color_palette('colorblind')

# + ===================== +
# | Functions used        |
# + ===================== +


def f_linear(p, x):
    """
    :param x: independent variable
    :param p: fitting parameters
    :return: y: a linear monomial
    """

    y = p[0] * x + p[1]
    return y


# + ===================== +
# | Root project location |
# + ===================== +
LOCAL_ROOT = '/home/cobr/Documents/JCMT/jcmt-trans-align/'
# external disk root location:
ROOT = '/media/cobr/JCMT-TRANSIENT/'

# + ===================== +
# | Global parameters     |
# + ===================== +
RADIUS = 7  # the distance used for linear fitting and gaussian fitting (use width = RADIUS*2 + 1)
length = 200  # The size we clip the reference matrix to. size MxM = length*2 x length*2
TEST = False


REGIONS = ['IC348', 'NGC1333', 'NGC2024', 'NGC2071', 'OMC23', 'OPH_CORE', 'SERPENS_MAIN', 'SERPENS_SOUTH']
with open("/home/cobr/Documents/JCMT/jcmt-var/data/data.pickle", 'rb') as DATA:
    DATA = pickle.load(DATA)

for region in REGIONS:
    data = DATA[region]
    Dates850 = []
    Dates450 = []

    MetaData850 = np.loadtxt(ROOT + region + '/A3_images_cal/' + region + '_850_EA3_cal_metadata.txt', dtype=str)
    MetaData450 = np.loadtxt(ROOT + region + '/A3_images_cal_450/' + region + '_450_EA3_cal_metadata.txt', dtype=str)

    FN850 = MetaData850.T[1]  # filename of the 850 metadata files (ordered)
    FN450 = MetaData450.T[1]  # filename of the 450 metadata files (ordered)

    Dates850.extend([''.join(d[1:].split('-')) for d in MetaData850.T[2]])  # the dates of all the 850 metadata files
    Dates450.extend([''.join(d[1:].split('-')) for d in MetaData450.T[2]])  # the dates of all the 450 metadata files

    # + ===================================== +
    # | Calibration factor via linear fitting |
    # + ===================================== +
    """
    This had to be split into two parts:

    1. For 850um all epochs have cal_f values
    attributed to them

    2. For 450um only some epochs have been
    calibrated, need to know which ones.
    """
    model = odr.Model(f_linear)

    x450 = []
    x450_err = []
    i = 0
    for date450 in data['450']['dates']:
        if str(date450[:8]) in Dates450:
            x450.append(np.sqrt(np.abs(data['450']['linear']['m'][date450])))
            x450_err.append(0.5 * data['450']['linear']['m_err'][date450] / x450[i])
            i += 1
    cal_f_450 = np.array(MetaData450.T[10], dtype=float)
    cal_f_err_450 = np.array(MetaData450.T[11], dtype=float)
    DATE = MetaData450.T[2]
    BAD_EPOCH = np.where(DATE == "\"2019-04-18")
    if region == "SERPENS_SOUTH":
        cal_f_450 = np.delete(cal_f_450, BAD_EPOCH)
        cal_f_err_450 = np.delete(cal_f_err_450, BAD_EPOCH)
    data450 = odr.RealData(x450, cal_f_450, sx=x450_err, sy=cal_f_err_450)
    odr450 = odr.ODR(data450, model, beta0=[1, 1])
    out450 = odr450.run()
    opt450 = out450.beta
    err450 = out450.sd_beta

    x850 = np.sqrt(-1 * np.array(list(data['850']['linear']['m'].values())))
    x850_err = 0.5 * np.array(list(data['850']['linear']['m_err'].values())) / x850
    cal_f_850 = np.array(MetaData850.T[10], dtype=float)
    cal_f_err_850 = np.array(MetaData850.T[11], dtype=float)

    data850 = odr.RealData(x850, cal_f_850, sx=x850_err, sy=cal_f_err_850)
    odr850 = odr.ODR(data850, model, beta0=[1, 1])
    out850 = odr850.run()
    opt850 = out850.beta
    err850 = out850.sd_beta

    hdr = 'KEY MDate MDate450 File_Name JD Elev T225 RMS RMS_450 ' \
          'Steve_offset_x Steve_offset_y ' \
          'Cal_f Cal_f_err ' \
          'Cal_f_450 Cal_f_err_450 ' \
          'AC_cal AC_cal_err ' \
          'AC_cal_450 AC_cal_err_450 ' \
          'JCMT_Offset_x JCMT_Offset_y ' \
          'JCMT_Offset_x_450 JCMT_Offset_y_450 ' \
          'XC_off_x XC_off_x_err ' \
          'XC_off_y XC_off_y_err ' \
          'XC_off_x_450 XC_off_x_err_450 ' \
          'XC_off_y_450 XC_off_y_err_450 ' \
          'B B_err ' \
          'M M_err ' \
          'B_450 B_err_450 ' \
          'M_450 M_err_450 ' \
          'AC_amp AC_amp_err ' \
          'AC_sig_x AC_sig_x_err ' \
          'AC_sig_y AC_sig_y_err ' \
          'AC_theta AC_theta_err ' \
          'AC_amp_450 AC_amp_err_450 ' \
          'AC_sig_x_450 AC_sig_x_err_450 ' \
          'AC_sig_y_450 AC_sig_y_err_450 ' \
          'AC_theta_450 AC_theta_err_450 ' \
          'dx dy dx_450 dy_450 ddx ddy '

    li = np.zeros(len(hdr.split()), dtype=str)  # How many columns are in the header above?
    index450 = 0
    index450_2 = 0
    index850 = 0
    for date450, date in zip(data['450']['dates'], data['850']['dates']):
        AC_cal_f_m = opt850[0]
        AC_cal_f_m_err = err850[0]
        AC_cal_f_b = opt850[1]
        AC_cal_f_b_err = err850[1]

        AC_cal_f_m_450 = opt450[0]
        AC_cal_f_m_err_450 = err450[0]
        AC_cal_f_b_450 = opt450[1]
        AC_cal_f_b_err_450 = err450[1]

        if str(date450[:8]) in Dates450:
            metadate450 = Dates450[index450_2]
            if region == "SERPENS_SOUTH":
                if metadate450 == "20190418":
                    index450_2 += 1
                    metadate450 = Dates450[index450_2]
            rms_450 = str(MetaData450[index450_2][8])  # RMS level
            calf450 = MetaData450[index450_2][10]
            calferr450 = MetaData450[index450_2][11]
            x_450 = x450[index450]
            x_err_450 = x450_err[index450]/x_450
            index450 += 1
            index450_2 += 1
        else:
            metadate450 = -1
            rms_450 = -1
            x_450 = -1
            x_err_450 = -1
            calf450 = -1
            calferr450 = -1

        if str(date[:8]) in Dates850:
            e_num = str(MetaData850[index850][0])  # index850 number
            metadate = Dates850[index850]
            name = str(MetaData850[index850][1][:-4])  # name of index850
            jd = str(MetaData850[index850][4])  # julian date
            elev = str(MetaData850[index850][6])  # elevation
            t225 = str(MetaData850[index850][7])  # tau-225
            rms = str(MetaData850[index850][8])  # RMS level3
            steve_offset_x = str(MetaData850[index850][-2])
            steve_offset_y = str(MetaData850[index850][-1])
            cal_f = str(MetaData850[index850][10])  # calibration factor from Steve
            cal_f_err = str(MetaData850[index850][11])  # error in calibration factor from Steve
            x = x850[index850]
            x_err = x850_err[index850]/x
            index850 += 1
        else:
            e_num = str(-1)  # index850 number
            metadate = str(-1)
            name = str(-1)  # name of index850
            jd = str(-1)  # julian date
            elev = str(-1)  # elevation
            t225 = str(-1)  # tau-225
            rms = str(-1)  # RMS level
            steve_offset_x = str(-1)
            steve_offset_y = str(-1)
            cal_f = str(-1)  # calibration factor from Steve
            cal_f_err = str(-1)  # error in calibration factor from Steve
            x = -1
            x_err = -1

        jcoffx450 = data['450']['JCMT_offset'][date][0]
        jcoffy450 = data['450']['JCMT_offset'][date][1]

        jcoffx850 = data['850']['JCMT_offset'][date][0]
        jcoffy850 = data['850']['JCMT_offset'][date][1]

        xcoffx450 = data['450']['XC']['offset'][date][0]
        xcoffx450_err = data['450']['XC']['offset_err'][date][0]

        xcoffx850 = data['850']['XC']['offset'][date][0]
        xcoffx850_err = data['850']['XC']['offset_err'][date][0]

        xcoffy450 = data['450']['XC']['offset'][date][1]
        xcoffy450_err = data['450']['XC']['offset_err'][date][1]

        xcoffy850 = data['850']['XC']['offset'][date][1]
        xcoffy850_err = data['850']['XC']['offset_err'][date][1]

        acamp450 = data['450']['AC']['amp'][date]
        acamp450_err = data['450']['AC']['amp_err'][date]

        acamp850 = data['850']['AC']['amp'][date]
        acamp850_err = data['850']['AC']['amp_err'][date]

        acsigx450 = data['450']['AC']['sig_x'][date]
        acsigx450_err = data['450']['AC']['sig_x_err'][date]

        acsigx850 = data['850']['AC']['sig_x'][date]
        acsigx850_err = data['850']['AC']['sig_x_err'][date]

        acsigy450 = data['450']['AC']['sig_y'][date]
        acsigy450_err = data['450']['AC']['sig_y_err'][date]

        acsigy850 = data['850']['AC']['sig_y'][date]
        acsigy850_err = data['850']['AC']['sig_y_err'][date]

        actheta450 = data['450']['AC']['theta'][date]
        actheta450_err = data['450']['AC']['theta_err'][date]

        actheta850 = data['850']['AC']['theta'][date]
        actheta850_err = data['850']['AC']['theta_err'][date]

        b450 = data['450']['linear']['b'][date]
        b450_err = data['450']['linear']['b_err'][date]

        b850 = data['850']['linear']['b'][date]
        b850_err = data['850']['linear']['b_err'][date]

        m450 = data['450']['linear']['m'][date]
        m450_err = data['450']['linear']['m_err'][date]

        m850 = data['850']['linear']['m'][date]
        m850_err = data['850']['linear']['m_err'][date]

        # 450 micron
        dx_450 = (jcoffx450 - xcoffx450) * 2
        dy_450 = (jcoffy450 - xcoffy450) * 2

        # 850 micron
        dx = (jcoffx850 - xcoffx850) * 3
        dy = (jcoffy850 - xcoffy850) * 3

        ddx = dx - dx_450
        ddy = dy - dy_450

        if m850 < 0:
            AC_CAL_F = AC_cal_f_m * np.sqrt(-m850) + AC_cal_f_b
            AC_CAL_F_err = 0.5 * m850_err / np.sqrt(-m850)
        else:
            AC_CAL_F = -1
            AC_CAL_F_err = -1

        if m450 < 0:
            AC_CAL_F_450 = AC_cal_f_m_450 * np.sqrt(-m450) + AC_cal_f_b_450
            AC_CAL_F_err_450 = 0.5 * m450_err / np.sqrt(-m450)
        else:
            AC_CAL_F_450 = -1
            AC_CAL_F_err_450 = -1

        P = np.array(
            [date, metadate, metadate450, name, jd, elev, t225, rms, rms_450,
             steve_offset_x, steve_offset_y,
             cal_f, cal_f_err, calf450, calferr450, AC_CAL_F, AC_CAL_F_err, AC_CAL_F_450, AC_CAL_F_err_450,
             jcoffx850, jcoffy850, jcoffx450, jcoffy450,
             xcoffx850, xcoffx850_err, xcoffy850, xcoffy850_err, xcoffx450, xcoffx450_err, xcoffy450, xcoffy450_err,
             b850, b850_err, m850, m850_err, b450, b450_err, m450, m450_err,
             acamp850, acamp850_err, acsigx850, acsigx850_err, acsigy850, acsigy850_err, actheta850, actheta850_err,
             acamp450, acamp450_err, acsigx450, acsigx450_err, acsigy450, acsigy450_err, actheta450, actheta450_err,
             dx, dy, dx_450, dy_450, ddx, ddy],
            dtype=str)
        li = np.vstack((li, P))

    form = '%s'
    np.savetxt(LOCAL_ROOT + 'tables/' + region + '.table',
               li[1:],
               fmt=form,
               header=hdr
               )
