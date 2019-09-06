"""
Nanoscale group studies 2019
Optical SPR data calibration and analysis
Charles Tam
"""
import os
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.odr.odrpack as odr
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import sympy as sym

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 16})

# parameters
prism_no = 21
t_prism = 29 #degrees
d_sens = 7.1e-2 #metres

delta_1 = 100  # lower offset for sigmoid
delta_2 = 40 # offset from max reflectance for sigmoid
delta_3 = 0 # offset from max reflectance for asymmetric

details = True #show extra details on plot
export = False # export data

s_filename = "redux/21/s1.csv"
p_filename = "redux/21/p1.csv"
bg_filename = "redux/18/b1.csv"
plotname = "output/prism26a.pdf"
txtname = "output/prism26a.csv"

n_prism = 1.7988 # for SF6 at 632.8 nm
prism_diam = 4.775 # mm
prism_height = 2.95 # mm
t_err = 0.5 #degrees
d_sens_err = 1e-3 #metres
w_pix = 14e-6 #metres
n_av = 50

def moving_average(data, n=3):
    """moving average of some data"""
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def asymmetric(beta, x):
    """asymmetric function for fitting"""
    a, b, c, d, e = beta
    return a * (1 - (b + c * (x - d))/((x - d) ** 2 + e ** 2))

def spr_asy(beta):
    """find spr angle analytically"""
    a, b, c, d, e = beta
    return d + (-b + np.sqrt(b**2 + (c*e)**2)/c)

def sigmoid(beta, x):
    """sigmoid function for fitting"""
    a, b, c, d = beta
    return a/(1 + np.exp(b*(x - c))) + d

def sig_asy(beta, x):
    """
    combined asymmetric and sigmoid functions for plotting
    """
    a, b, c, d, e, f, g, h = beta
    return a*(1 - (b + c*(x - d))/((x - d)**2 + e**2)) + (f/(1+np.exp(g*(x - h))))

def gaus(x, a, b, c):
    """gaussisan function for fitting"""
    return a*np.exp(-(x-b)**2/(2*c**2))

def err_angle(x, dx):
    """uncertainty on angle by propogating errors on parameters"""
    y = d_sens
    dy = d_sens_err
    a = w_pix
    dz = t_err
    t = (a*x)**2 + y**2
    err_rad = np.sqrt((a * x * dy / t)**2 + (a * y * dx / t)**2 + np.radians(dz)**2)
    return np.degrees(err_rad)

def err_edge(dbeta, beta, r):
    """uncertainty on angle edge by propogating errors on parameters"""
    da, db, dc = dbeta
    a, _, c = beta
    t = np.log(a/r)
    return 2*np.sqrt((c*da)**2/2*t*a**2 + db**2 + 2*t*dc**2)

def beam_angle(px, dist=d_sens, width=w_pix):
    """return the beam angle as a function of beam width using the known parameters"""
    angle = np.arctan(px * width / dist)
    return np.degrees(angle)

def find_nearest(array, value):
    """nearest values in an array to a given value"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def angle_offset(iterations=5):
    """correction factors for prism geometry"""
    x, y = sym.symbols("x y")
    a = prism_diam/2
    b = prism_height - a
    a1 = []
    a2 = []
    theta1 = np.deg2rad(np.linspace(1, 15, iterations))
    for i in theta1:
        t2 = math.asin(math.sin(i)/n_prism)
        t3 = i - t2
        eq1 = a**2 + x**2 - b**2 - 2*a*x*sym.cos(t2)
        c = np.asarray(sym.solve(eq1, x))[1]
        eq2 = a**2 + y**2 - (b + sym.sqrt(c**2 + y**2 - 2*c*y*sym.cos(t3)))**2 - 2*a*y*sym.cos(i)
        e = sym.re(np.asarray(sym.solve(eq2, y))[0])
        d = math.sqrt(c**2 + e**2 - 2*c*e*math.cos(t3))
        a1.append(np.rad2deg(math.acos((e**2 + d**2 - c**2)/(2*d*e))))
        a2.append(np.rad2deg(math.acos((b**2 + c**2 - a**2)/(2*b*c))))
    return np.polyfit(a1, a2, 1)

def main():
    # make output directory
    if not os.path.exists("output"):
        os.makedirs("output")

    # get data and caluclate reflectance
    n_pix = np.genfromtxt(s_filename, delimiter=",")[:, 0]
    i_s = np.genfromtxt(s_filename, delimiter=",")[:, 1]
    i_p = np.genfromtxt(p_filename, delimiter=",")[:, 1]
    i_bg = np.genfromtxt(bg_filename, delimiter=",")[:, 1]
    ref = (i_p - i_bg) / (i_s - i_bg)

    # find beam edges
    popt, pcov = curve_fit(gaus, n_pix, i_s, p0=[1, 1000, 50])
    i_width = gaus(popt[1], *popt)/np.e**2
    i_max = int(popt[1])
    beam_lim = np.argmax(i_s[:i_max] - i_width > 0), len(n_pix) - np.argmax(i_s[:-i_max :-1] -
                                                                            i_width > 0)
    idx_centre = int(np.diff(beam_lim) / 2) + beam_lim[0] - 1

    # plot raw data and threshold limits
    plt.figure(figsize=(8, 6))
    plt.plot(n_pix, i_s, label="S polarised")
    plt.plot(n_pix, i_p, label="P polarised")
    plt.plot(n_pix, i_bg, label="Background")
    plt.plot(n_pix, gaus(n_pix, *popt), label="Gaussian")
    plt.plot(n_pix[idx_centre], i_width, "kx", ms=8)
    plt.plot((n_pix[beam_lim[1]], n_pix[beam_lim[0]]), (i_s[beam_lim[1]], i_s[beam_lim[0]]), "kx", ms=8)
    plt.text(n_pix[beam_lim[0]]+20, i_width-0.01, r"$n_{edge, low}$", ha="left", va="top")
    plt.text(n_pix[beam_lim[1]]+20, i_width-0.01, r"$n_{edge, up}$", ha="left", va="top")
    plt.text(n_pix[idx_centre]+20, i_width-0.01, r"$n_{centre}$", ha="left", va="top")
    plt.xlabel("Pixel number")
    plt.ylabel("Intensity")
    plt.title("S and P polarised intensities for sample 21")
    plt.legend()
    plt.show()

    # get range of angles and error
    n_pix_shift = np.arange(0, 2048, 1) - idx_centre
    correct = angle_offset(5)
    theta_range = beam_angle(n_pix_shift) + (t_prism*correct[0] + correct[1])
    n_err = err_edge(np.sqrt(np.diagonal(pcov)), popt, i_width)
    theta_err = err_angle(n_pix_shift, n_err)
    err_red = theta_err[beam_lim[0]:beam_lim[1]]
    ref_red_ = ref[beam_lim[0]:beam_lim[1]]
    theta_red = theta_range[beam_lim[0]:beam_lim[1]]
    ref_red = ref_red_/max(ref_red_[500:-500])

    # moving average
    ref_av = moving_average(ref_red, n=n_av)
    theta_av = np.linspace(min(theta_red), max(theta_red), len(ref_av))

    # fit whole reflectivity to asymmetric
    asymm = odr.Model(asymmetric)
    data_full = odr.RealData(theta_red, ref_red, sx=err_red)
    fit_asy_full = odr.ODR(data_full, asymm, beta0=[0.9, 0.4, 0.4, t_prism+6, -0.6], maxit=100).run()
    ref_max = argrelextrema(asymmetric(fit_asy_full.beta, theta_red), np.greater)[0][0]

    # fit right side of reflectivity to asymmetric
    idx_asy = ref_max + delta_3
    data_right = odr.RealData(theta_red[idx_asy:-400], ref_red[idx_asy:-400], sx=err_red[idx_asy:-400])
    fit_asy = odr.ODR(data_right, asymm, beta0=[0.9, 0.4, 0.4, t_prism+6, -0.6], maxit=100).run()
    r_sq_asy = 1 - fit_asy.res_var/np.var(ref_red[idx_asy:], ddof=1)
    data_asy = asymmetric(fit_asy.beta, theta_red)

    # fit left side of reflectivity to sigmoid
    sig = odr.Model(sigmoid)
    idx_sig = ref_max + delta_2
    data_left = odr.RealData(theta_red[delta_1:idx_sig], ref_red[delta_1:idx_sig],
                             sx=err_red[delta_1:idx_sig])
    fit_sig = odr.ODR(data_left, sig, beta0=[0.15, -2.3, t_prism+4, 0.8], maxit=100).run()
    r_sq_sig = 1 - fit_sig.res_var/np.var(ref_red[delta_1:idx_sig], ddof=1)
    data_sig = sigmoid(fit_sig.beta, theta_red)
    data_sig_asy = sig_asy((*fit_asy.beta, *fit_sig.beta[:-1]), theta_red) - fit_sig.beta[0]

    # first derrivative of asymmetric-sigmoid
    a, b, c, d, e, f, g, h, x = sym.symbols("a b c d e f g h x")
    sym_fit = a*(1-(b+c*(x-d))/((x-d)**2+e**2))+(f/(1+sym.exp(g*(x-h))))
    fit = sym.lambdify((a, b, c, d, e, f, g, h, x), sym_fit.diff(x), "numpy")
    d_fit = fit(*(*fit_asy.beta, *fit_sig.beta[:-1]), theta_red)

    # spr, critical angle, min reflectance and FWHM
    idx_spr = argrelextrema(data_sig_asy, np.less)[0][0]
    idx_crit = argrelextrema(d_fit, np.greater)[0][0]
    theta_spr = theta_red[idx_spr]
    spr_err = np.sqrt(err_red[idx_spr]**2)
    theta_crit = theta_red[idx_crit]
    crit_err = np.sqrt(err_red[idx_crit]**2)
    min_ref = data_sig_asy[idx_spr]
    err_min_ref = min_ref*np.linalg.norm((*fit_sig.sd_beta/fit_sig.beta,
                                          *fit_asy.sd_beta/fit_asy.beta))
    fwhm_1 = find_nearest(data_sig_asy[ref_max:idx_spr], data_sig_asy[ref_max]/2) + ref_max
    fwhm_2 = find_nearest(data_sig_asy[idx_spr:], data_sig_asy[ref_max]/2) + idx_spr
    fwhm = np.abs(theta_red[fwhm_2] - theta_red[fwhm_1])
    err_fwhm = np.sqrt(err_red[fwhm_1]**2 + err_red[fwhm_2]**2)

    # plot ratio of s and p intensities, moving average and fitted curves
    if isinstance(prism_no, float):
        prism_no_str = str(int(prism_no))+"a"
    else:
        prism_no_str = prism_no
    plt.figure(figsize=(8, 6))
    plt.plot(theta_red, ref_red, "silver", label="Reflectance data")
    plt.plot(theta_red, data_sig_asy, label="Asymmetric-sigmoid")
    if details:
        plt.plot(theta_av, ref_av, label="Moving average, $n={}$".format(n_av))
        plt.plot(theta_red, data_asy, label="Asymmetric")
        plt.plot(theta_red, data_sig, label="Sigmoid")
        plt.plot(theta_red, d_fit, label="1st derivative of \n asymmetric-sigmoid")
        plt.plot((theta_red[delta_1], theta_red[idx_sig]),
                 (ref_red[delta_1], ref_red[idx_sig]), "s", label="Sigmoid data range")
        plt.plot((theta_red[idx_asy], theta_red[-1]),
                 (ref_red[idx_asy], ref_red[-1]), "s", label="Asymmetric data range")
        plt.plot((theta_red[fwhm_1], theta_red[fwhm_2]),
                 (ref_red[fwhm_1], ref_red[fwhm_2]), "s", label="FWHM limits")
        plt.plot(theta_crit, d_fit[idx_crit], "kx", ms=8)
        plt.plot(theta_spr, data_asy[idx_spr], "kx", ms=8)
        plt.plot(theta_red[ref_max], asymmetric(fit_asy.beta, theta_red)[ref_max], "kx", ms=8)
        plt.text(theta_crit+0.2, d_fit[idx_crit]-0.02, r"$\theta_{crit}$")
        plt.text(theta_spr+0.2, data_asy[idx_spr]-0.02, r"$\theta_{spr}$")
        plt.text(theta_red[ref_max]-0.2, asymmetric(fit_asy.beta, theta_red)[ref_max]-0.08,
                 r"$\theta_{max}$")
    plt.xlim(theta_red[0], theta_red[-1])
    plt.ylim(-0.1, 1.1)
    plt.xlabel(r"Incidence angle /$^\circ$")
    plt.ylabel("Normalised reflectance")
    plt.title("Reflectance plot for sample {}".format(prism_no_str))
    plt.legend(loc=7)
    # formatted string containing the parameters
    text = """Sample {} on {}: \nprism angle {} deg\nsensor distance {} m\nSPR angle = {} +/- {} deg
Critical angle = {} +/- {} deg \nR squared (asymmetric)= {} \nR squared (sigmoid)= {}
Asymmetric parameters: {} \nAsymmetric parameter errors: {} \nSigmoid parameters: {}
Sigmoid parameter errors: {} \nMinimum reflectance {} +/- {} \nFWHM {} +/- {} deg
delta (1, 2, 3): {}, {}, {}
\n""".format(prism_no_str, datetime.datetime.now().strftime("%d/%m/%y %H:%M"), t_prism, d_sens,
             theta_spr, spr_err, theta_crit, crit_err, r_sq_asy, r_sq_sig, fit_asy.beta,
             fit_asy.sd_beta, fit_sig.beta, fit_sig.sd_beta, min_ref, err_min_ref, fwhm, err_fwhm,
             delta_1, delta_2, delta_3)
    if export:
        # save figure
        plt.savefig(plotname, dpi=100, bbox_inches="tight")
        # log parameters
        file = open("log.txt", "w")
        file = open("output/log.txt", "a")
        file.write(text)
        file.close()
        # write data to csv
        np.savetxt(txtname, np.vstack((theta_red, ref_red)).T, delimiter=",", fmt="%f", header="theta,R")
        # write parameters to csv
        arr = [prism_no, theta_spr, spr_err, theta_crit, crit_err, r_sq_asy,
               r_sq_sig, min_ref, err_min_ref, fwhm, err_fwhm]
        with open("output/param.csv", "ab") as f:
            np.savetxt(f, arr, delimiter=",", newline=",", fmt="%f")
            f.write(b'\n')
    plt.show()
    print(text)

if __name__ == "__main__":
    main()
