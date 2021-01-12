import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bkg_x,bkg_y = np.random.multivariate_normal((0,0),[[1,0],[0,1]],10000).T
bkg = pd.DataFrame()
bkg["x"] = bkg_x
bkg["y"] = bkg_y
bkg["target"] = np.ones(len(bkg_x))
bkg.to_csv("toy_bkg.csv",index=False)

sig_x,sig_y = np.random.multivariate_normal((0.5,0.5),[[0.1,0],[0,0.1]],10000).T
sig = pd.DataFrame()
sig["x"] = sig_x
sig["y"] = sig_y
sig["target"] = np.zeros(len(sig_x))
sig.to_csv("toy_sig.csv",index=False)



# plt.hist2d(sig_bkg_x,sig_bkg_y, bins=100, cmap="RdPu")
# plt.set_xscale("log")
# plt.set_yscale("log")
# plt.colorbar()
# plt.show()
