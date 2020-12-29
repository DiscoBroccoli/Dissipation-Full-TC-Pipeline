from scipy.interpolate import interp1d
import numpy as np
from datasetTC import DATASET_DIR, Instance_number
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate


class DissipationDataset:
    def __init__(self):
        path = DATASET_DIR / '590_D_features.xlsx'
        self.X_feature = self._load_dataset(path)

    def _load_dataset(self, path) -> pd.DataFrame:
        print(f'Loading following dataset: {path}.')
        df = pd.read_excel(path, engine='openpyxl')
        return df


D = DissipationDataset()
dicts = {}

x_i = np.linspace(0, 2, Instance_number)

dataframe = D.X_feature
dataframe = dataframe.drop(['y/d'], axis=1)

# creating dictionnary
for i in dataframe:
    fspline = interp1d(D.X_feature['y/d'], D.X_feature[i])
    dicts[i]=fspline(x_i)

dicts['y/d_i'] = x_i

# converting the dict to a dataframe
output_df = pd.DataFrame(data=dicts).sort_values(by=['y/d_i'])
output_df.to_excel(r'.\interpolated_f_590.xlsx', index = False)
# ______________________________________________________
plt.title('Velocities Interpolation Validation- $\overline{\\rho u_i}$')
vel = dataframe[['u', 'v', 'w']]

fig = plt.gcf()
for i, fname in enumerate(vel):
    plt.plot(D.X_feature['y/d'], vel[fname], '-', color="blue")
    plt.plot(dicts['y/d_i'], dicts[fname], '--', color="orange")


plt.legend(['$\overline{\\rho u_i}$',
            'interpolated'], loc='best')
plt.show()
fig.savefig(r'./Image/velocities_590.png', dpi=500)
# ______________________________________________________

plt.title('Density Gradients Interpolation Validation - \
$\\frac{\partial \overline{\\rho}}{\partial x_i}$')
density_grad = dataframe[['drho_x', 'drho_y', 'drho_z']]

fig = plt.gcf()
for i, fname in enumerate(density_grad):
    plt.plot(D.X_feature['y/d'], density_grad[fname], '-', color="blue")
    plt.plot(dicts['y/d_i'], dicts[fname], '--', color="orange")

plt.legend(['$\\frac{\partial \overline{\\rho}}{\partial x_i}$',
            'interpolated'], loc='best')
plt.show()
fig.savefig(r'./Image/density_grad_590.png', dpi=500)
# ______________________________________________________

plt.title('Velocity Gradients Interpolation Validation - \
$\\frac{\partial \overline{\\rho u}}{\partial x_i}$')
vel_grad = dataframe[['drhou_x', 'drhou_y', 'drhou_z',]]


fig = plt.gcf()
for i, fname in enumerate(vel_grad):
    plt.plot(D.X_feature['y/d'], vel_grad[fname], '-', color="blue")
    plt.plot(dicts['y/d_i'], dicts[fname], '--', color="orange")

plt.legend(['$\\frac{\partial \overline{\\rho u}}{\partial y}$',
            'interpolated'], loc='best')
plt.show()
fig.savefig(r'./Image/vg_u_590.png', dpi=500)
# ______________________________________________________

plt.title('Velocity Gradients Interpolation Validation - \
$\\frac{\partial \overline{\\rho v}}{\partial x_i}$')
vel_grad = dataframe[['drhov_x', 'drhov_y', 'drhov_z',]]


fig = plt.gcf()
for i, fname in enumerate(vel_grad):
    plt.plot(D.X_feature['y/d'], vel_grad[fname], '-', color="blue")
    plt.plot(dicts['y/d_i'], dicts[fname], '--', color="orange")

plt.legend(['$\\frac{\partial \overline{\\rho v}}{\partial y}$',
            'interpolated'], loc='best')
plt.show()
fig.savefig(r'./Image/vg_v_590.png', dpi=500)
# ______________________________________________________

plt.title('Velocity Gradients Interpolation Validation - \
$\\frac{\partial \overline{\\rho w}}{\partial x_i}$')
vel_grad = dataframe[['drhow_x', 'drhow_y', 'drhow_z',]]


fig = plt.gcf()
for i, fname in enumerate(vel_grad):
    plt.plot(D.X_feature['y/d'], vel_grad[fname], '-', color="blue")
    plt.plot(dicts['y/d_i'], dicts[fname], '--', color="orange")

plt.legend(['$\\frac{\partial \overline{\\rho w}}{\partial y}$',
            'interpolated'], loc='best')
plt.show()
fig.savefig(r'./Image/vg_w_590.png', dpi=500)
# ______________________________________________________

plt.title('Total Energy Interpolation Validation - $\overline{\\rho E}$')
total_favre_energy = dataframe[['rho_E']]

fig = plt.gcf()
for i, fname in enumerate(total_favre_energy):
    plt.plot(D.X_feature['y/d'], total_favre_energy[fname], '-', color="blue")
    plt.plot(dicts['y/d_i'], dicts[fname], '--', color="orange")

plt.legend(['$\overline{\\rho E}$', 'interpolated'], loc='best')
plt.show()
fig.savefig(r'./Image/total_E_590.png', dpi=500)


'''
class ProductionDataset:
    def __init__(self):
        path = DATASET_DIR / '285_P_label.xlsx'
        self.X_feature = self._load_dataset(path)

    def _load_dataset(self, path) -> pd.DataFrame:
        print(f'Loading following dataset: {path}.')
        df = pd.read_excel(path)
        return df


P = ProductionDataset()

x = D.X_feature['yplus']
y = D.X_feature['Prod']
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')

xnew = np.linspace(0, 569, num=41, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()
'''




