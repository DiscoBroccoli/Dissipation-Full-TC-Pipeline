from scipy.interpolate import interp1d
import numpy as np
from datasetTC import DATASET_DIR, Instance_number
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate


class DissipationDataset:
    def __init__(self):
        path = DATASET_DIR / '180_D_label.xlsx'
        self.X_label = self._load_dataset(path)

    def _load_dataset(self, path) -> pd.DataFrame:
        print(f'Loading following dataset: {path}.')
        df = pd.read_excel(path, engine='openpyxl')
        return df


D = DissipationDataset()
dicts = {}

x_i = np.linspace(0, 2, Instance_number)

dataframe = D.X_label
dataframe = dataframe.drop(['y/d'], axis=1)

# creating dictionnary
for i in dataframe:
    fspline = interp1d(D.X_label['y/d'], D.X_label[i])
    dicts[i]=fspline(x_i)

dicts['y/d_i'] = x_i

# converting the dict to a dataframe
output_df = pd.DataFrame(data=dicts).sort_values(by=['y/d_i'])
output_df.to_excel(r'.\interpolated_label_180.xlsx', index = False)

# ______________________________________________________
plt.title('Dissipation Interpolation Validation- $\overline{\\rho u_i}$')
vel = dataframe['Diss']

fig = plt.gcf()

plt.plot(D.X_label['y/d'], vel, '-', color="blue")
plt.plot(dicts['y/d_i'], dicts['Diss'], '--', color="orange")


plt.legend(['$\\epsilon$', 'interpolated'], loc='best')
plt.show()
fig.savefig(r'./Image/dissipation_180.png', dpi=500)