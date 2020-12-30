import pandas as pd


class Remove_firstlast:
    
    def __init__(self):
        
        reynolds = [180, 285, 395, 450, 590]

        for Re in reynolds:
            path = f'{Re}_D_features.xlsx'
            self.feature = self._remove_fl(path)
            self.feature = self.feature.to_excel(f'{Re}_D_features_case.xlsx', index=False)

        for Re in reynolds:
            path = f'{Re}_D_label.xlsx'
            self.label = self._remove_fl(path)
            self.label = self.label.to_excel(f'{Re}_D_label_case.xlsx', index=False)

    def _remove_fl(self, path) -> pd.DataFrame:
        print(f'Loading following dataset: {path}.')
        df = pd.read_excel(path)
        # df = df[df['y/d'] > 0]
        # df = df[df['y/d'] < 1]
        return df


R = Remove_firstlast()
# calling to apply the _remove_fl on each dataset
r = R.feature
rr = R.label

"""
Load, process and clean the data for the Production dataset

"""


import pandas as pd

# filenames
excel_D = ["180_D_features_case.xlsx", "285_D_features_case.xlsx", "D_features_inter_case.xlsx", "450_D_features_case.xlsx", "590_D_features_case.xlsx"]
excel_DL = ["180_D_label_case.xlsx", "285_D_label_case.xlsx", "DL_label_inter_case.xlsx", "450_D_label_case.xlsx", "590_D_label_case.xlsx"]

# read them in
excels_D = [pd.ExcelFile(name) for name in excel_D]
excels_DL = [pd.ExcelFile(name) for name in excel_DL]

# turn them into dataframes
frames_D = [x.parse(x.sheet_names[0], header=None,index_col=None) for x in excels_D]
frames_DL = [x.parse(x.sheet_names[0], header=None,index_col=None) for x in excels_DL]

# delete the first row for all frames except the first
# i.e. remove the header row -- assumes it's the first
frames_D[1:] = [df[1:] for df in frames_D[1:]]
frames_DL[1:] = [df[1:] for df in frames_DL[1:]]

# concatenate them..
combined_D = pd.concat(frames_D)
combined_DL = pd.concat(frames_DL)

# write it out
combined_D.to_excel("D_features_case.xlsx", header=False, index=False)
combined_DL.to_excel("DL_label_case.xlsx", header=False, index=False)


