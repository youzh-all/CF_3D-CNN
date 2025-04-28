#%%
import numpy as np
from matplotlib import pyplot as plt
import os

def save(arr: np.ndarray, name: str, path: str, vmin: float = None, vmax: float = None):
    os.makedirs(path, exist_ok=True)
    if vmin is None or vmax is None:
        plt.imshow(arr, cmap='jet')
    else:
        plt.imshow(arr, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(f"{path}/{name}.png")
    plt.clf()
    plt.close()
    np.savetxt(f"{path}/{name}.csv", arr, delimiter=',')

def process_folder(folder_name: str):
    open_path = folder_name
    save_path = f"output/{folder_name}"
    os.makedirs(save_path, exist_ok=True)

    data = {
        'f0': np.loadtxt(f"{open_path}/F0.csv", delimiter=',', dtype=np.float64),
        'fm': np.loadtxt(f"{open_path}/Fm.csv", delimiter=',', dtype=np.float64),
        'fm_p': [np.loadtxt(f"{open_path}/fm_p_{i}.csv", delimiter=',', dtype=np.float64) for i in range(5)],
        'ft_p': [np.loadtxt(f"{open_path}/ft_p_{i}.csv", delimiter=',', dtype=np.float64) for i in range(5)]
    }

    fv = data['fm'] - data['f0']
    fv[fv < 0] = 0
    fvfm = fv / data['fm']
    save(fvfm, "fvfm", save_path)

    F0_L1, F0_Lss = None, None
    for i, fm_p_i in enumerate(data['fm_p']):
        F0_Ln = data['f0'] / ((data['fm'] - data['f0']) / data['fm'] + data['f0'] / fm_p_i)
        F0_Ln[F0_Ln < 0] = 0
        F0_Ln[F0_Ln > 1] = 1
        if i != 4:
            save(F0_Ln, f"F0_L{i+1}", save_path)
            if i == 0:
                F0_L1 = F0_Ln
        else:
            save(F0_Ln, "F0_Lss", save_path)
            F0_Lss = F0_Ln

    for i, fm_p_i in enumerate(data['fm_p']):
        FvFm_Ln = (fm_p_i - F0_Lss) / fm_p_i
        FvFm_Ln[FvFm_Ln < 0] = 0
        FvFm_Ln[FvFm_Ln > 1] = 1
        save(FvFm_Ln, f"FvFm_L{i+1}" if i != 4 else "FvFm_Lss", save_path)

    for i, fm_p_i in enumerate(data['fm_p']):
        NPQ_Ln = (data['fm'] - fm_p_i) / fm_p_i
        NPQ_Ln[NPQ_Ln > 1] = 1
        save(NPQ_Ln, f"NPQ_L{i+1}" if i != 4 else "NPQ_Lss", save_path)

    for i, (fm_p_i, ft_p_i) in enumerate(zip(data['fm_p'], data['ft_p'])):
        Y_Ln = (fm_p_i - ft_p_i) / fm_p_i
        Y_Ln[Y_Ln < 0] = 0
        Y_Ln[Y_Ln > 1] = 1
        save(Y_Ln, f"Y_L{i+1}" if i != 4 else "Y_Lss", save_path)

    for i, (fm_p_i, ft_p_i) in enumerate(zip(data['fm_p'], data['ft_p'])):
        qP_Ln = (fm_p_i - ft_p_i) / (fm_p_i - F0_L1)
        qP_Ln[qP_Ln < 0] = 0
        qP_Ln[qP_Ln > 1] = 1
        save(qP_Ln, f"qP_L{i+1}" if i != 4 else "qP_Lss", save_path)

def main():
    base_folder = "CF_raw"
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

    for subfolder in subfolders:
        print(f"Processing folder: {subfolder}")
        process_folder(subfolder)

if __name__ == "__main__":
    main()

# %%
