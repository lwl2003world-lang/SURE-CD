from pathlib import Path
import timeit
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]


BANDNAME_2_BANDS = {
    "oscd96_ms-rgb": [BANDS.index(b) for b in ["B04", "B03", "B02"]],
    "oscd96_ms-10": [
        BANDS.index(b)
        for b in ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    ],
    "oscd96_ms_BEN1": [
        BANDS.index(b)
        for b in ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"]
    ],
    "oscd96_ms_BEN2": [
        BANDS.index(b)
        for b in ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    ],
    "oscd96_ms-all": [
        BANDS.index(b)
        for b in [
            "B01",
            "B04",
            "B03",
            "B02",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B10",
            "B11",
            "B12",
        ]
    ],
    "s2-13": [
        BANDS.index(b)
        for b in [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B10",
            "B11",
            "B12",
        ]
    ],
    "oscd96_ms-satlas": [
        BANDS.index(b)
        for b in ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12"]
    ],
    "oscd96_ms-satlasREV": [
        BANDS.index(b)
        for b in ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]
    ],
    "s1_vv": [0],
    "deo": [
        BANDS.index(b)
        for b in ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    ],
}


class CDDataset(Dataset):
    """Change detection dataset loader."""

    def __init__(
        self,
        data: str | list[str],
        transform,
        bands,
        use_hf=True,
        load_in_mem="direct",
    ):
        self.transform = transform
        self.use_hf = use_hf
        self.load_in_mem = load_in_mem
        self.bands = bands

        if use_hf:
            print("Using HF data")
            self.data = data
        else:
            if load_in_mem == "direct":
                self.data = self.load_from_dir(data)
            elif load_in_mem == "hdf5":
                if isinstance(data, list):
                    print("Loading multiple datasets")
                    arr = []
                    for d in data:
                        arr.extend(self.load_from_hdf5(d))
                    self.data = arr
                else:
                    self.data = self.load_from_hdf5(data)
                    print(data, "->", len(self.data))
            else:
                self.data = self.load_paths_from_dir(data)

    def load_paths_from_dir(self, path):
        path = Path(path)
        file_list = list(Path(path).glob("A/*.png"))
        if len(file_list) == 0:
            raise ValueError(f"No images found in {path}")

        file_dict_list = []
        for img_path in tqdm(
            file_list, desc="Loading paths from dir.", total=len(file_list)
        ):
            fname = Path(img_path).name
            t_el = {
                "imageA": img_path,
                "imageB": path / "B" / fname,
                "label": path / "label" / fname,
                "img_idx": img_path.stem,
            }
            file_dict_list.append(t_el)

        return file_dict_list

    def load_from_dir(self, path):
        path = Path(path)
        file_list = list(Path(path).glob("A/*"))
        if len(file_list) == 0:
            raise ValueError(f"No images found in {path}")

        data_list = []
        for img_path in tqdm(
            file_list, desc="Loading images into memory.", total=len(file_list)
        ):
            fname = Path(img_path).name
            t_el = {
                "imageA": np.array(Image.open(img_path)),
                "imageB": np.array(Image.open(path / "B" / fname)),
                "label": np.array(Image.open(path / "label" / f"{img_path.stem}.png")),
                "img_idx": img_path.stem,
            }
            data_list.append(t_el)

        return data_list

    def load_imgs_from_path(self, path_dict):
        return {
            "imageA": np.array(Image.open(path_dict["imageA"])),
            "imageB": np.array(Image.open(path_dict["imageB"])),
            "label": np.array(Image.open(path_dict["label"])),
            "img_idx": path_dict["img_idx"],
        }

    def load_from_hdf5(self, path):
        d_name = Path(path).parts[-2]
        split = Path(path).parts[-1]
        print(f"Loading from HDF5 {d_name} {split}")
        t1 = timeit.default_timer()
        file = h5py.File(str(path) + ".h5", "r")

        imageA = np.array(file["/imageA"])
        imageB = np.array(file["/imageB"])

        reindex = self.bands not in ["rgb"]
        if reindex:
            band_idx = BANDNAME_2_BANDS[self.bands]
            imageA = imageA[..., band_idx]
            imageB = imageB[..., band_idx]

        label = np.array(file["/label"])
        img_idx = np.array(file["/img_idx"])
        file.close()

        t2 = timeit.default_timer()
        print(f"Done in {t2 - t1:.2f} seconds")

        ret_list = []
        for a, b, lbl, idx in zip(imageA, imageB, label, img_idx):
            ret_list.append(
                {
                    "imageA": a,
                    "imageB": b,
                    "label": lbl,
                    "img_idx": idx,
                    "d_name": d_name,
                }
            )

        return ret_list

    def __getitem__(self, index):
        data = self.data[index]

        if self.use_hf:
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = np.copy(v)
        elif not self.load_in_mem:
            data = self.load_imgs_from_path(data)

        out_dict = {}
        transformed = self.transform(data)

        out_dict["imageA"] = transformed["imageA"]
        out_dict["imageB"] = transformed["imageB"]
        out_dict["label"] = transformed["label"]

        out_dict["imageA_unnorm"] = data["imageA"]
        out_dict["imageB_unnorm"] = data["imageB"]
        out_dict["img_idx"] = data["img_idx"]
        if "d_name" in data:
            out_dict["d_name"] = data["d_name"]

        return out_dict

    def __len__(self):
        return len(self.data)
