from datasets import load_dataset


def get_levircd(path):
    return load_dataset("ericyu/LEVIRCD_Cropped256", cache_dir=path)


def get_sysu(path):
    return load_dataset("ericyu/SYSU_CD", cache_dir=path)


def get_clcd(path):
    return load_dataset("ericyu/CLCD_Cropped_256", cache_dir=path)


def get_gvlm(path):
    return load_dataset("ericyu/GVLM_Cropped_256", cache_dir=path)


def get_oscd96(path):
    return load_dataset("blaz-r/OSCD_RGB_Cropped_96", cache_dir=path)
