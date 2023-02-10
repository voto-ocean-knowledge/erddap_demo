from pathlib import Path
import sys
library_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(library_dir))
import utils

ds_list = utils.find_glider_datasets()


def test_datasets_list():
    assert len(ds_list) > 100


def test_get_meta():
    meta = utils.get_meta(ds_list[0])
    assert meta


def test_download_datasets():
    ds_id = ds_list[0]
    ds_dict = utils.download_glider_dataset([ds_id])
    ds = ds_dict[ds_id]
    assert len(ds["time"]) > 1000
