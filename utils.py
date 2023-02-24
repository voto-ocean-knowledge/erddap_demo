import datetime
import numpy as np
import pathlib
import xarray as xr
import pandas as pd
from erddapy import ERDDAP
from tqdm.notebook import tqdm
from argopy import DataFetcher as ArgoDataFetcher

cache_dir = pathlib.Path('voto_erddap_data_cache')


def init_erddap(protocol="tabledap"):
    # Setup initial ERDDAP connection
    e = ERDDAP(
        server="https://erddap.observations.voiceoftheocean.org/erddap",
        protocol=protocol,
    )
    return e


def _clean_dims(ds):
    if "timeseries" in ds.dims.keys() and "obs" in ds.dims.keys():
        ds = ds.drop_dims("timeseries")
    if "obs" in list(ds.dims):
        ds = ds.swap_dims({"obs": "time"})
    return ds


def find_glider_datasets(nrt_only=True):
    """
    Find the dataset IDs of all glider datasets on the VOTO ERDDAP server
    nrt_only: if True, only returns nrt datasets
    """
    e = init_erddap()

    # Fetch dataset list
    e.response = "csv"
    e.dataset_id = "allDatasets"
    df_datasets = e.to_pandas()

    datasets = df_datasets.datasetID
    # Select only nrt datasets
    if nrt_only:
        datasets = datasets[datasets.str[:3] == "nrt"]
    return datasets.values


def _get_meta_griddap(dataset_id):
    e = init_erddap(protocol="griddap")
    e.dataset_id = dataset_id
    e.griddap_initialize()
    time = pd.read_csv(f"https://erddap.observations.voiceoftheocean.org/erddap/griddap/{dataset_id}.csvp?time")[
        "time (UTC)"].values
    e.constraints['time>='] = str(time[-20])
    ds = e.to_xarray()
    attrs = ds.attrs
    # Clean up formatting of variables list
    if "variables" in attrs.keys():
        if "\n" in attrs["variables"]:
            attrs["variables"] = attrs["variables"].split("\n")
    # evaluate dictionaries
    for key, val in attrs.items():
        if type(val) == str:
            if "{" in val:
                attrs[key] = eval(val)
    return attrs


def get_meta(dataset_id, protocol="tabledap"):
    if "adcp" in dataset_id or protocol=="griddap":
        # Cannot use to_ncCF with griddap
        return _get_meta_griddap(dataset_id)
    e = init_erddap(protocol=protocol)
    e.dataset_id = dataset_id
    meta = e.to_ncCF()
    attrs = {}
    for key_name in dir(meta):
        if key_name[0] != "_":
            attrs[key_name] = meta.__getattribute__(key_name)
    # Clean up formatting of variables list
    if "variables" in attrs.keys():
        if type(attrs["variables"]) is dict:
            attrs["variables"] = list(attrs["variables"].keys())
    # evaluate dictionaries
    for key, val in attrs.items():
        if type(val) == str:
            if "{" in val:
                attrs[key] = eval(val)
    return attrs


def add_profile_time(ds):
    profile_num = ds.pressure.copy()
    profile_num.attrs = {}
    profile_num.name = "profile_num"
    profile_num[:] = 0
    start = 0
    for i, prof_index in enumerate(ds.profile_index):
        rowsize = ds.rowSize.values[i]
        profile_num[start:start + rowsize] = prof_index
        start = start + rowsize
    ds["profile_num"] = profile_num
    profile_time = ds.time.values.copy()
    profile_index = ds.profile_num
    for profile in np.unique(profile_index.values):
        mean_time = ds.time[profile_index == profile].mean().values
        new_times = np.empty((len(ds.time[profile_index == profile])), dtype='datetime64[ns]')
        new_times[:] = mean_time
        profile_time[profile_index == profile] = new_times
    profile_time_var = ds.time.copy()
    profile_time_var.values = profile_time
    profile_time_var.name = "profile_mean_time"
    ds["profile_mean_time"] = profile_time_var
    ds = _clean_dims(ds)
    return ds


def _cached_dataset_exists(ds_id, request):
    """
    Returns True if all the following conditions are met:
    1. A dataset corresponding to ds_id exists in the cache
    2. The cached dataset was downloaded with the same request
    3. The dataset has not been updated on the VOTO ERDDAP since it was last downloaded
    Otherwise, returns False
    """
    if not cache_dir.exists():
        print(f"Creating directory to cache datasets at {cache_dir.absolute()}")
        pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return False
    dataset_nc = cache_dir / f"{ds_id}.nc"
    if not dataset_nc.exists():
        print(f"Dataset {ds_id} not found in cache")
        return False
    try:
        df = pd.read_csv(cache_dir / "cache_info.csv", index_col=0)
    except:
        print(f"no cache records file found")
        return False

    if ds_id in df.index:
        stats = df.loc[ds_id]
    else:
        print(f"no cache record found for {ds_id}")
        return False
    if not stats["request"] == request:
        print(f"request has changed for {ds_id}")
        return False

    nc_time = pd.to_datetime(stats["date_created"])
    meta = get_meta(ds_id)
    erddap_time = pd.to_datetime(meta["date_created"])
    if nc_time < erddap_time - datetime.timedelta(seconds=60):
        print(f"Dataset {ds_id} has been updated on ERDDAP")
        return False
    
    return True


def _update_stats(ds_id, request):
    """
    Update the stats for a specified dataset
    """
    dataset_nc = cache_dir / f"{ds_id}.nc"
    ds = xr.open_dataset(dataset_nc)
    try:
        df = pd.read_csv(cache_dir / "cache_info.csv", index_col=0)
    except:
        df = pd.DataFrame()

    nc_time = ds.attrs["date_created"]
    new_stats = {"request": request, "date_created": pd.to_datetime(nc_time)}
    if ds_id in df.index:
        df.loc[ds_id] = new_stats
    else:
        new_row = pd.DataFrame(new_stats, index=[ds_id])
        df = pd.concat((df, new_row))
    df = df.sort_index()
    df.to_csv(cache_dir / "cache_info.csv")
    ds.close()


def add_adcp_data(ds):
    dataset_id = ds.attrs["dataset_id"]
    parts = dataset_id.split("_")
    adcp_id = f"adcp_{parts[1]}_{parts[2]}"
    cached_ds = _cached_dataset_exists(adcp_id, "adcp")
    dataset_nc = cache_dir / f"{adcp_id}.nc"
    if cached_ds:
        print(f"Found {dataset_nc}. Loading from disk")
        adcp = xr.open_dataset(dataset_nc)
    else:
        dataset_ids = find_glider_datasets(nrt_only=False)
        if adcp_id not in dataset_ids:
            print(f"Requested ADCP dataset {adcp_id} does not exist on server! Returning standard dataset")
            return ds
        print(f"Downloading {adcp_id}")
        e = ERDDAP(server="https://erddap.observations.voiceoftheocean.org/erddap/", protocol="griddap", )
        e.dataset_id = adcp_id
        e.griddap_initialize()
        time = pd.read_csv(f"https://erddap.observations.voiceoftheocean.org/erddap/griddap/{adcp_id}.csvp?time")[
            "time (UTC)"].values
        e.constraints['time>='] = str(time[0])
        adcp = e.to_xarray()
        adcp.to_netcdf(dataset_nc)
        _update_stats(adcp_id, "adcp")
    ds = _clean_dims(ds)

    if parts[0] == "nrt":
        print("WARNING: matching adcp data to nearest nrt timestamp. Potential missmatch of ~ 15 seconds. "
              "Use delayed mode data for closer timestamp match")
        adcp = adcp.reindex(time=ds.time, method="nearest")
    for var_name in list(adcp):
        ds[{var_name}] = adcp[var_name]
    adcp_attrs_dict = {i: j for i, j in adcp.attrs.items() if i not in ds.attrs}
    ds.attrs["adcp_attributes"] = str(adcp_attrs_dict)
    return ds


def download_glider_dataset(dataset_ids, variables=(), constraints={}, nrt_only=False, delayed_only=False,
                            cache_datasets=True, adcp=False):
    """
    Download datasets from the VOTO server using a supplied list of dataset IDs.
    dataset_ids: list of datasetIDs present on the VOTO ERDDAP
    variables: data variables to download. If left empty, will download all variables
    """
    if nrt_only and delayed_only:
        raise ValueError("Cannot set both nrt_only and delayed_only")
    if nrt_only:
        ids_to_download = []
        for name in dataset_ids:
            if "nrt" in name:
                ids_to_download.append(name)
            else:
                print(f"{name} is not nrt. Ignoring")
    elif delayed_only:
        ids_to_download = []
        for name in dataset_ids:
            if "delayed" in name:
                ids_to_download.append(name)
            else:
                print(f"{name} is not delayed. Ignoring")
    else:
        ids_to_download = dataset_ids

    e = init_erddap()
    # Specify variables of interest if supplied
    if variables:
        e.variables = variables
    if constraints:
        e.constraints = constraints

    # Download each dataset as xarray
    glider_datasets = {}
    for ds_name in tqdm(ids_to_download):
        if cache_datasets and "delayed" in ds_name:
            e.dataset_id = ds_name
            request = e.get_download_url()
            cached_dataset = _cached_dataset_exists(ds_name, request)
            dataset_nc = cache_dir / f"{ds_name}.nc"
            if cached_dataset:
                print(f"Found {ds_name} in {cache_dir}. Loading from disk")
                ds = xr.open_dataset(dataset_nc)
                if adcp:
                    ds = add_adcp_data(ds)
                glider_datasets[ds_name] = ds
            else:
                print(f"Downloading {ds_name}")
                try:
                    ds = e.to_xarray()
                except:
                    print(f"No matching data for {ds_name}")
                    continue
                ds = _clean_dims(ds)
                print(f"Writing {dataset_nc}")
                ds.to_netcdf(dataset_nc)
                if adcp:
                    ds = add_adcp_data(ds)
                glider_datasets[ds_name] = ds
                _update_stats(ds_name, request)
        else:
            print(f"Downloading {ds_name}")
            e.dataset_id = ds_name
            try:
                ds = e.to_xarray()
            except:
                print(f"No matching data for {ds_name}")
                continue
            ds = _clean_dims(ds)
            if adcp:
                ds = add_adcp_data(ds)
            glider_datasets[ds_name] = ds
    return glider_datasets


def format_difference(deg_e, deg_n, ns_ahead):
    """
    Pretty formatting for a lon, lat, time difference between two points
    """
    km_n = (111 * deg_n).round(1)
    km_e = (111 * deg_e * np.cos(np.deg2rad(deg_n))).round(1)
    h_ahead = (np.float64(ns_ahead) / (1e9 * 60 * 60)).round(1)
    if km_n > 0:
        north_str = f"{km_n} km N"
    else:
        north_str = f"{-km_n} km S"
    if km_e > 0:
        east_str = f"{km_e} km E"
    else:
        east_str = f"{-km_e} km W"
    if h_ahead > 0:
        time_str = f"{h_ahead} hours later"
    else:
        time_str = f"{-h_ahead} hours earlier"
    return east_str, north_str, time_str


def smhi_profiles_in_range(station_visit_df, lon, lat, time, lon_window, lat_window, time_window, min_depth=80):
    """
    Returns the station IDs of stations within a certain range of a point in space and time
    """
    min_lon = lon - lon_window
    max_lon = lon + lon_window
    min_lat = lat - lat_window
    max_lat = lat + lat_window
    min_time = time - time_window
    max_time = time + time_window
    lon_filter = np.logical_and(station_visit_df['sample_longitude_dd'] > min_lon,
                                station_visit_df['sample_longitude_dd'] < max_lon)
    lat_filter = np.logical_and(station_visit_df['sample_latitude_dd'] > min_lat,
                                station_visit_df['sample_latitude_dd'] < max_lat)
    time_filter = np.logical_and(station_visit_df['visit_date'] > min_time, station_visit_df['visit_date'] < max_time)
    df_in_range = station_visit_df[lon_filter & lat_filter & time_filter]
    # Filter out shallow stations
    df_in_range = df_in_range[df_in_range['water_depth_m'] > min_depth]
    if df_in_range.empty:
        return None

    closest_arg = np.argmin(np.abs(df_in_range['visit_date'] - time))
    closest_datasetid = df_in_range.index[closest_arg]
    return closest_datasetid


def nearest_smhi_station(df, ds_glider, lat_window=0.5, lon_window=1, time_window=np.timedelta64(10, "D")):
    """
    Finds the nearest SMHI station profile to a supplied glidermission. Uses sharkweb data file
    """
    station_visit_df = df.groupby('station_visit').first()
    mean_lon = ds_glider.longitude.mean().values
    mean_lat = ds_glider.latitude.mean().values
    mean_time = ds_glider.time.mean().values
    nearest_profile = smhi_profiles_in_range(station_visit_df, mean_lon, mean_lat, mean_time, lat_window, lon_window,
                                             time_window)
    if nearest_profile:
        closest_station = station_visit_df[station_visit_df.index == nearest_profile]
        deg_e = mean_lon - closest_station['sample_longitude_dd'].values[0]
        deg_n = mean_lat - closest_station['sample_latitude_dd'].values[0]
        time_diff = mean_time - closest_station['visit_date'].values[0]
        east_diff, north_diff, time_diff = format_difference(deg_e, deg_n, time_diff)
        loc_str = f"Nearest station profile is {east_diff}, {north_diff} & {time_diff} than mean of glider data"
        print(loc_str)
        df_nearest = df[df.station_visit == nearest_profile]
        return df_nearest
    else:
        print("No SMHI profiles found within tolerances")
        return None


def nearest_argo_profile(ds_glider, lat_window=0.5, lon_window=1, time_window=np.timedelta64(7, "D")):
    """
    Finds the nearest argo profile to a supplied glidermission. Uses ifremer ERDDAP
    """
    mean_lon = ds_glider.longitude.mean().values
    mean_lat = ds_glider.latitude.mean().values
    mean_time = ds_glider.time.mean().values
    max_pressure = ds_glider.pressure.values.max()
    min_time = str(mean_time - time_window)[:10]
    max_time = str(mean_time + time_window)[:10]
    search_region = [mean_lon - lon_window, mean_lon + lon_window,
                     mean_lat - lat_window, mean_lat + lat_window,
                     0, max_pressure,
                     min_time, max_time]
    try:
        ds = ArgoDataFetcher(src='erddap').region(search_region).to_xarray()
        ds2 = ds.argo.point2profile()
        closest_time_index = np.abs(ds2.TIME.values - mean_time).argmin()
        profile = ds2.isel({"N_PROF": closest_time_index})
        deg_n = profile.LATITUDE.values - np.nanmean(ds_glider.latitude)
        deg_e = profile.LONGITUDE.values - np.nanmean(ds_glider.longitude)
        ns_ahead = profile.TIME.values - ds_glider.time.mean()
        east_diff, north_diff, time_diff = format_difference(deg_e, deg_n, ns_ahead)
        loc_str = f"Nearest float is {east_diff}, {north_diff} & {time_diff} than mean of glider data"
        print(loc_str)
        return profile
    except:
        print("No floats found within tolerances")
        return None
