import bz2
import datetime
import json
import logging
import re
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pytz
import requests
import requests.sessions
from flask import Flask, make_response
from flask_json import FlaskJSON
from google.cloud import storage
from metpy.plots import SkewT
from metpy.units import units
from urllib3 import Retry

import skewt.plot.config as config

app = Flask(__name__)
FlaskJSON(app)

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


class WeatherModelSoundingMetaData:
    def __init__(self, model_time, valid_time):
        self.model_time = model_time
        self.valid_time = valid_time


class WeatherModelSounding:
    def __init__(self, latitude, longitude, p, T, QV, Td, U, V, HHL, metadata):
        self.latitude = latitude
        self.longitude = longitude
        self.p = p
        self.T = T
        self.QV = QV
        self.Td = Td
        self.U = U
        self.V = V
        self.HHL = HHL
        self.metadata = metadata

        latitude_pretty = str(abs(round(latitude, 2)))
        if latitude < 0:
            latitude_pretty = latitude_pretty + "S"
        else:
            latitude_pretty = latitude_pretty + "N"

        longitude_pretty = str(abs(round(longitude, 2)))
        if longitude > 0:
            longitude_pretty = longitude_pretty + "E"
        else:
            longitude_pretty = longitude_pretty + "W"

        self.latitude_pretty = latitude_pretty
        self.longitude_pretty = longitude_pretty


class SkewTResult:
    def __init__(self, model_time, valid_time, plot_full, plot_detail):
        self.model_time = model_time
        self.valid_time = valid_time
        self.plot_full = plot_full
        self.plot_detail = plot_detail

    def __json__(self):
        return {
            "model_time": self.model_time,
            "valid_time": self.valid_time,
            "plot_full": self.plot_full,
            "plot_detail": self.plot_detail
        }


def download_bz2(url, target_file, session=requests.sessions.Session()):
    r = session.get(url)
    r.raise_for_status()

    decompressor = bz2.BZ2Decompressor()

    with open(target_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(decompressor.decompress(chunk))


def download_content():
    Path("tmp").mkdir(parents=True, exist_ok=True)
    download_bz2(config.content_log_url, "tmp/content.log")


def latest_run(model, valid_time):
    download_content()

    pattern = re.compile("./%s" \
                         "/grib" \
                         "\\/(\d{2})" \
                         "/t" \
                         "/icon-eu_europe_regular-lat-lon_model-level_(\d{10})_(\d{3})_1_T.grib2.bz2" % model)

    max_t: int = 0
    result = None

    for i, line in enumerate(open('tmp/content.log')):
        for match in re.finditer(pattern, line):
            matches = match.groups()

            match_valid_at = datetime.datetime.strptime(matches[1], "%Y%m%d%H")
            match_valid_at = pytz.timezone('UTC').localize(match_valid_at)
            match_valid_at = match_valid_at + datetime.timedelta(hours=int(matches[2]))

            delta_t = abs((match_valid_at - valid_time).total_seconds())

            if delta_t <= 30 * 60 and int(matches[1]) > max_t:
                result = matches
                max_t = int(matches[1])

    return result


class AllLevelData:
    def __init__(self, data, model_time, valid_time):
        self.data = data
        self.model_time = model_time
        self.valid_time = valid_time


def parameter_all_levels(model, run, parameter, latitude, longitude, level_type="model_level", session=requests.Session()):
    run_hour = run[0]
    run_datetime = run[1]
    timestep = int(run[2])
    logging.info(f"Loading sounding for latitude={latitude} longitude={longitude} with "
                 f"run_hour={run_hour} run_datetime={run_datetime} timestep={timestep} "
                 f"level_type={level_type} and parameter={parameter}")

    levels = np.floor(np.linspace(60, 0, config.level_workers)).astype(int).tolist()
    urls = list()

    for i in range(0, len(levels) - 1):
        base = levels[i]
        top = levels[i + 1] + 1
        # example URL:
        # https://nwp-sounding-mw5zsrftba-ew.a.run.app/48.21/16.37/06/2020030406/4/p
        url = f"{config.sounding_api}" \
               f"/{latitude}" \
               f"/{longitude}" \
               f"/{run_hour}" \
               f"/{run_datetime}" \
               f"/{timestep}" \
               f"/{parameter}" \
               f"?level_type={level_type}" \
               f"&base={base}" \
               f"&top={top}"
        urls.append(url)

    result = AllLevelData(data=np.empty(0), model_time=None, valid_time=None)

    with ThreadPoolExecutor(max_workers=config.level_workers) as executor:
        responses = list(executor.map(session.get, urls))

        for response in responses:
            response.raise_for_status()
            json_result = json.loads(response.content)
            result.data = np.append(result.data, np.array(json_result["data"]))

        json_first = json.loads(responses[0].content)
        result.model_time = np.datetime64(json_result["model_time"])
        result.valid_time = np.datetime64(json_result["valid_time"])

    return result


def find_closest_model_level(p, needle):
    return np.abs(p.to("hPa") - needle.to("hPa")).argmin()


def full_level_height(HHL, idx):
    return (HHL[idx] + HHL[idx + 1]) / 2


def load_weather_model_sounding(latitude, longitude, valid_time):
    model = "icon-eu"
    run = latest_run(model, valid_time)

    http_session = session()

    with ThreadPoolExecutor(max_workers=config.parameter_all_levels_workers) as executor:
        p_future = executor.submit(parameter_all_levels, model, run, "p", latitude, longitude, session=http_session)
        T_future = executor.submit(parameter_all_levels, model, run, "T", latitude, longitude, session=http_session)
        QV_future = executor.submit(parameter_all_levels, model, run, "QV", latitude, longitude, session=http_session)
        U_future = executor.submit(parameter_all_levels, model, run, "U", latitude, longitude, session=http_session)
        V_future = executor.submit(parameter_all_levels, model, run, "V", latitude, longitude, session=http_session)
        HHL_future = executor.submit(parameter_all_levels, model, run, "HHL", latitude, longitude, "time_invariant", session=http_session)

    # Pressure Pa
    p_raw = p_future.result()
    p = p_raw.data

    # Temperature K
    T = T_future.result().data

    # Specific Humidty kg/kg
    QV = QV_future.result().data

    # Dewpoint K
    Td = mpcalc.dewpoint_from_specific_humidity(QV * units("kg/kg"), T * units.K, p * units.Pa)

    # Wind m/s
    U = U_future.result().data
    V = V_future.result().data

    # Height above MSL for model level
    HHL = HHL_future.result().data

    meta_data = WeatherModelSoundingMetaData(p_raw.model_time, p_raw.valid_time)

    return WeatherModelSounding(latitude, longitude, p, T, QV, Td, U, V, HHL, meta_data)


def session():
    http_session = requests.Session()
    connection_pool_adapter = requests.adapters.HTTPAdapter(pool_connections=800, pool_maxsize=800)
    http_session.mount('http://', connection_pool_adapter)
    http_session.mount('https://', connection_pool_adapter)
    retry_adapater = Retry(total=config.sounding_api_retries, read=config.sounding_api_retries,
                           connect=config.sounding_api_retries,
                           backoff_factor=0.3, status_forcelist=(500, 502, 504))
    http_session.mount('http://', retry_adapater)
    http_session.mount('https://', retry_adapater)
    return http_session


def plot_skewt_icon(sounding, parcel=None, base=1000, top=100, skew=45):
    model_time = np.datetime_as_string(sounding.metadata.model_time, unit='m')
    valid_time = np.datetime_as_string(sounding.metadata.valid_time, unit='m')

    top_idx = find_closest_model_level(sounding.p * units.Pa, top * units("hPa"))

    fig = plt.figure(figsize=(11, 11), constrained_layout=True)
    skew = SkewT(fig, rotation=skew)

    skew.plot(sounding.p * units.Pa, sounding.T * units.K, 'r')
    skew.plot(sounding.p * units.Pa, sounding.Td, 'b')
    skew.plot_barbs(sounding.p[:top_idx] * units.Pa, sounding.U[:top_idx] * units.mps,
                    sounding.V[:top_idx] * units.mps, plot_units=units.knot, alpha=0.6, xloc=1.13, x_clip_radius=0.3)

    if parcel == "surface-based":
        prof = mpcalc.parcel_profile(sounding.p * units.Pa, sounding.T[0] * units.K, sounding.Td[0]).to('degC')
        skew.plot(sounding.p * units.Pa, prof, 'y', linewidth=2)

    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    skew.plot(sounding.p * units.Pa, np.zeros(len(sounding.p)) * units.degC, "#03d3fc", linewidth=1)
    skew.ax.set_ylim(base, top)

    plt.title(f"Model run: {model_time}Z", loc='left')
    plt.title(f"Valid time: {valid_time}Z", fontweight='bold', loc='right')
    plt.xlabel("Temperature [°C]")
    plt.ylabel("Pressure [hPa]")

    fig.suptitle(f"ICON-EU Model for {sounding.latitude_pretty}, {sounding.longitude_pretty}", fontsize=14)

    ax1 = plt.gca()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = '#333333'
    ax2.set_ylabel('Geometric Altitude [kft]', color=color)  # we already handled the x-label with ax1
    ax2_data = (sounding.p * units.Pa).to('hPa')
    ax2.plot(np.zeros(len(ax2_data)), ax2_data, color=color, alpha=0.0)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')
    ax2.set_ylim((base, top))
    ticks = np.linspace(base, top, num=10)

    ideal_ticks = np.geomspace(base, top, 20)
    real_tick_idxs = [find_closest_model_level(sounding.p * units.Pa, p_level * units("hPa")) for p_level in
                      ideal_ticks]
    ticks = (sounding.p * units.Pa).to("hPa")[real_tick_idxs]
    full_levels = [full_level_height(sounding.HHL, idx) for idx in real_tick_idxs]
    tick_labels = np.around((full_levels * units.m).m_as("kft"), decimals=1)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(tick_labels)
    ax2.minorticks_off()

    return fig


@app.route("/<float:latitude>/<float:longitude>/<valid_at>")
def skewt(latitude, longitude, valid_at):
    valid_at_parsed = datetime.datetime.strptime(valid_at, "%Y%m%d%H")
    valid_at_parsed = pytz.timezone('UTC').localize(valid_at_parsed)

    sounding = load_weather_model_sounding(latitude, longitude, valid_at_parsed)

    model_time = str(np.datetime_as_string(sounding.metadata.model_time))
    valid_time = str(np.datetime_as_string(sounding.metadata.valid_time))

    model_time_for_file_name = str(np.datetime_as_string(sounding.metadata.model_time, unit='m')).replace(":", "_")
    valid_time_for_file_name = str(np.datetime_as_string(sounding.metadata.valid_time, unit='m')).replace(":", "_")

    full_plot = plot_skewt_icon(sounding=sounding, parcel="surface-based")
    full_plot_filename = f"plot_{sounding.latitude_pretty}_{sounding.longitude_pretty}_" \
                         f"{model_time_for_file_name}_{valid_time_for_file_name}_full.png"
    full_plot.savefig(full_plot_filename)

    detail_plot = plot_skewt_icon(sounding=sounding, parcel="surface-based", base=1000, top=500, skew=15)
    detail_plot_filename = f"plot_{sounding.latitude_pretty}_{sounding.longitude_pretty}_" \
                           f"{model_time_for_file_name}_{valid_time_for_file_name}_detail.png"
    detail_plot.savefig(detail_plot_filename)

    # Google Cloud Upload

    storage_client = storage.Client()
    bucket = storage_client.bucket(config.bucket_name)
    blob_full = bucket.blob(full_plot_filename)
    blob_detail = bucket.blob(detail_plot_filename)

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(blob_full.upload_from_filename, full_plot_filename)
        executor.submit(blob_detail.upload_from_filename, detail_plot_filename)

    result = json.dumps(SkewTResult(model_time, valid_time,
                                    config.bucket_public_url + full_plot_filename,
                                    config.bucket_public_url + detail_plot_filename).__dict__)
    response = make_response(result)
    response.mimetype = 'application/json'
    return response


if __name__ == "__main__":
    app.run()
