import bz2
import datetime
import json
import logging
import os
import os.path
import re
from pathlib import Path

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pytz
import requests
import xarray as xr
from metpy.plots import SkewT
from metpy.units import units
from flask import Flask, make_response
from flask_json import FlaskJSON, JsonError, json_response, as_json, request

import config

app = Flask(__name__)
FlaskJSON(app)

logging.basicConfig(level=logging.DEBUG)


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


def download_bz2(url, target_file):
    r = requests.get(url)
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


def load_level(model, run_hour, run_datetime, timestep, parameter, level, level_type="model_level"):
    """
    Load grib files for a single level from the local disk or from the OpenData server
    """
    if level_type == "model_level":
        path = f"./{model}" \
               f"/grib" \
               f"/{run_hour}" \
               f"/{parameter.lower()}" \
               f"/icon-eu_europe_regular-lat-lon_model-level_{run_datetime}_{timestep:03d}_{level:d}_{parameter.upper()}.grib2"
    elif level_type == "time_invariant":
        path = f"./{model}" \
               f"/grib" \
               f"/{run_hour}" \
               f"/{parameter.lower()}" \
               f"/icon-eu_europe_regular-lat-lon_time-invariant_{run_datetime}_{level:d}_{parameter.upper()}.grib2"
    else:
        raise AttributeError("Invalid level type")

    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        download_url = config.dwd_base_url + path[2:] + ".bz2"
        print("Download file from: " + download_url)
        download_bz2(download_url, path)

    return xr.open_dataset(path, engine='cfgrib')


def parameter_all_levels(model, run, parameter, latitude, longitude, level_type="model_level"):
    result = np.empty(0)

    run_hour = run[0]
    run_datetime = run[1]
    timestep = int(run[2])

    for level in range(60, 0, -1):
        level_grib = load_level(model, run_hour, run_datetime, timestep, parameter, level, level_type)

        result = np.append(result, level_grib.to_array()[0].interp(latitude=latitude, longitude=longitude).values)

    return result


def find_closest_model_level(p, needle):
    return np.abs(p.to("hPa") - needle.to("hPa")).argmin()


def full_level_height(HHL, idx):
    return (HHL[idx] + HHL[idx + 1]) / 2


def load_weather_model_sounding(latitude, longitude, valid_time):
    model = "icon-eu"
    run = latest_run(model, valid_time)

    # Pressure Pa
    p = parameter_all_levels(model, run, "p", latitude, longitude)

    # Temperature K
    T = parameter_all_levels(model, run, "T", latitude, longitude)

    # Specific Humidty kg/kg
    QV = parameter_all_levels(model, run, "QV", latitude, longitude)

    # Dewpoint K
    Td = mpcalc.dewpoint_from_specific_humidity(QV * units("kg/kg"), T * units.K, p * units.Pa)

    # Wind m/s
    U = parameter_all_levels(model, run, "u", latitude, longitude)
    V = parameter_all_levels(model, run, "v", latitude, longitude)

    # Height above MSL for model level
    HHL = parameter_all_levels(model, run, "hhl", latitude, longitude, "time_invariant")

    lowest_P_level = load_level(model, run[0], run[1], int(run[2]), "P", 60)

    model_time = lowest_P_level.time.values
    valid_time = lowest_P_level.valid_time.values
    meta_data = WeatherModelSoundingMetaData(model_time, valid_time)

    return WeatherModelSounding(latitude, longitude, p, T, QV, Td, U, V, HHL, meta_data)


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


def main():
    latitude = 48.2082
    longitude = 16.3738
    valid_at = b = datetime.datetime(2020, 3, 2, 11).replace(tzinfo=pytz.utc)
    sounding = load_weather_model_sounding(latitude, longitude, valid_at)

    large_plot = plot_skewt_icon(sounding=sounding, parcel="surface-based")
    large_plot.savefig('full_skewt.png')


@app.route("/<float:latitude>/<float:longitude>/<int:run_hour>/<int:run_datetime>/<int:timestep>/<parameter>/<level>")
def nwp_value(latitude, longitude, run_hour, run_datetime, timestep, parameter, level):
    level_type = request.args.get("level_type", "model_level")
    level_grib = load_level(config.model, run_hour, run_datetime, timestep, parameter, level, level_type)
    return level_grib.to_array()[0].interp(latitude=latitude, longitude=longitude).values[0]


@app.route("/<float:latitude>/<float:longitude>/<valid_at>")
def skewt(latitude, longitude, valid_at):
    valid_at_parsed = datetime.datetime.strptime(valid_at, "%Y%m%d%H")
    valid_at_parsed = pytz.timezone('UTC').localize(valid_at_parsed)

    sounding = load_weather_model_sounding(latitude, longitude, valid_at_parsed)
    model_time = str(np.datetime_as_string(sounding.metadata.model_time, unit='m'))
    valid_time = str(np.datetime_as_string(sounding.metadata.valid_time, unit='m'))
    model_time_for_file_name = model_time.replace(":", "_")
    valid_time_for_file_name = valid_time.replace(":", "_")

    full_plot = plot_skewt_icon(sounding=sounding, parcel="surface-based")
    full_plot_filename = f"plot_{sounding.latitude_pretty}_{sounding.longitude_pretty}_" \
                         f"{model_time_for_file_name}_{valid_time_for_file_name}_full.png"
    full_plot.savefig(full_plot_filename)

    detail_plot = plot_skewt_icon(sounding=sounding, parcel="surface-based", base=1000, top=500, skew=15)
    detail_plot_filename = f"plot_{sounding.latitude_pretty}_{sounding.longitude_pretty}_" \
                           f"{model_time_for_file_name}_{valid_time_for_file_name}_detail.png"
    detail_plot.savefig(detail_plot_filename)

    result = json.dumps(SkewTResult(model_time, valid_time, full_plot_filename, detail_plot_filename).__dict__)
    response = make_response(result)
    response.mimetype = 'application/json'
    return response


if __name__ == "__main__":
    app.run()
