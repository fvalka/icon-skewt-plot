model = "icon-eu"
dwd_base_url = "https://opendata.dwd.de/weather/nwp/"
content_log_url = "%scontent.log.bz2" % dwd_base_url
parameter_all_levels_workers = 10
http_download_pool = 50
cfgrib_parallel = True
level_workers = 30

bucket_name = "icon-skewt-plot"
bucket_public_url = "https://storage.googleapis.com/icon-skewt-plot/"

sounding_api = "https://nwp-sounding-mw5zsrftba-ew.a.run.app"
sounding_api_retries = 3
#sounding_api = "http://localhost:5001"
