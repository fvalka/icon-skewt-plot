model = "icon-eu"
dwd_base_url = "https://opendata.dwd.de/weather/nwp/"
content_log_url = "%scontent.log.bz2" % dwd_base_url
http_download_pool = 50
cfgrib_parallel = True

bucket_name = "icon-skewt-plot"
bucket_public_url = "https://storage.googleapis.com/icon-skewt-plot/"
