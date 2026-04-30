"""LS API constants — dataset names, geographic identifiers, sentinel rules."""

API_HOST = "https://api.wisenvr.com"
LS_BASE = f"{API_HOST}/LS"
TOKEN_URL = f"{API_HOST}/apis/token"
DEFAULT_SCOPE = "LS:ALL"

# Dataset long-names (use with /dataset/{ds}/station/{sid}/data)
GNSS_WRA = "ls-wra-gnss-obs"
GNSS_NCKU_1D = "ls-ncku-gnss1day-obs"
GNSS_NCKU_7D = "ls-ncku-gnss7day-obs"
MLCW = "ls-wra-mlcw-obs"
DBM = "ls-wra-dbm-obs"
LSP = "ls-wra-lsp-obs"
GW_10MIN = "gw-wra-gw10min-obs"
RAIN_10MIN = "met-cwa-rain10min-obs"
MET_HOURLY = "met-cwa-met1hr-obs"

ZHUOSHUI_ZONE_ID = 50

# Negative rainfall values are sentinels (-998 or otherwise); clip to zero
RAINFALL_NEG_REPLACE = 0.0
