"""Fixed six-hour model data contract."""

WEATHER = [
    "APCP_surface",
    "TMP_2maboveground",
    "SPFH_2maboveground",
    "PRES_surface",
    "DSWRF_surface",
    "DLWRF_surface",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
]
HISTORY, HORIZON = 120, 40
INPUT_CHANNELS = 66
