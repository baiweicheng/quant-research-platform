"""Build qrp/data/sp500_history.csv from Wikipedia's S&P 500 change log.

Source: Wikipedia "Historical components of the S&P 500"
(https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500),
"Changes" table: every dated index addition / removal.
Retrieved: 2026-10-07. Change-log coverage: 1976-07-01 -> 2026-10-06.
Oct 2026 entries (WBD->TWLO, CTVA->VYLR, PSKY->SKYD) are from S&P Dow Jones
Indices announcements; Wikipedia's change log had not recorded them as of 2026-10-07.

Method: the table below was transcribed by hand from the Wikipedia change
log (each row: effective date, ticker added, ticker removed). This script
walks the log backwards from today's constituent list and reconstructs every
(ticker, date_added, date_removed) membership interval.

Honesty notes (also surfaced to users in plain English by sp500_history.py):
- The Wikipedia log is comprehensive from ~2007 on; earlier years are
  partial (only major changes were recorded). Membership answers for dates
  before 2007 are best-effort; every backtest on this platform starts in
  2013+, where the log is complete.
- A ticker whose true add date predates the log (or fell in a gap) gets
  date_added = 1976-07-01, the log's first entry ("on or before the log
  starts"). This never affects post-2013 backtests.
- Ticker renames (e.g. CDAY -> DAY) appear as removal+addition pairs, which
  is historically accurate: the old ticker really did leave the index.
- Removal-date convention: date_removed is the change's effective date and
  a stock counts as a member only for dates strictly before it
  (members_on uses d < date_removed), so a removed stock is unbuyable on
  the removal day itself. S&P changes are typically effective after the
  close, so this is conservative by one day — deliberately cautious for
  backtests, and immaterial at monthly rebalancing.
- Dots are normalized to dashes (BRK.B -> BRK-B), matching the codebase.

Run: python qrp/data/build_sp500_history.py
Writes: qrp/data/sp500_history.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

SOURCE = "Wikipedia 'Historical components of the S&P 500' change log"
RETRIEVED = "2026-10-07"
LOG_START = "1976-07-01"  # first change-log entry; used as "on-or-before" add date

# (effective date, ticker added, ticker removed); None when one side is empty.
# Transcribed from the Wikipedia change log, newest first.
CHANGES: list[tuple[str, str | None, str | None]] = [
    # 2026-10-06: Paramount Skydance's $81B WBD acquisition closed (S&P DJI
    # announcements; Wikipedia change log lags as of 2026-10-07).
    ("2026-10-06", "TWLO", "WBD"),
    ("2026-10-06", None, "CTVA"),
    # 2026-10-01: Corteva spinoff Vylor added; CTVA itself left 2026-10-06.
    ("2026-10-01", "VYLR", None),
    # 2026-10-06: PSKY -> SKYD is a pure rename (Paramount Skydance Corp ->
    # Skydance Corp, Nasdaq -> NYSE); encoded via the seed-list swap, same
    # convention as the FB->META / PCLN->BKNG renames below.
    ("2026-09-21", "BE", "TAP"),
    ("2026-09-21", "P", "TTD"),
    ("2026-09-21", "ILMN", "BLDR"),
    ("2026-08-18", "RDDT", "AVB"),
    ("2026-08-05", "FERG", "EA"),
    ("2026-06-30", None, "CAG"),
    ("2026-06-29", "HONA", None),
    ("2026-06-22", "MRVL", "POOL"),
    ("2026-06-22", "FLEX", "CPB"),
    ("2026-06-02", None, "EPAM"),
    ("2026-06-01", "FDXF", None),
    ("2026-05-07", "VEEV", "CTRA"),
    ("2026-04-09", "CASY", "HOLX"),
    ("2026-03-23", "VRT", "MTCH"),
    ("2026-03-23", "LITE", "MOH"),
    ("2026-03-23", "COHR", "LW"),
    ("2026-03-23", "ECHO", "PAYC"),  # SATS->ECHO 2026-06-24
    ("2026-02-09", "CIEN", "DAY"),
    ("2025-12-22", "CRH", "LKQ"),
    ("2025-12-22", "CVNA", "SOLS"),
    ("2025-12-22", "FIX", "MHK"),
    ("2025-12-11", "ARES", "K"),
    ("2025-11-28", "SNDK", "IPG"),
    ("2025-11-04", None, "EMN"),
    ("2025-11-03", "Q", None),
    ("2025-10-31", None, "KMX"),
    ("2025-10-30", "SOLS", None),
    ("2025-09-22", "EME", "ENPH"),
    ("2025-09-22", "HOOD", "CZR"),
    ("2025-09-22", "APP", "MKTX"),
    ("2025-08-28", "IBKR", "WBA"),
    ("2025-07-23", "XYZ", "HES"),
    ("2025-07-18", "TTD", "ANSS"),
    ("2025-07-09", "DDOG", "JNPR"),
    ("2025-05-19", "COIN", "DFS"),
    ("2025-03-24", "DASH", "BWA"),
    ("2025-03-24", "TKO", "TFX"),
    ("2025-03-24", "WSM", "CE"),
    ("2025-03-24", "EXE", "FMC"),
    ("2024-12-23", "APO", "QRVO"),
    ("2024-12-23", "WDAY", "AMTM"),
    ("2024-12-23", "LII", "CTLT"),
    ("2024-11-26", "TPL", "MRO"),
    ("2024-10-01", None, "BBWI"),
    ("2024-09-30", "AMTM", None),
    ("2024-09-23", "PLTR", "AAL"),
    ("2024-09-23", "DELL", "ETSY"),
    ("2024-09-23", "ERIE", "BIO"),
    ("2024-06-24", "KKR", "RHI"),
    ("2024-06-24", "CRWD", "CMA"),
    ("2024-06-24", "GDDY", "ILMN"),
    ("2024-05-08", "VST", "PXD"),
    ("2024-04-03", None, "XRAY"),
    ("2024-04-03", None, "VFC"),
    ("2024-04-02", "GEV", None),
    ("2024-04-01", "SOLV", None),
    ("2024-03-18", "SMCI", "WHR"),
    ("2024-03-18", "DECK", "ZION"),
    ("2023-12-18", "UBER", "SEE"),
    ("2023-12-18", "JBL", "ALK"),
    ("2023-12-18", "BLDR", "SEDG"),
    ("2023-10-18", "HUBB", "OGN"),
    ("2023-10-18", "LULU", "ATVI"),
    ("2023-10-03", None, "DXC"),
    ("2023-10-02", "VLTO", None),
    ("2023-09-18", "BX", "LNC"),
    ("2023-09-18", "ABNB", "NWL"),
    ("2023-08-25", "KVUE", "AAP"),
    ("2023-06-20", "PANW", "DISH"),
    ("2023-05-04", "AXON", "FRC"),
    ("2023-03-20", "FICO", "LUMN"),
    ("2023-03-15", "BG", "SBNY"),
    ("2023-03-15", "PODD", "SIVB"),
    ("2023-01-05", None, "VNO"),
    ("2023-01-04", "GEHC", None),
    ("2022-12-22", "STLD", "ABMD"),
    ("2022-12-19", "FSLR", "FBHS"),
    ("2022-12-19", None, "MBC"),
    ("2022-12-15", "MBC", None),
    ("2022-11-01", "ACGL", "TWTR"),
    ("2022-10-12", "TRGP", "NLSN"),
    ("2022-10-03", "PCG", "CTXS"),
    ("2022-10-03", "EQT", "DRE"),
    ("2022-09-19", "CSGP", "PVH"),
    ("2022-09-19", "INVH", "PENN"),
    ("2022-06-21", None, "UA"),
    ("2022-06-21", "KDP", "UAA"),
    ("2022-06-21", "ON", "IPGP"),
    ("2022-06-08", "VICI", "CERN"),
    ("2022-04-11", "WBD", "DISCA"),
    ("2022-04-11", None, "DISCK"),
    ("2022-04-04", "CPT", "PBCT"),
    ("2022-03-02", "MOH", "INFO"),
    ("2022-02-15", "NDSN", "XLNX"),
    ("2022-02-03", None, "GPS"),
    ("2022-02-02", "CEG", None),
    ("2021-12-20", "SBNY", "LEG"),
    ("2021-12-20", "SEDG", "HBI"),
    ("2021-12-20", "FDS", "WU"),
    ("2021-12-14", "EPAM", "KSU"),
    ("2021-09-20", "MTCH", "PRGO"),
    ("2021-09-20", "DAY", "UNM"),  # CDAY->DAY 2024-02-01
    ("2021-09-20", "BRO", "NOV"),
    ("2021-08-30", "TECH", "MXIM"),
    ("2021-07-21", "MRNA", "ALXN"),
    ("2021-06-04", None, "HFC"),
    ("2021-06-03", "OGN", None),
    ("2021-05-14", "CRL", "FLIR"),
    ("2021-04-20", "PTC", "VAR"),
    ("2021-03-22", "NXPI", "FLS"),
    ("2021-03-22", "PENN", "SLG"),
    ("2021-03-22", "GNRC", "XRX"),
    ("2021-03-22", "CZR", "VNT"),
    ("2021-02-12", "MPWR", "FTI"),
    ("2021-01-21", "TRMB", "CXO"),
    ("2021-01-07", "ENPH", "TIF"),
    ("2020-12-21", "TSLA", "AIV"),
    ("2020-10-12", None, "NBL"),
    ("2020-10-09", "VNT", None),
    ("2020-10-07", "POOL", "ETFC"),
    ("2020-09-21", "ETSY", "HRB"),
    ("2020-09-21", "TER", "COTY"),
    ("2020-09-21", "CTLT", "KSS"),
    ("2020-06-22", "BIO", "ADS"),
    ("2020-06-22", "TDY", "HOG"),
    ("2020-06-22", "TYL", "JWN"),
    ("2020-05-22", "WST", "HP"),
    ("2020-05-12", "DPZ", "CPRI"),
    ("2020-05-12", "DXCM", "AGN"),
    ("2020-04-06", None, "M"),
    ("2020-04-06", None, "RTN"),
    ("2020-04-03", "OTIS", None),
    ("2020-04-03", "CARR", None),
    ("2020-04-01", "HWM", "ARNC"),
    ("2020-03-02", "IR", "XEC"),
    ("2020-01-28", "PAYC", "WCG"),
    ("2019-12-23", "LYV", "AMG"),
    ("2019-12-23", "ZBRA", "TRIP"),
    ("2019-12-23", "STE", "MAC"),
    ("2019-12-09", "ODFL", "STI"),
    ("2019-12-05", "WRB", "VIAB"),
    ("2019-11-21", "NOW", "CELG"),
    ("2019-10-03", "LVS", "NKTR"),
    ("2019-09-26", "NVR", "JEF"),
    ("2019-09-23", "CDW", "TSS"),
    ("2019-08-09", "LDOS", "APC"),
    ("2019-08-09", "IEX", "FL"),
    ("2019-07-15", "TMUS", "RHT"),
    ("2019-07-01", "MKTX", "LLL"),
    ("2019-06-11", "AMCR", "BMS"),
    ("2019-06-07", "BMS", "MAT"),
    ("2019-06-03", "DD", "DWDP"),
    ("2019-06-03", "CTVA", "FLR"),
    ("2019-04-02", "DOW", "BHF"),
    # 2019-03-19 FOXA/FOX: 21st Century Fox -> Fox Corporation, same tickers,
    # no membership break, so no row.
    ("2019-02-27", "WAB", "GT"),
    ("2019-02-15", "ATO", "NFX"),
    ("2019-01-18", "TFX", "PCG"),
    ("2019-01-02", "FRC", "SCG"),
    ("2018-12-24", "CE", "ESRX"),
    ("2018-12-03", "LW", "COL"),
    ("2018-12-03", "MXIM", "AET"),
    ("2018-12-03", "FANG", "SRCL"),
    ("2018-11-13", "JKHY", "EQT"),
    ("2018-11-06", "KEYS", "CA"),
    ("2018-10-11", "FTNT", "EVHC"),
    ("2018-10-01", "ROL", "ANDV"),
    ("2018-09-14", "WCG", "XL"),
    ("2018-08-28", "ANET", "GGP"),
    ("2018-07-02", "CPRT", "DPS"),
    ("2018-06-20", "CPAY", "TWX"),  # FLT->CPAY 2024-03-25
    ("2018-06-18", "BR", "RRC"),
    ("2018-06-18", "HFC", "AYI"),
    ("2018-06-07", "TWTR", "MON"),
    ("2018-06-05", "EVRG", "NAVI"),
    ("2018-05-31", "ABMD", "WYN"),
    ("2018-04-04", "MSCI", "CSRA"),
    ("2018-03-19", "TTWO", "SIG"),
    ("2018-03-19", "SIVB", "PDCO"),
    ("2018-03-19", "NKTR", "CHK"),
    ("2018-03-07", "IPGP", "SNI"),
    ("2018-01-03", "HII", "BCR"),
    ("2017-10-13", "NCLH", "LVLT"),
    ("2017-09-18", "CDNS", "SPLS"),
    ("2017-09-01", "DWDP", "DOW"),
    ("2017-09-01", "SBAC", "DD"),
    ("2017-08-29", "IQV", "WFM"),  # Q->IQV 2017-11-15 (QuintilesIMS; different company from 2011 Qwest Q)
    ("2017-08-08", "BHF", "AN"),
    ("2017-07-26", "DRE", "RIG"),
    ("2017-07-26", "AOS", "BBBY"),
    ("2017-07-26", "PKG", "MUR"),
    ("2017-07-26", "RMD", "MNK"),
    ("2017-07-26", "MGM", "RAI"),
    ("2017-07-07", "BKR", "BHI"),
    ("2017-06-19", "HLT", "YHOO"),
    ("2017-06-19", "ALGN", "TDC"),
    ("2017-06-19", "ANSS", "R"),
    ("2017-06-19", "EG", "MJN"),  # RE->EG (Everest Group rename)
    ("2017-06-02", "INFO", "TGNA"),
    ("2017-04-05", "IT", "DNB"),
    ("2017-04-04", "DXC", "SWN"),
    ("2017-03-20", "AMD", "URBN"),
    ("2017-03-20", "RJF", "FTR"),
    ("2017-03-20", "ARE", "FSLR"),
    ("2017-03-16", "SNPS", "HAR"),
    ("2017-03-13", "DISH", "LLTC"),
    ("2017-03-02", "REG", "ENDP"),
    ("2017-03-01", "CBOE", "PBI"),
    ("2017-02-28", "INCY", "SE"),
    ("2017-01-05", "IDXX", "STJ"),
    ("2016-12-02", "MAA", "OI"),
    ("2016-12-02", "EVHC", "LM"),
    ("2016-11-01", "ARNC", "AA"),
    ("2016-09-30", "COTY", "DO"),
    ("2016-09-22", "COO", "HOT"),
    ("2016-09-08", "CHTR", "EMC"),
    ("2016-09-06", "MTD", "TYC"),
    ("2016-07-05", "FTV", "CPGX"),
    ("2016-07-01", "LNT", "GAS"),
    ("2016-07-01", "ALB", "TE"),
    ("2016-06-22", "FBHS", "CVC"),
    ("2016-06-03", "TDG", "BXLT"),
    ("2016-05-31", "AJG", "CCE"),
    ("2016-05-23", "LKQ", "ARG"),
    ("2016-05-18", "DLR", "TWC"),
    ("2016-05-13", "ALK", "SNDK"),
    ("2016-05-03", "AYI", "ADT"),
    ("2016-04-25", "GPN", "GME"),
    ("2016-04-18", "ULTA", "THC"),
    ("2016-04-08", "UA", None),
    ("2016-04-04", "FL", "CAM"),
    ("2016-03-30", "HOLX", "POM"),
    ("2016-03-30", "CNC", "ESV"),
    ("2016-03-07", "UDR", "GMCR"),
    ("2016-03-04", "AWK", "CNX"),
    ("2016-02-22", "CXO", "PCL"),
    ("2016-02-01", "CFG", "PCP"),
    ("2016-02-01", "FRT", "BRCM"),
    ("2016-01-19", "EXR", "ACE"),
    ("2016-01-05", "WTW", "FOSL"),  # WLTW->WTW 2022-01-10
    ("2015-12-29", "CHD", "ALTR"),
    ("2015-12-15", None, "CMCSK"),
    ("2015-12-01", "CSRA", "CSC"),
    ("2015-11-19", "ILMN", "SIAL"),
    ("2015-11-18", "SYF", "GNW"),
    ("2015-11-02", "HPE", "HCBK"),
    ("2015-10-07", "VRSK", "JOY"),
    ("2015-09-18", "CMCSK", None),
    ("2015-09-18", "FOX", None),
    ("2015-09-18", "NWS", None),
    ("2015-09-02", "UAL", "HSP"),
    ("2015-08-28", "ATVI", "PLL"),
    ("2015-07-29", "SIG", "DTV"),
    ("2015-07-20", "PYPL", "NE"),
    ("2015-07-08", "AAP", "FDO"),
    ("2015-07-06", "KHC", "KRFT"),
    ("2015-07-02", "CPGX", "ATI"),
    ("2015-07-01", "JBHT", "TEG"),
    ("2015-07-01", "BXLT", "QEP"),
    ("2015-06-11", "QRVO", "LO"),
    ("2015-04-07", "O", "WIN"),
    # 2015-03-23: Actavis bought Allergan and renamed itself Allergan (keeping
    # the AGN ticker). The listing that left the index was Actavis (ACT);
    # AGN membership is continuous, so AAL replaced ACT here.
    ("2015-03-23", "AAL", "ACT"),
    ("2015-03-23", "EQIX", "DNR"),
    ("2015-03-23", "SLG", "NBR"),
    ("2015-03-23", "HBI", "AVP"),
    ("2015-03-18", "HSIC", "CFN"),
    ("2015-03-12", "SWKS", "PETM"),
    ("2015-01-27", "HCA", "SWY"),
    ("2015-01-27", "ENDP", "COV"),
    ("2014-12-05", "RCL", "BMS"),
    ("2014-11-05", "LVLT", "JBL"),
    ("2014-09-20", "URI", "BTU"),
    ("2014-09-20", "UHS", "GHC"),
    ("2014-08-18", "MNK", "RDC"),
    ("2014-08-06", "DISCK", None),
    ("2014-07-02", "MLM", "X"),
    ("2014-07-01", "AMG", "FRX"),
    ("2014-06-20", "XEC", "IGT"),
    ("2014-05-08", "AVGO", "LSI"),
    # 2014-05-01: Under Armour joined as UA; in Apr 2016 the old Class A
    # shares were renamed UAA and a new Class C listed as UA. Encoding the
    # 2014 add as UAA keeps both tickers' membership continuous and matches
    # Yahoo's (backfilled) price history for each ticker.
    ("2014-05-01", "UAA", "BEAM"),
    ("2014-05-01", "NAVI", "SLM"),
    ("2014-04-03", "GOOGL", None),
    ("2014-04-02", "ESS", "CLF"),
    ("2014-03-21", "GMCR", "WPX"),
    ("2014-01-24", "TSCO", "LIFE"),
    ("2013-12-23", "ADS", "ANF"),
    ("2013-12-23", "MHK", "JDSU"),
    ("2013-12-23", "META", "TER"),  # FB->META 2022-06-09
    ("2013-12-10", "GGP", "MOLX"),
    ("2013-12-02", "ALLE", "JCP"),
    ("2013-11-13", "CPRI", "NYX"),  # KORS->CPRI 2019-01-02
    ("2013-10-29", "RIG", "DELL"),
    ("2013-09-20", "VRTX", "AMD"),
    ("2013-09-20", "AME", "SAI"),
    ("2013-09-10", "DAL", "BMC"),
    ("2013-07-08", "NLSN", "S"),
    ("2013-07-01", "NWSA", "APOL"),
    ("2013-06-21", "ZTS", "FHN"),
    ("2013-06-06", "GM", "HNZ"),
    ("2013-05-23", "KSU", "DF"),
    ("2013-05-08", "MAC", "CVH"),
    ("2013-04-30", "REGN", "PCS"),
    ("2013-02-15", "PVH", "BIG"),
    ("2013-01-02", "ABBV", "FII"),
    ("2012-12-21", "APTV", "TIE"),  # DLPH->APTV (Aptiv rename)
    ("2012-12-11", "GRMN", "RRD"),
    ("2012-12-03", "DG", "CBE"),
    ("2012-10-10", "PETM", "SUN"),
    ("2012-10-02", "KRFT", "ANR"),
    ("2012-10-02", "MDLZ", "KFT"),
    ("2012-10-01", "ADT", "LXK"),
    ("2012-10-01", "PNR", "DV"),
    ("2012-09-05", "LYB", "SHLD"),
    ("2012-07-31", "ESV", "GR"),
    ("2012-07-02", "STX", "PGN"),
    ("2012-06-29", "MNST", "SLE"),
    ("2012-06-05", "LRCX", "NVLS"),
    ("2012-05-21", "ALXN", "MMI"),
    ("2012-05-17", "KMI", "EP"),
    ("2012-04-23", "PSX", "SVU"),
    ("2012-04-03", "FOSL", "MHS"),
    ("2012-03-13", "CCI", "CEG"),
    ("2011-12-31", "WPX", "CPWR"),
    ("2011-12-20", "TRIP", "TLAB"),
    ("2011-12-16", "BWA", "AKS"),
    ("2011-12-16", "PRGO", "MWW"),
    ("2011-12-16", "DLTR", "WFR"),
    # 2011-12-12 GAS/GAS: Nicor acquired by AGL, kept the ticker - no break.
    ("2011-11-18", "CBE", "JNS"),
    ("2011-10-31", "XYL", "ITT"),
    ("2011-10-14", "TEL", "CEPH"),
    ("2011-09-23", "MOS", "NSM"),
    ("2011-07-05", "ACN", "MI"),
    ("2011-06-30", "MPC", "RSH"),
    ("2011-06-01", "ANR", "MEE"),
    ("2011-04-27", "CMG", "NOVL"),
    ("2011-04-01", "BLK", "GENZ"),
    ("2011-03-31", "EW", "Q"),
    ("2011-02-28", "COV", "MFE"),
    ("2011-02-25", "JOY", "AYE"),  # JOYG->JOY ticker change
    ("2011-01-03", "MMI", "MDP"),
    ("2010-12-17", "CVC", "KG"),
    ("2010-12-17", "FFIV", "EK"),
    ("2010-12-17", "NFLX", "ODP"),
    ("2010-12-17", "NFX", "NYT"),
    ("2010-11-17", "TT", "PTV"),  # old IR->TT 2020-02-29 (Trane; new IR is a different company)
    ("2010-08-26", "TYC", "SII"),
    ("2010-07-14", "CB", "MIL"),
    ("2010-06-30", "QEP", "STR"),
    ("2010-06-28", "KMX", "XTO"),
    ("2010-04-29", "CERN", "BJS"),
    ("2010-02-26", "HP", "RX"),
    ("2009-12-18", "V", "CIEN"),
    ("2009-12-18", "MJN", "DYN"),
    ("2009-12-18", "CLF", "KBH"),
    ("2009-12-18", "SAI", "CVG"),
    ("2009-12-18", "ROST", "MBI"),
    ("2009-11-03", "BKNG", "SGP"),  # PCLN->BKNG 2018-02-27
    ("2009-09-28", "ARG", "CBE"),
    ("2009-08-19", "FMC", "CTX"),
    ("2009-06-29", "PCS", None),
    ("2009-06-25", None, "TEL"),
    ("2009-06-05", "FTI", "COV"),
    ("2009-03-03", "HRL", "ACAS"),
    ("2009-03-03", "VTR", "JNY"),
    ("2008-12-31", "OI", "WB"),
    ("2008-09-16", "LHX", "LEH"),  # HRS->LHX 2019-07-01 (L3Harris merger)
    ("2008-09-12", "CRM", "FRE"),
    ("2008-09-12", "FAST", "FNM"),
    ("2008-07-01", "AKS", "CFC"),
    ("2008-06-23", "CTRA", "BC"),  # COG->CTRA 2021-10-01 (Coterra)
    ("2008-06-23", "MEE", "OMX"),
    ("2008-06-10", "LO", "ABK"),
    ("2007-12-20", "RRC", "TRB"),
    ("2007-12-13", "GME", "DJ"),
    ("2007-10-26", "J", "AV"),  # JEC->J (Jacobs rename)
    ("2007-10-02", "EXPE", "SLR"),
    ("2007-10-01", "TDC", "NCR"),
    ("2007-09-27", "ANDV", "MXIM"),  # TSO->ANDV 2017-08-01 (Andeavor)
    ("2007-09-26", "ICE", "FDC"),
    ("2007-08-24", "JEF", "KSE"),  # LUK->JEF 2018-06-01 (Jefferies)
    ("2007-07-02", "DFS", "ADCT"),
    ("2007-03-30", "KFT", "TSG"),
    ("2007-01-10", "AVB", "SBL"),
    ("2006-06-02", "JNPR", "ABS"),
    # 2005-11-18: SBC bought AT&T Corp and took the T ticker itself; the T
    # listing never left the index, so this is an addition with no removal
    # (the merger freed the slot AMZN filled).
    ("2005-11-18", "AMZN", None),
    ("2005-07-01", "STZ", "GLK"),
    ("2003-09-25", "ESRX", "QTRN"),
    ("2000-12-05", "INTU", "BS"),
    ("2000-12-05", "SBL", "OI"),
    ("2000-12-05", "AYE", "GRA"),
    ("2000-12-05", "ABK", "CCK"),
    ("2000-07-27", "JDSU", "RAD"),
    ("2000-06-12", "CVG", "TMC"),
    ("2000-06-07", "SBUX", "SMS"),
    ("1999-12-08", "YHOO", "LDW"),
    ("1999-06-09", "ELV", "HPH"),  # WLP->ANTM->ELV (Anthem/Elevance)
    ("1999-04-12", "ACT", None),
    ("1998-12-11", "FSR", "AN"),
    ("1998-12-11", "CCL", "GRN"),
    ("1998-12-11", "CPWR", "SUN"),
    ("1997-06-17", "CCR", "USL"),
    # 1994-09-30: McKesson (MCK) is a current constituent, so the log's 1994
    # "removal" cannot stand as written (its re-addition was never logged).
    # Treated as continuous membership; NCC's later exit is likewise unlogged.
    ("1994-09-30", "NCC", None),
    ("1976-07-01", "BUD", "HNG"),
    ("1976-07-01", "DIS", "AYE"),
]


def normalize(ticker: str | None) -> str | None:
    """BRK.B -> BRK-B, matching the codebase's dash convention."""
    if ticker is None:
        return None
    return ticker.replace(".", "-").upper()


def current_tickers() -> list[str]:
    """Today's constituent list, the seed of the reverse walk."""
    path = Path(__file__).resolve().parent / "sp500_tickers.txt"
    tickers: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tickers.append(normalize(line))
    return tickers


def build_intervals() -> tuple[list[tuple[str, str, str]], list[str]]:
    """Reverse-walk the change log into (ticker, added, removed) intervals.

    removed == "" means "still a member". Anomalies (log inconsistencies,
    mostly from the partial pre-2007 coverage) are collected, not raised.
    """
    current = current_tickers()
    # ticker -> end date of its currently-open interval (None = still in)
    open_end: dict[str, str | None] = {t: None for t in current}
    intervals: list[tuple[str, str, str]] = []
    anomalies: list[str] = []

    for day, added, removed in sorted(CHANGES, key=lambda r: r[0], reverse=True):
        a, r = normalize(added), normalize(removed)
        if a is not None and a == r:
            continue  # same-ticker company change: no membership break
        if a is not None:
            if a in open_end:
                intervals.append((a, day, open_end.pop(a) or ""))
            else:
                # The log shows an addition with no matching open interval:
                # a removal row is missing (partial early coverage). Record
                # the add date but leave the end open would wrongly claim
                # current membership, so record a zero-length interval and
                # flag it.
                anomalies.append(
                    f"addition with no open interval: {a} on {day}"
                )
                intervals.append((a, day, day))
        if r is not None:
            if r in open_end:
                # Removed twice with no logged re-add (partial early
                # coverage): the ticker was clearly a member until the
                # later date, so keep the later end.
                anomalies.append(
                    f"double removal without re-add: {r} on {day} "
                    f"(keeping later end {open_end[r]})"
                )
                open_end[r] = max(open_end[r], day) if open_end[r] else day
            else:
                open_end[r] = day

    for ticker, end in sorted(open_end.items()):
        # Still open after the oldest change: member since on-or-before the
        # log's start.
        intervals.append((ticker, LOG_START, end or ""))

    intervals.sort(key=lambda row: (row[0], row[1]))
    # Zero-length intervals (addition with no matchable open interval, from
    # gaps in the log's early coverage) carry no membership information, so
    # they are reported as anomalies but not written to the CSV.
    dropped = sum(1 for _, a, r in intervals if a == r)
    intervals = [row for row in intervals if row[1] != row[2]]
    return intervals, anomalies, dropped


def main() -> None:
    intervals, anomalies, dropped = build_intervals()
    out = Path(__file__).resolve().parent / "sp500_history.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "date_added", "date_removed"])
        w.writerows(intervals)
    print(f"wrote {out} ({len(intervals)} intervals)")
    current_open = sum(1 for _, _, removed in intervals if not removed)
    print(f"intervals still open (should equal current list): {current_open}")
    print(f"zero-length intervals dropped: {dropped}")
    if anomalies:
        print(f"{len(anomalies)} anomalies:")
        for a in anomalies:
            print("  -", a)


if __name__ == "__main__":
    main()
