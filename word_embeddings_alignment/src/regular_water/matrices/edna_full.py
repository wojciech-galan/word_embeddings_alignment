# EDNAFULL
#     A   T   G   C   S   W   R   Y   K   M   B   V   H   D   N
# A   5  -4  -4  -4  -4   1   1  -4  -4   1  -4  -1  -1  -1  -2
# T  -4   5  -4  -4  -4   1  -4   1   1  -4  -1  -4  -1  -1  -2
# G  -4  -4   5  -4   1  -4   1  -4   1  -4  -1  -1  -4  -1  -2
# C  -4  -4  -4   5   1  -4  -4   1  -4   1  -1  -1  -1  -4  -2
# S  -4  -4   1   1  -1  -4  -2  -2  -2  -2  -1  -1  -3  -3  -1
# W   1   1  -4  -4  -4  -1  -2  -2  -2  -2  -3  -3  -1  -1  -1
# R   1  -4   1  -4  -2  -2  -1  -4  -2  -2  -3  -1  -3  -1  -1
# Y  -4   1  -4   1  -2  -2  -4  -1  -2  -2  -1  -3  -1  -3  -1
# K  -4   1   1  -4  -2  -2  -2  -2  -1  -4  -1  -3  -3  -1  -1
# M   1  -4  -4   1  -2  -2  -2  -2  -4  -1  -3  -1  -1  -3  -1
# B  -4  -1  -1  -1  -1  -3  -3  -1  -1  -3  -1  -2  -2  -2  -1
# V  -1  -4  -1  -1  -1  -3  -1  -3  -3  -1  -2  -1  -2  -2  -1
# H  -1  -1  -4  -1  -3  -1  -3  -1  -3  -1  -2  -2  -1  -2  -1
# D  -1  -1  -1  -4  -3  -1  -1  -3  -1  -3  -2  -2  -2  -1  -1
# N  -2  -2  -2  -2  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1

EDNAFULL_matrix = {'AA': 5,
                   'AT': -4,
                   'AG': -4,
                   'AC': -4,
                   'AS': -4,
                   'AW': 1,
                   'AR': 1,
                   'AY': -4,
                   'AK': -4,
                   'AM': 1,
                   'AB': -4,
                   'AV': -1,
                   'AH': -1,
                   'AD': -1,
                   'AN': -2,
                   'TA': -4,
                   'TT': 5,
                   'TG': -4,
                   'TC': -4,
                   'TS': -4,
                   'TW': 1,
                   'TR': -4,
                   'TY': 1,
                   'TK': 1,
                   'TM': -4,
                   'TB': -1,
                   'TV': -4,
                   'TH': -1,
                   'TD': -1,
                   'TN': -2,
                   'GA': -4,
                   'GT': -4,
                   'GG': 5,
                   'GC': -4,
                   'GS': 1,
                   'GW': -4,
                   'GR': 1,
                   'GY': -4,
                   'GK': 1,
                   'GM': -4,
                   'GB': -1,
                   'GV': -1,
                   'GH': -4,
                   'GD': -1,
                   'GN': -2,
                   'CA': -4,
                   'CT': -4,
                   'CG': -4,
                   'CC': 5,
                   'CS': 1,
                   'CW': -4,
                   'CR': -4,
                   'CY': 1,
                   'CK': -4,
                   'CM': 1,
                   'CB': -1,
                   'CV': -1,
                   'CH': -1,
                   'CD': -4,
                   'CN': -2,
                   'SA': -4,
                   'ST': -4,
                   'SG': 1,
                   'SC': 1,
                   'SS': -1,
                   'SW': -4,
                   'SR': -2,
                   'SY': -2,
                   'SK': -2,
                   'SM': -2,
                   'SB': -1,
                   'SV': -1,
                   'SH': -3,
                   'SD': -3,
                   'SN': -1,
                   'WA': 1,
                   'WT': 1,
                   'WG': -4,
                   'WC': -4,
                   'WS': -4,
                   'WW': -1,
                   'WR': -2,
                   'WY': -2,
                   'WK': -2,
                   'WM': -2,
                   'WB': -3,
                   'WV': -3,
                   'WH': -1,
                   'WD': -1,
                   'WN': -1,
                   'RA': 1,
                   'RT': -4,
                   'RG': 1,
                   'RC': -4,
                   'RS': -2,
                   'RW': -2,
                   'RR': -1,
                   'RY': -4,
                   'RK': -2,
                   'RM': -2,
                   'RB': -3,
                   'RV': -1,
                   'RH': -3,
                   'RD': -1,
                   'RN': -1,
                   'YA': -4,
                   'YT': 1,
                   'YG': -4,
                   'YC': 1,
                   'YS': -2,
                   'YW': -2,
                   'YR': -4,
                   'YY': -1,
                   'YK': -2,
                   'YM': -2,
                   'YB': -1,
                   'YV': -3,
                   'YH': -1,
                   'YD': -3,
                   'YN': -1,
                   'KA': -4,
                   'KT': 1,
                   'KG': 1,
                   'KC': -4,
                   'KS': -2,
                   'KW': -2,
                   'KR': -2,
                   'KY': -2,
                   'KK': -1,
                   'KM': -4,
                   'KB': -1,
                   'KV': -3,
                   'KH': -3,
                   'KD': -1,
                   'KN': -1,
                   'MA': 1,
                   'MT': -4,
                   'MG': -4,
                   'MC': 1,
                   'MS': -2,
                   'MW': -2,
                   'MR': -2,
                   'MY': -2,
                   'MK': -4,
                   'MM': -1,
                   'MB': -3,
                   'MV': -1,
                   'MH': -1,
                   'MD': -3,
                   'MN': -1,
                   'BA': -4,
                   'BT': -1,
                   'BG': -1,
                   'BC': -1,
                   'BS': -1,
                   'BW': -3,
                   'BR': -3,
                   'BY': -1,
                   'BK': -1,
                   'BM': -3,
                   'BB': -1,
                   'BV': -2,
                   'BH': -2,
                   'BD': -2,
                   'BN': -1,
                   'VA': -1,
                   'VT': -4,
                   'VG': -1,
                   'VC': -1,
                   'VS': -1,
                   'VW': -3,
                   'VR': -1,
                   'VY': -3,
                   'VK': -3,
                   'VM': -1,
                   'VB': -2,
                   'VV': -1,
                   'VH': -2,
                   'VD': -2,
                   'VN': -1,
                   'HA': -1,
                   'HT': -1,
                   'HG': -4,
                   'HC': -1,
                   'HS': -3,
                   'HW': -1,
                   'HR': -3,
                   'HY': -1,
                   'HK': -3,
                   'HM': -1,
                   'HB': -2,
                   'HV': -2,
                   'HH': -1,
                   'HD': -2,
                   'HN': -1,
                   'DA': -1,
                   'DT': -1,
                   'DG': -1,
                   'DC': -4,
                   'DS': -3,
                   'DW': -1,
                   'DR': -1,
                   'DY': -3,
                   'DK': -1,
                   'DM': -3,
                   'DB': -2,
                   'DV': -2,
                   'DH': -2,
                   'DD': -1,
                   'DN': -1,
                   'NA': -2,
                   'NT': -2,
                   'NG': -2,
                   'NC': -2,
                   'NS': -1,
                   'NW': -1,
                   'NR': -1,
                   'NY': -1,
                   'NK': -1,
                   'NM': -1,
                   'NB': -1,
                   'NV': -1,
                   'NH': -1,
                   'ND': -1,
                   'NN': -1}
