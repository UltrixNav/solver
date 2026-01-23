from data_classes import AnchorReport, RemoteAnchorReport

# Input data: master is src=2, others matched by seq
# DATA = (
# 	AnchorReport(
# 		src=2,
# 		seq=10,
# 		tx_ts=3762232832,
# 		rx_ts=684407504252,
# 		remote_count=2,
# 		remotes=[
# 			RemoteAnchorReport(id=3, seq=68, rx_ts=1867183020, dist_ticks=17050),
# 			RemoteAnchorReport(id=1, seq=69, rx_ts=2812612544, dist_ticks=17013),
# 		],
# 		position=(0.0, 0.0, 0.0),
# 	),
# 	[
# 		AnchorReport(
# 			src=3,
# 			seq=68,
# 			tx_ts=3283352576,
# 			rx_ts=682512426389,
# 			remote_count=2,
# 			remotes=[
# 				RemoteAnchorReport(id=1, seq=67, rx_ts=1353624937, dist_ticks=16952),
# 				RemoteAnchorReport(id=2, seq=8, rx_ts=2878270153, dist_ticks=17031),
# 			],
# 			position=(0.0, 0.0, 0.0),
# 		),
# 		AnchorReport(
# 			src=1,
# 			seq=69,
# 			tx_ts=2307677696,
# 			rx_ts=683457861423,
# 			remote_count=2,
# 			remotes=[
# 				RemoteAnchorReport(id=3, seq=68, rx_ts=1362268454, dist_ticks=16967),
# 				RemoteAnchorReport(id=2, seq=7, rx_ts=3974289807, dist_ticks=17014),
# 			],
# 			position=(0.0, 0.0, 0.0),
# 		),
# 	],
# )
DATA = (AnchorReport(src=2, seq=25, tx_ts=1922949120, rx_ts=287514884432, remote_count=2, remotes=[RemoteAnchorReport(id=1, seq=96, rx_ts=35729449, dist_ticks=17030), RemoteAnchorReport(id=3, seq=14, rx_ts=1052537890, dist_ticks=17025)], position=(0.0, 0.0, 0.0)), [AnchorReport(src=1, seq=96, tx_ts=1370324992, rx_ts=285627635376, remote_count=2, remotes=[RemoteAnchorReport(id=2, seq=23, rx_ts=1149103605, dist_ticks=17003), RemoteAnchorReport(id=3, seq=12, rx_ts=3934696338, dist_ticks=16979)], position=(0.0, 0.0, 0.0)), AnchorReport(src=3, seq=14, tx_ts=1381863936, rx_ts=286644450554, remote_count=2, remotes=[RemoteAnchorReport(id=1, seq=96, rx_ts=365043548, dist_ticks=16964), RemoteAnchorReport(id=2, seq=22, rx_ts=3096970323, dist_ticks=16992)], position=(0.0, 0.0, 0.0))])


def truncate(value: int) -> int:
    """Simulate 32-bit unsigned integer overflow."""
    return value & 0x00FFFFFFFF
# z = 3762232832
# d = 684407504252
# y = 1867183020
# b = 17050
# x = 2812612544
# a = 17013
# e = 682512426389
# f = 683457861423
# rxAn_by_T_in_cl_T = 684407504252
# txAn_in_cl_An = 3762232832
# # 2-3
# tof_Ar_to_An_in_cl_An = 17013
# rxAr_by_An_in_cl_An = 1867183020
# clockCorrection = 1
# rxAr_by_T_in_cl_T = 682512426389
def calcTdoa(otherAnchor: AnchorReport, receivedAnchor: AnchorReport) -> int:
    otherAnchorId = otherAnchor.src
    txAn_in_cl_An = receivedAnchor.tx_ts
    rxAn_by_T_in_cl_T = receivedAnchor.rx_ts
    rxAr_by_T_in_cl_T = otherAnchor.rx_ts
    tof_Ar_to_An_in_cl_An = None
    rxAr_by_An_in_cl_An = None
    for remote in receivedAnchor.remotes:
        if remote.id == otherAnchorId:
            tof_Ar_to_An_in_cl_An = remote.dist_ticks
            rxAr_by_An_in_cl_An = remote.rx_ts
            break
    if tof_Ar_to_An_in_cl_An is None or rxAr_by_An_in_cl_An is None:
        raise ValueError("Matching remote anchor not found")
    
    clockCorrection = 1.0
    
    delta_txAr_to_txAn_in_cl_An = (tof_Ar_to_An_in_cl_An + truncate(txAn_in_cl_An - rxAr_by_An_in_cl_An))
    timeDiffOfArrival_in_cl_T = truncate(rxAn_by_T_in_cl_T - rxAr_by_T_in_cl_T) - delta_txAr_to_txAn_in_cl_An * clockCorrection
    
    return timeDiffOfArrival_in_cl_T
    
# clockCorrection = 1.0

# delta_txAr_to_txAn_in_cl_An = (tof_Ar_to_An_in_cl_An + truncate(txAn_in_cl_An - rxAr_by_An_in_cl_An))
# timeDiffOfArrival_in_cl_T = truncate(rxAn_by_T_in_cl_T - rxAr_by_T_in_cl_T) - delta_txAr_to_txAn_in_cl_An * clockCorrection
def calcDistance(ticks: int) -> float:
    SPEED_OF_LIGHT = 299792458.0  # in meters per second
    TICK_DURATION = 1.0 / (499.2e6 * 128) # in seconds
    return ticks * TICK_DURATION * SPEED_OF_LIGHT

print(calcTdoa(DATA[1][1], DATA[0]))
print(calcTdoa(DATA[1][0], DATA[0]))
# print(calcDistance(11001.0))
# print(calcDistance(5528.0))
