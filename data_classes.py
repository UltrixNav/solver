class RemoteAnchorReport(object):
    id: int
    seq: int
    rx_ts: int
    dist_ticks: int
    
    def __init__(self, id: int, seq: int, rx_ts: int, dist_ticks: int):
        self.id = id
        self.seq = seq
        self.rx_ts = rx_ts
        self.dist_ticks = dist_ticks
        
    def __repr__(self):
        return f"RemoteAnchorReport(id={self.id}, seq={self.seq}, rx_ts={self.rx_ts}, dist_ticks={self.dist_ticks})"

class AnchorReport(object):
    src: int
    seq: int
    tx_ts: int
    rx_ts: int
    remote_count: int
    remotes: list[RemoteAnchorReport]
    position: tuple[float, float, float]
    
    def __init__(self, src: int, seq: int, tx_ts: int, rx_ts: int, remote_count: int, remotes: list[RemoteAnchorReport], position: tuple[float, float, float]):
        self.src = src
        self.seq = seq
        self.tx_ts = tx_ts
        self.rx_ts = rx_ts
        self.remote_count = remote_count
        self.remotes = remotes
        self.position = position
    
    def __repr__(self):
        return f"AnchorReport(src={self.src}, seq={self.seq}, tx_ts={self.tx_ts}, rx_ts={self.rx_ts}, remote_count={self.remote_count}, remotes={self.remotes}, position={self.position})"