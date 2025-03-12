def parse_r(r: int, n: int, layers: int):
    if r <= 0 or n <= 0:
        rs = [0 for _ in range(layers)]
    else:
        r_per_layer = r // n
        if r_per_layer % 2 == 1:
            # Ensure that `r_per_layer` is an even number.
            r_per_layer -= 1
        rs = [r_per_layer for _ in range(n - 1)]
        rs.append(r - sum(rs))
        for _ in range(layers - n):
            rs.append(0)
    return rs


def do_nothing(x, *args, **kwargs):
    return x


def trim_none(*args, **kwargs):
    return do_nothing


def build_none(r: int, n: int, k: int, layers: int = 6):
    r_list = [0 for _ in range(layers)]
    tgtg_info = dict(r=r_list, n=n, k=k, enable=False)
    return trim_none, tgtg_info
