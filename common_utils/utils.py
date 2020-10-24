

def args2str(args):
    out = "===== Console arguments =====\n"
    for arg in vars(args):
        out += arg + ": " + str(getattr(args, arg)) + "\n"
    return out


def decode_time_str(time_str):
    assert type(time_str) == str
    assert len(time_str) >= 2
    try:
        number = int(time_str[:-1])
        unit = time_str[-1:]
        if unit == "m":
            number *= 60
        if unit == "h":
            number *= 60 * 60
        if unit == "d":
            number *= 60 * 60 * 24
        return number
    except Exception as e:
        print(f"Invalid time string: {time_str} {e}")
        return 0


def validate_box(box, xsz, ysz):
    x, y, w, h = box
    x2, y2 = x + w - 1, y + h - 1
    x = max(0, min(int(xsz) - 1, x))
    x2 = max(0, min(int(xsz) - 1, x2))
    y = max(0, min(int(ysz) - 1, y))
    y2 = max(0, min(int(ysz) - 1, y2))
    return x, y, x2 - x + 1, y2 - y + 1


