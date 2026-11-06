from debuglog import DEBUG_LOG_MAX_BYTES, DebugLog


def test_append_and_dump_roundtrip():
    log = DebugLog()
    log.append("first line\nsecond line")
    log.append("third line")
    assert log.dump() == "first line\nsecond line\nthird line"


def test_cap_drops_oldest_whole_lines():
    log = DebugLog(max_bytes=20)
    for entry in ("aaaa", "bbbb", "cccc", "dddd"):
        log.append(entry)
    log.cap()
    dumped = log.dump()
    assert len(dumped.encode("utf-8")) < 20
    assert "dddd" in dumped
    assert "aaaa" not in dumped


def test_single_line_over_limit_empties_log():
    log = DebugLog(max_bytes=10)
    log.append("x" * 50)
    log.cap()
    assert log.dump() == ""


def test_dump_caps_implicitly():
    log = DebugLog(max_bytes=20)
    for entry in ("aaaa", "bbbb", "cccc", "dddd", "eeee"):
        log.append(entry)
    dumped = log.dump()
    assert len(dumped.encode("utf-8")) < 20
    assert "eeee" in dumped
    assert "aaaa" not in dumped


def test_byte_accounting_not_char_counting():
    # "őőő" is 3 chars but 6 UTF-8 bytes; with the newline that is 7 bytes.
    log = DebugLog(max_bytes=7)
    log.append("őőő")
    log.cap()
    assert log.dump() == ""


def test_default_limit_is_under_one_megabyte():
    assert DEBUG_LOG_MAX_BYTES == 1_000_000
