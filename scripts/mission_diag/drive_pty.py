#!/usr/bin/env python3
"""Test helper: run an interactive tool in a pseudo-terminal and type keys at given times.
    python3 scripts/mission_diag/drive_pty.py "COMMAND" "3:t" "3.5:FLY\\n" "20:q"
Each "SECONDS:TEXT" is typed at that time (escape sequences allowed, e.g. \\x1b[A = arrow up); prints the
cleaned output (repeated status lines collapsed) and ends 4 s after the last key."""
import os, pty, re, select, sys, time
pid, fd = pty.fork()
if pid == 0:
    os.execvp("bash", ["bash", "-c", sys.argv[1]])
t0 = time.time()
steps = [(float(a.split(":", 1)[0]), a.split(":", 1)[1].encode().decode("unicode_escape")) for a in sys.argv[2:]]
out, end = b"", (steps[-1][0] if steps else 0) + 4
while time.time() - t0 < end:
    while steps and time.time() - t0 >= steps[0][0]:
        os.write(fd, steps.pop(0)[1].encode())
    if select.select([fd], [], [], 0.05)[0]:
        try:
            out += os.read(fd, 4096)
        except OSError:
            break
txt = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", out.decode(errors="ignore")).replace("\r", "\n")
prev = None
for line in (l.strip() for l in txt.split("\n") if l.strip()):
    key = re.sub(r"[-+]?\d+\.\d+", "#", line)
    if key != prev:
        print("   |", line[:200])
    prev = key
