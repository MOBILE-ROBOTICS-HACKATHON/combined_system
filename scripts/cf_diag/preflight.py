# Pre-flight check of a Crazyflie over the radio (cflib directly, NO ROS). Run it BEFORE cf_server:
# only one program can use the Crazyradio at a time. The drone must sit flat on the takeoff spot.
#   cf_check                (helper; uses $CF's uri from $CF_YAML)
#   python3 scripts/cf_diag/preflight.py radio://0/80/2M/E7E7E7E7E7
# Reads: battery, which decks are mounted, Lighthouse base stations seen (Lighthouse deck) or height
# sensor + flow quality (Flow deck), and whether the position estimate stays STILL for 3 s.
# Ends with READY or NOT READY (and why). Motors never run.
import os
import sys
import time
import yaml
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger

DECKS = {"bcFlow2": "Flow deck v2", "bcFlow": "Flow deck v1", "bcLighthouse4": "Lighthouse deck",
         "bcLoco": "Loco deck", "bcZRanger2": "Z-ranger v2", "bcMultiranger": "Multi-ranger", "bcAI": "AI deck"}


def uri_from_yaml():
    cfg = yaml.safe_load(open(os.environ.get("CF_YAML", "/hackathon/ws/src/team_pkg/config/crazyflies.yaml")))
    return cfg["robots"][os.environ.get("CF", "cf1")]["uri"]


def main():
    uri = sys.argv[1] if len(sys.argv) > 1 else uri_from_yaml()
    cflib.crtp.init_drivers()
    print(f"connecting to {uri} ...")
    warnings, blockers = [], []          # 11 Oct (user): lighter checks, only crashed/locked block
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="/tmp/cf_cache")) as scf:
        cf = scf.cf
        has_param = lambda n: cf.param.toc.get_element_by_complete_name(n) is not None
        has_log = lambda n: cf.log.toc.get_element_by_complete_name(n) is not None
        decks = [label for k, label in DECKS.items() if has_param(f"deck.{k}") and int(cf.param.get_value(f"deck.{k}"))]
        print(f"decks: {', '.join(decks) or 'NONE'}")
        positioning = [d for d in decks if d.startswith(("Flow", "Lighthouse", "Loco"))]
        if not positioning:
            warnings.append("no positioning deck: the drone can't hold position or go to points")
        cf.param.set_value("stabilizer.estimator", "2")        # Kalman, as cf_server will set it
        cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        cf.param.set_value("kalman.resetEstimation", "0")
        time.sleep(1.5)                                        # let the estimate settle after the reset
        log = LogConfig("preflight", period_in_ms=100)
        names = ["pm.vbat", "stateEstimate.x", "stateEstimate.y", "stateEstimate.z", "supervisor.info"]
        if any(d.startswith("Flow") for d in decks):
            names += ["range.zrange", "motion.squal"]
        if "Lighthouse deck" in decks:
            names += ["lighthouse.bsReceive", "lighthouse.bsActive"]
        types = {"range.zrange": "uint16_t", "motion.squal": "uint8_t", "supervisor.info": "uint16_t",
                 "lighthouse.bsReceive": "uint16_t", "lighthouse.bsActive": "uint16_t"}
        for n in names:
            if has_log(n):
                log.add_variable(n, types.get(n, "float"))
        rows = []
        with SyncLogger(scf, log) as logger:
            t0 = time.time()
            for _, data, _ in logger:
                rows.append(data)
                if time.time() - t0 > 3.0:
                    break
    last = rows[-1]   # latest sample
    vbat = last.get("pm.vbat")
    print(f"battery: {vbat:.2f} V" if vbat else "battery: ?")
    if vbat and vbat < 3.6:
        warnings.append(f"battery {vbat:.2f} V < 3.6 V: charge or swap it soon")
    if "range.zrange" in last:
        z = last["range.zrange"]
        print(f"height sensor: {z} mm (on the floor: ~0-40 mm)")
        if z > 100:
            warnings.append(f"height sensor reads {z} mm while on the floor: deck faulty or the drone isn't on the floor")
    if "supervisor.info" in last:                    # 11 Oct: a drone stuck in "crashed" ignored takeoff silently
        info = int(last["supervisor.info"])
        bits = ["canBeArmed", "isArmed", "autoArm", "canFly", "isFlying", "isTumbled", "isLocked", "isCrashed",
                "hlControlActive", "hlTrajFinished", "hlControlDisabled"]
        print(f"supervisor: {info} = {', '.join(b for i, b in enumerate(bits) if info >> i & 1) or 'nothing set'}")
        if info & 128:
            blockers.append("supervisor says CRASHED: switch the drone OFF and ON (power-cycle), then cf_check again")
        elif info & 64:
            blockers.append("supervisor says LOCKED (after a kill): power-cycle the drone")
        elif info & 32:
            warnings.append("supervisor says TUMBLED: put it flat on its feet")
    if "lighthouse.bsReceive" in last:
        seen = bin(last["lighthouse.bsReceive"]).count("1")
        used = bin(last["lighthouse.bsActive"]).count("1")
        print(f"Lighthouse base stations: {seen} received, {used} used for position (need >= 2)")
        if used < 2:
            warnings.append(f"Lighthouse: only {used} base station(s) usable. Are they switched on (green LED)? "
                            "Does the deck on top of the drone have a free view of them? Is the drone inside "
                            "their area?")
    if "motion.squal" in last:
        print(f"flow quality: {last['motion.squal']} (higher = more floor texture; ~>20 is fine)")
    xyz = [[r[f"stateEstimate.{a}"] for r in rows] for a in "xyz"]
    move = max(max(v) - min(v) for v in xyz)     # a drone on the ground should not "move" at all
    print(f"position estimate: ({xyz[0][-1]:+.3f}, {xyz[1][-1]:+.3f}, {xyz[2][-1]:+.3f}) m, "
          f"moved {move * 100:.1f} cm in 3 s while standing still")
    if move > 0.20:
        warnings.append(f"position estimate jumps {move * 100:.0f} cm while the drone stands still: "
                        "the positioning system isn't working, flying now would crash")
    elif "Lighthouse deck" in decks and abs(xyz[2][-1]) > 0.05:
        print(f"note: on the floor z = {xyz[2][-1]:+.2f} m (Lighthouse origin not at floor level). cf_keys and "
              "cf_guard handle it; the cf_takeoff/cf_land/cf_descend helpers assume the floor is z = 0")
    print()
    if blockers:
        print("NOT READY (the firmware will refuse to fly):\n  - " + "\n  - ".join(blockers))
        sys.exit(1)
    if warnings:
        print("WARNINGS (you may still fly; cf_keys asks for FORCE if the position is bad):\n  - " + "\n  - ".join(warnings))
    print("READY: start cf_server with the drone exactly where it is now")


if __name__ == "__main__":
    main()
