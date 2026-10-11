# Find Crazyflies on the radio and tell OURS from other teams' (no ROS; the Crazyradio must be FREE:
# stop cf_server / cfclient first).
#   python3 scripts/cf_diag/find_drones.py scan [ADDR ...]     list drones that answer (default: common addresses)
#   python3 scripts/cf_diag/find_drones.py identify            scan, you switch YOUR drones OFF, scan again:
#                                                              the ones that VANISHED are yours (then back ON)
# Why: 10 Oct, the factory address E7E7E7E7E7 found nothing; our drone was E7E7E7E702 (channel 70) and another
# team's drone E7E7E7E704 answered too. A drone often answers on neighbouring channels as well (79 + 81 for a
# drone on 80): its real channel is the middle one; confirm with cf_check radio://0/<ch>/2M/<ADDR>.
import os
import sys
import yaml
import cflib.crtp

DEFAULT = [0xE7E7E7E7E7] + [0xE7E7E7E700 + i for i in range(0x10)]


def configured():
    try:
        cfg = yaml.safe_load(open(os.environ.get("CF_YAML", "/hackathon/ws/src/team_pkg/config/crazyflies.yaml")))
        return {name: r["uri"] for name, r in cfg["robots"].items() if r.get("enabled", True)}
    except Exception:
        return {}


def scan(addrs):
    found = {}
    for a in addrs:
        uris = [u for u, _ in cflib.crtp.scan_interfaces(a)]
        if uris:
            found[f"{a:010X}"] = uris
            print(f"  {a:010X}: {', '.join(uris)}", flush=True)
    return found


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
    cflib.crtp.init_drivers()
    cfg = configured()
    addrs = list(dict.fromkeys(DEFAULT + [int(u.split("/")[-1], 16) for u in cfg.values()]
                               + [int(x, 16) for x in sys.argv[2:]]))
    print(f"scanning {len(addrs)} addresses (~{len(addrs) * 1.4:.0f} s) ...")
    before = scan(addrs)
    if not before:
        print("NOTHING answers: Crazyradio plugged in (lsusb | grep 1915)? drones ON? cf_server/cfclient closed?")
    for name, uri in cfg.items():
        addr = uri.split("/")[-1].upper()
        print(f"config {name}: {uri} -> {'answers' if addr in before else 'NOT FOUND (off? wrong address?)'}")
    if cmd != "identify":
        return
    input("\nSwitch YOUR drone(s) OFF now (only yours), then press Enter ... ")
    after = scan(list(int(a, 16) for a in before))
    mine = [a for a in before if a not in after]
    others = [a for a in before if a in after]
    print(f"\nYOURS (vanished when switched off): {', '.join(mine) or 'none: were they really off?'}")
    print(f"OTHER TEAMS' (still answering):    {', '.join(others) or 'none'}")
    if mine:
        input("\nSwitch your drone(s) back ON, wait until the LEDs settle, press Enter ... ")
        back = scan([int(a, 16) for a in mine])
        for a in mine:
            chans = sorted({int(u.split("/")[3]) for u in before[a]})
            ch = chans[len(chans) // 2]
            print(f"  {a}: {'back' if a in back else 'NOT back yet'}; channel(s) {chans} -> use radio://0/{ch}/2M/{a}")
        print("Put these URIs into ws/src/team_pkg/config/crazyflies.yaml (cf1) / crazyflies_two.yaml (cf1, cf2),")
        print("then check each one: CF=cf1 cf_check, CF=cf2 cf_check (with CF_DRONES=2 for the second).")


if __name__ == "__main__":
    main()
