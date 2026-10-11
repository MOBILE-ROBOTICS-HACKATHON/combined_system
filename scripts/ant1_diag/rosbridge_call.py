# Call a robot service through its rosbridge (port 5001). Runs on the HOST (python3 + websockets), no ROS.
# Use it when Humble can't call a Jazzy-only service type, or when DDS service discovery over Wi-Fi is flaky.
#   python3 scripts/ant1_diag/rosbridge_call.py /olive/olixRobot/ant1/id001/slam_toolbox/reset '{"pause_new_measurements": false}'
#   python3 scripts/ant1_diag/rosbridge_call.py /olive/olixRobot/ant1/id001/controller_server/get_state '{}'
# Robot IP: ANT1_IP env (default 10.66.76.31 = phone hotspot; Ethernet 192.168.7.100).
import asyncio, json, os, sys, websockets
async def main():
    async with websockets.connect(f"ws://{os.environ.get('ANT1_IP', '10.66.76.31')}:5001", open_timeout=10) as ws:
        await ws.send(json.dumps({"op": "call_service", "id": "1", "service": sys.argv[1], "args": json.loads(sys.argv[2])}))
        print(await asyncio.wait_for(ws.recv(), 20))
asyncio.run(main())
