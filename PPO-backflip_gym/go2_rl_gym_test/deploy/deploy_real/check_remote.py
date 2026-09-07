"""Read-only Go2 low-state and wireless-controller check."""

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

from common.remote_controller import KeyMap, RemoteController


KEY_NAMES = {
    KeyMap.start: "START",
    KeyMap.select: "SELECT",
    KeyMap.A: "A",
    KeyMap.B: "B",
    KeyMap.X: "X",
    KeyMap.Y: "Y",
}


def main():
    parser = argparse.ArgumentParser(description="Read-only Go2 controller check")
    parser.add_argument("net", help="robot network interface")
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.net)
    remote = RemoteController()
    received = False
    last_print = 0.0

    def callback(msg):
        nonlocal received, last_print
        received = True
        remote.set(msg.wireless_remote)
        now = time.monotonic()
        if now - last_print < 0.1:
            return
        last_print = now
        pressed = [name for index, name in KEY_NAMES.items() if remote.button[index]]
        print(
            f"tick={msg.tick} buttons={pressed or ['none']} "
            f"lx={remote.lx:+.2f} ly={remote.ly:+.2f} "
            f"rx={remote.rx:+.2f} ry={remote.ry:+.2f}",
            flush=True,
        )

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(callback, 10)
    print("Read-only check: press START, A, SELECT one at a time; Ctrl-C exits.")
    started = time.monotonic()
    try:
        while True:
            if not received and time.monotonic() - started > 10.0:
                raise TimeoutError("10 s 内没有收到 rt/lowstate")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exit: no motor command was sent.")


if __name__ == "__main__":
    main()
