#!/usr/bin/env python3
import argparse
import select
import socket
import sys


def bind_udp(ip: str, port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((ip, port))
    return s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-listen-ip", default="0.0.0.0")
    parser.add_argument("--robot-listen-port", type=int, default=6510)

    parser.add_argument("--container-listen-ip", default="0.0.0.0")
    parser.add_argument("--container-listen-port", type=int, default=6512)

    parser.add_argument("--container-dst-ip", default="127.0.0.1")
    parser.add_argument("--container-dst-port", type=int, default=6511)

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    robot_sock = bind_udp(args.robot_listen_ip, args.robot_listen_port)
    container_sock = bind_udp(args.container_listen_ip, args.container_listen_port)

    container_dst = (args.container_dst_ip, args.container_dst_port)
    robot_addr = None

    print(f"[relay] robot-facing listen: {args.robot_listen_ip}:{args.robot_listen_port}")
    print(f"[relay] container-facing listen: {args.container_listen_ip}:{args.container_listen_port}")
    print(f"[relay] forwarding robot packets to container: {args.container_dst_ip}:{args.container_dst_port}")

    try:
        while True:
            readable, _, _ = select.select([robot_sock, container_sock], [], [])

            for sock in readable:
                data, addr = sock.recvfrom(65535)

                if sock is robot_sock:
                    robot_addr = addr
                    if args.verbose:
                        print(f"[robot -> container] {addr} -> {container_dst} ({len(data)} bytes)")
                    container_sock.sendto(data, container_dst)

                else:
                    if robot_addr is None:
                        if args.verbose:
                            print("[container -> robot] dropped: robot endpoint unknown")
                        continue
                    if args.verbose:
                        print(f"[container -> robot] {addr} -> {robot_addr} ({len(data)} bytes)")
                    robot_sock.sendto(data, robot_addr)

    except KeyboardInterrupt:
        print("\n[relay] shutting down")
        return 0


if __name__ == "__main__":
    sys.exit(main())