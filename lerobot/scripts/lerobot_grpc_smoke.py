#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pickle  # nosec
import time

import grpc

from lerobot.async_inference.helpers import TimedObservation
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="gRPC Ready + SendObservations smoke test.")
    parser.add_argument("--address", default="127.0.0.1:5555", help="gRPC server address")
    parser.add_argument("--ready-only", action="store_true", help="Only call Ready() and exit")
    args = parser.parse_args()

    channel = grpc.insecure_channel(args.address)
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    try:
        stub.Ready(services_pb2.Empty())
        print("Ready OK")

        if args.ready_only:
            return 0

        dummy_obs = {"dummy_state": [0.0, 1.0, 2.0]}
        timed_obs = TimedObservation(
            timestamp=time.time(),
            observation=dummy_obs,
            timestep=0,
            must_go=True,
        )
        payload = pickle.dumps(timed_obs)  # nosec

        iterator = send_bytes_in_chunks(payload, services_pb2.Observation, silent=True)
        stub.SendObservations(iterator)

        print("SendObservations OK")
        return 0
    except Exception as exc:
        print(f"gRPC FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
