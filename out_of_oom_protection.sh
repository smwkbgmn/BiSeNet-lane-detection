#!/bin/bash
systemd-run --user --scope -p MemoryHigh=infinity -p ManagedOOMPreference=omit ./run_lane_training.sh