#!/bin/bash
export PATH=/home/hoanganh/.cargo/bin:$PATH
cd "$(dirname "$0")"
cargo furiosa-opt build --release --locked 2>&1 | tail -40
