#!/bin/bash
set -eu

port=5000

lsof -ti:$port | xargs kill -9
