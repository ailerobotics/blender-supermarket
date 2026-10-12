#!/usr/bin/env python3
"""Extract a Jazzy Nav2 overlay into a caller-owned directory, without sudo.

Uses the host's trusted apt indexes and installed-package status to resolve the
overlay. apt --print-uris is read-only; dpkg-deb --extract never runs maintainer
scripts or changes the host package database. Do not source into Isaac Python:
run ROS nodes with the host Jazzy Python ABI in a separate process.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    cache=out/'packages'; cache.mkdir(exist_ok=True)
    root=out/'root'; root.mkdir(exist_ok=True)
    command=['apt-get','--print-uris','--yes','--download-only',
             '-o','Debug::NoLocking=true','-o','Acquire::ForceHash=sha256',
             'install','ros-jazzy-nav2-bringup','ros-jazzy-navigation2']
    result=subprocess.run(command,text=True,capture_output=True,check=True)
    (out/'apt-resolution.txt').write_text(result.stdout)
    packages=[]
    for line in result.stdout.splitlines():
        if not line.startswith("'"):
            continue
        url,name,size,checksum=shlex.split(line)
        algorithm,digest=checksum.split(':',1)
        algorithm={'SHA256':'sha256','MD5Sum':'md5'}.get(algorithm,algorithm.lower())
        packages.append(dict(url=url,name=name,size=int(size),algorithm=algorithm,digest=digest))
    if not packages:
        raise RuntimeError('No overlay packages resolved; use an existing complete ROS environment instead')
    (out/'packages.lock.json').write_text(json.dumps(packages,indent=2)+'\n')

    def download(p):
        dest=cache/p['name']
        if dest.exists() and hashlib.new(p['algorithm'],dest.read_bytes()).hexdigest()==p['digest']:
            return dest
        with urllib.request.urlopen(p['url'],timeout=90) as response:
            data=response.read()
        if len(data)!=p['size'] or hashlib.new(p['algorithm'],data).hexdigest()!=p['digest']:
            raise RuntimeError('apt package hash/size mismatch: '+p['name'])
        dest.write_bytes(data)
        return dest

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        paths=list(pool.map(download,packages))
    for path in paths:
        subprocess.run(['dpkg-deb','--extract',str(path),str(root)],check=True)
    prefix=root/'opt/ros/jazzy'
    env=out/'env.sh'
    env.write_text('''# Generated task-owned ROS overlay. Source only in a host-Python ROS shell.
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
'''+f'export RESTOCK_NAV_ROOT={shlex.quote(str(root))}\n'+'''
export AMENT_PREFIX_PATH="$RESTOCK_NAV_ROOT/opt/ros/jazzy:$AMENT_PREFIX_PATH"
export CMAKE_PREFIX_PATH="$RESTOCK_NAV_ROOT/opt/ros/jazzy:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$RESTOCK_NAV_ROOT/opt/ros/jazzy/lib:$RESTOCK_NAV_ROOT/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$RESTOCK_NAV_ROOT/opt/ros/jazzy/lib/python3.12/site-packages:$RESTOCK_NAV_ROOT/opt/ros/jazzy/local/lib/python3.12/dist-packages:$RESTOCK_NAV_ROOT/usr/lib/python3/dist-packages:${PYTHONPATH:-}"
export PATH="$RESTOCK_NAV_ROOT/opt/ros/jazzy/bin:$PATH"
''')
    print(json.dumps({'overlay':str(prefix),'environment':str(env),'packages':len(packages)}))


if __name__=='__main__':
    main()
