#!/usr/bin/env python3
"""Run tiny C concept simulations and independently check the results."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

DEFAULTS = {"event_period": 50, "threshold": 1.0, "bits": 3,
            "phase": 0.0, "overhead_ms": 5.0, "openmp": False}

def validate(s):
    assert set(s) == set(DEFAULTS), "Use the documented settings only"
    for key, lo, hi in [("event_period",1,1000),("bits",2,12)]:
        assert type(s[key]) is int and lo <= s[key] <= hi, key
    for key, lo, hi in [("threshold",0.1,2),("phase",0,2*math.pi),
                        ("overhead_ms",0,1000)]:
        assert type(s[key]) in (int,float) and math.isfinite(s[key]) and lo <= s[key] <= hi, key
    assert type(s["openmp"]) is bool

def command(args, cwd, env=None):
    p = subprocess.run([str(a) for a in args],cwd=cwd,env=env,text=True,
                       capture_output=True,timeout=20)
    if p.returncode:
        raise RuntimeError(f"Command failed ({p.returncode}): {args}\n{p.stdout}\n{p.stderr}")
    return p.stdout

def record(text):
    line = next(x for x in text.splitlines() if x.startswith("RESULT "))
    return {k: float(v) for k,v in (part.split("=",1) for part in line.split()[2:])}

def main(folder, settings):
    validate(settings)
    folder=Path(folder).resolve()
    source=folder/"m6s1_lab.c"
    result={"settings":settings,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "provenance":{"host":platform.node(),"platform":platform.platform(),
             "python":sys.version.split()[0],"slurm_job_id":os.environ.get("SLURM_JOB_ID"),
             "context":"SLURM_CPU" if os.environ.get("SLURM_JOB_ID") else "LOCAL_CPU",
             "flags":"-O2 -std=c11 -Wall -Wextra -Werror -lm"},
            "experiments":{}, "optional":{}, "raw":{}}
    cc=shutil.which("gcc")
    if not cc:
        for name in ("E1","E2","E3","E4"):
            result["experiments"][name]={"status":"SKIPPED","reason":"gcc unavailable. Use site compiler instructions."}
    else:
        result["provenance"]["compiler"]=command([cc,"--version"],folder).splitlines()[0]
        try:
            command([cc,"-O2","-std=c11","-Wall","-Wextra","-Werror",source,"-lm","-o",folder/"m6s1_lab"],folder)
        except Exception as exc:
            for name in ("E1","E2","E3","E4"):
                result["experiments"][name]={"status":"FAIL","reason":str(exc)}
        else:
            for name, mode, values in [
                ("E1","events",[50,100,settings["event_period"]]),
                ("E2","lif",[1.0,1.3,settings["threshold"]]),
                ("E3","matvec",[3,8,settings["bits"]]),
                ("E4","quantum",[0,math.pi/2,math.pi,settings["phase"]])]:
                rows=[]
                try:
                    for value in dict.fromkeys(values):
                        text=command([folder/"m6s1_lab",mode,value],folder)
                        row=record(text)
                        result["raw"][f"{mode}:{value}"]=text
                        if name=="E1":
                            x=[0]*16; expected=0; count=0
                            for t in range(1000):
                                if t%value==0:
                                    x[count%16] += count%3+1
                                    count+=1
                                expected+=sum((i+1)*v for i,v in enumerate(x))
                            assert row["checksum"]==expected and row["events"]==count
                            assert row["event_updates"]==count and row["dense_updates"]==16000
                        elif name=="E2":
                            v=0; spikes=0
                            for _ in range(12):
                                v=.8*v+.3
                                if v>=value: v=0; spikes+=1
                            assert row["spikes"]==spikes and abs(row["final_v"]-v)<1e-8
                        elif name=="E3":
                            levels=2**value-1
                            errors=[]
                            for i in range(4):
                                approximate=sum((math.floor((i+1)*(j+1)/17*levels+.5)/levels)*(j+1)/10 for j in range(4))
                                errors.append(abs(approximate-(i+1)*3/17))
                            assert abs(row["max_error"]-max(errors))<1e-10
                        else:
                            assert abs(row["p1"]-math.sin(value/2)**2)<1e-10
                            assert abs(row["p0"]+row["p1"]-1)<1e-10
                        rows.append(row)
                    result["experiments"][name]={"status":"PASS","evidence":"CPU_SIMULATION","rows":rows}
                except Exception as exc:
                    result["experiments"][name]={"status":"FAIL","reason":str(exc),"rows":rows}
        result["optional"]["OpenMP"]={"status":"NOT RUN","reason":"Set openmp=True and provide two allocated CPUs to request this optional check."}
        if settings["openmp"] and all(result["experiments"][k]["status"]=="PASS" for k in ("E1","E2","E3","E4")):
            allocated=int(os.environ.get("SLURM_CPUS_PER_TASK","1"))
            if os.environ.get("SLURM_JOB_ID") and allocated<2:
                result["optional"]["OpenMP"]={"status":"SKIPPED","reason":"Need two CPUs. Request --cpus-per-task=2."}
            else:
                p=subprocess.run([cc,"-O2","-std=c11","-Wall","-Wextra","-Werror","-fopenmp",str(source),"-lm","-o",str(folder/"m6s1_omp")],capture_output=True,text=True,timeout=20)
                if p.returncode:
                    result["optional"]["OpenMP"]={"status":"SKIPPED","reason":"OpenMP build unavailable", "diagnostic":p.stderr}
                else:
                    try:
                        env=dict(os.environ,OMP_NUM_THREADS="2",OMP_DYNAMIC="FALSE")
                        out=command([folder/"m6s1_omp","matvec",settings["bits"]],folder,env)
                        expected=command([folder/"m6s1_lab","matvec",settings["bits"]],folder)
                        assert record(out)==record(expected), "OpenMP/serial mismatch"
                        result["optional"]["OpenMP"]={"status":"PASS","evidence":"Two-thread correctness only, no timing claim","output":out}
                    except Exception as exc:
                        result["optional"]["OpenMP"]={"status":"FAIL","reason":str(exc)}
    for key in ("GPU","MPI","QPU","NEUROMORPHIC","PHOTONIC","RESISTIVE_ARRAY"):
        result["optional"][key]={"status":"SKIPPED","reason":"No hardware execution requested in this CPU concept lab."}
    if "OpenMP" not in result["optional"]:
        result["optional"]["OpenMP"]={"status":"SKIPPED","reason":"No working C compiler."}
    rows=[]
    for overhead in dict.fromkeys([5.0,80.0,settings["overhead_ms"]]):
        total=20+80/10+overhead
        assert total>=28 and 100/total>0
        rows.append({"overhead_ms":overhead,"hybrid_ms":total,"speedup":100/total})
    result["experiments"]["E5"]={"status":"PASS","evidence":"ILLUSTRATIVE_MODEL_NOT_MEASUREMENT","rows":rows}
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--settings",default="settings.json")
    p.add_argument("--output",default="results.json")
    a=p.parse_args()
    try:
        settings=json.loads(Path(a.settings).read_text())
        result=main(Path.cwd(),settings)
    except Exception as exc:
        print("FAIL:",exc,file=sys.stderr)
        sys.exit(2)
    Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
    for name,entry in result["experiments"].items():
        print(name,entry["status"],entry.get("reason",entry.get("evidence")))
    for name,entry in result["optional"].items(): print(name,entry["status"])
    sys.exit(1 if any(e["status"]=="FAIL" for e in list(result["experiments"].values())+list(result["optional"].values())) else 0)
